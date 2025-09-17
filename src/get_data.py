from torchvision import datasets, transforms
from src.plot import plot_floats_and_spikes, plot_audio_spectrograms_and_spikes
import librosa
import gammatone.gtgram
from snntorch import spikegen
import numpy as np
import torch
from tqdm import tqdm
import warnings
import os

os.environ["TQDM_DISABLE"] = "True"


def normalize_image(img, target_sum=1.0):
    current_sum = img.sum()
    return img * (target_sum / current_sum) if current_sum > 0 else img


warnings.filterwarnings("ignore")


def _mel_filterbank_envelopes(
    x, sr, n_ch=40, win_ms=25, hop_ms=10, fmin=50.0, fmax=None
):
    """
    Fallback cochlear-ish features using mel filterbank magnitudes + smoothing.
    Returns envelopes (F x T) and frame_dt (seconds).
    """
    if fmax is None:
        fmax = sr / 2.0
    n_fft = int(round(sr * win_ms / 1000.0))
    hop_length = int(round(sr * hop_ms / 1000.0))
    S = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_mels=n_ch,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,  # magnitude, not power^2
    )  # (F, T)
    # Simple envelope smoothing across time per channel
    # Convert each channel’s frame series to a “continuous-ish” envelope by smoothing along time
    # Here: a lightweight moving-average via IIR in frame-domain
    F, T = S.shape
    E = np.empty_like(S)
    alpha = np.exp(-1.0 / 3.0)  # ~3-frame time constant
    for f in range(F):
        acc = 0.0
        for t in range(T):
            acc = (1.0 - alpha) * S[f, t] + alpha * acc
            E[f, t] = acc
    frame_dt = hop_length / float(sr)
    return E, frame_dt


def _gammatone_envelopes(x, sr, n_ch=40, win_ms=25, hop_ms=10, fmin=50.0):
    """
    If 'gammatone' is installed, use its gtgram for a more auditory cochleagram.
    Returns envelopes (F x T) and frame_dt (seconds).
    """
    try:
        import gammatone.gtgram as gtg
    except Exception:
        return None, None
    win_time = win_ms / 1000.0
    hop_time = hop_ms / 1000.0
    G = gtg.gtgram(
        x, sr, window_time=win_time, hop_time=hop_time, channels=n_ch, f_min=fmin
    )
    # G is positive; light temporal smoothing
    F, T = G.shape
    E = np.empty_like(G)
    alpha = np.exp(-1.0 / 3.0)
    for f in range(F):
        acc = 0.0
        for t in range(T):
            acc = (1.0 - alpha) * G[f, t] + alpha * acc
            E[f, t] = acc
    frame_dt = hop_time
    return E, frame_dt


def cochlear_to_spikes_1s(
    wav_batch,  # list of 1D float32 arrays, variable length
    sr=22050,
    n_channels=40,
    win_ms=25,
    hop_ms=10,
    target_max_rate_hz=100.0,  # upper bound for an active channel
    env_cutoff_hz=120.0,  # used in raw-envelope path; mel/gt already smoothed
    out_T_ms=1000,  # exactly 1000 steps
    eps=1e-8,
    return_rates=False,
    show_progress=False,
    progress_desc="Encoding",
    debug=False,
):
    """
    Full pipeline:
      raw -> (gammatone|mel) envelopes -> CMVN -> sigmoid map -> resample to 1000 steps (1 ms) -> area-match -> Poisson.
    Guarantees: every file uses *all* of its content (global time-warp), output length = 1000.
    """
    B = len(wav_batch)
    T_out = int(out_T_ms)  # 1000
    dt_out = 1e-3  # 1 ms

    all_spikes, all_rates = [], []

    # Optionally show per-file progress
    if show_progress:
        # Ensure tqdm is enabled for this call
        try:
            os.environ["TQDM_DISABLE"] = "False"
        except Exception:
            pass
        iterator = tqdm(
            wav_batch,
            total=B,
            desc=progress_desc,
            leave=False,
        )
    else:
        iterator = wav_batch

    for x in iterator:
        # --- Cochlear front-end (prefer gammatone)
        E, frame_dt = _gammatone_envelopes(
            x, sr, n_ch=n_channels, win_ms=win_ms, hop_ms=hop_ms
        )
        if E is None:  # fallback: mel
            E, frame_dt = _mel_filterbank_envelopes(
                x, sr, n_ch=n_channels, win_ms=win_ms, hop_ms=hop_ms
            )

        # --- Log compression + per-channel CMVN
        # add small bias to avoid log(0)
        X = np.log1p(np.maximum(E, 0.0))
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + eps)

        # --- Map to [0, target_max_rate] with a smooth nonlinearity
        rates_hz_frames = (
            1.0 / (1.0 + np.exp(-X))
        ) * target_max_rate_hz  # (F, T_frames)

        # --- Time-normalize: resample from original length to 1000 steps (use *all* audio)
        F, T_frames = rates_hz_frames.shape
        t_old = np.arange(T_frames) * frame_dt
        if T_frames == 1:
            # Degenerate 1-frame case: tile
            rates_resamp = np.tile(rates_hz_frames[:, 0:1], (1, T_out))
        else:
            t_new = np.linspace(0.0, t_old[-1], T_out)  # covers full utterance
            rates_resamp = np.empty((F, T_out), dtype=np.float32)
            for f in range(F):
                rates_resamp[f] = np.interp(t_new, t_old, rates_hz_frames[f])

        # --- Area match: preserve expected spike count across the time-warp
        # Ensure sum(rate*dt) invariant before/after resample (per channel)
        area_old = (rates_hz_frames * frame_dt).sum(axis=1, keepdims=True)  # (F,1)
        area_new = (rates_resamp * dt_out).sum(axis=1, keepdims=True) + eps  # (F,1)
        scale = np.where(area_new > 0, area_old / area_new, 1.0)
        rates_resamp *= scale

        # --- Poisson sampling at 1 ms with exact Bernoulli thinning
        # p = 1 - exp(-rate * dt)
        p = 1.0 - np.exp(-np.clip(rates_resamp, 0.0, None) * dt_out)  # (F, 1000)
        spikes = (np.random.rand(F, T_out) < p).astype(np.int8).T  # (1000, F)
        all_spikes.append(spikes)
        if return_rates:
            all_rates.append(rates_resamp.T)  # (1000, F)

    spikes_bt = np.stack(all_spikes, axis=0)  # (B, 1000, F)
    if return_rates:
        rates_bt = np.stack(all_rates, axis=0)  # (B, 1000, F)
        return spikes_bt, rates_bt
    return spikes_bt


class ImageDataStreamer:
    """
    Streams image data in batches to avoid loading all files at once.
    Only loads and processes small batches as needed.
    """

    def __init__(
        self,
        data_dir,
        batch_size=100,
        pixel_size=15,
        num_steps=100,
        gain=1.0,
        offset=0,
        first_spike_time=0,
        time_var_input=False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.num_steps = num_steps
        self.gain = gain
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input

        # Load MNIST dataset
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((pixel_size, pixel_size)),
                transforms.ToTensor(),
            ]
        )

        self.mnist_train = datasets.MNIST(
            root=data_dir, train=True, transform=transform, download=True
        )
        self.mnist_test = datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=True
        )

        # Combine train and test data
        self.all_images = []
        self.all_labels = []

        for i in range(len(self.mnist_train)):
            self.all_images.append(self.mnist_train[i][0])
            self.all_labels.append(self.mnist_train[i][1])

        for i in range(len(self.mnist_test)):
            self.all_images.append(self.mnist_test[i][0])
            self.all_labels.append(self.mnist_test[i][1])

        self.all_images = torch.stack(self.all_images)
        self.all_labels = np.array(self.all_labels)

        # Shuffle indices for random access
        self.indices = np.arange(len(self.all_images))
        np.random.shuffle(self.indices)

        print(f"Found {len(self.all_images)} image samples")
        print(f"Label distribution: {np.bincount(self.all_labels)}")

    def get_batch(self, start_idx, num_samples):
        """
        Load a batch of image samples starting from start_idx.
        Returns (spike_data, labels) or (None, None) if no more data.
        """
        if start_idx >= len(self.all_images):
            return None, None

        end_idx = min(start_idx + num_samples, len(self.all_images))
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        batch_images = self.all_images[batch_indices]
        batch_labels = self.all_labels[batch_indices]

        # Convert to spikes
        spike_data = self._convert_images_to_spikes(batch_images)

        return spike_data, batch_labels

    def _convert_images_to_spikes(self, images):
        """Convert image batch to spikes."""
        # Normalize images
        target_sum = (self.pixel_size**2) * 0.1
        norm_images = torch.stack(
            [normalize_image(img=img, target_sum=target_sum) for img in images]
        )

        # Convert to spikes
        S_data = torch.zeros(size=norm_images.shape)
        S_data = S_data.repeat(self.num_steps, 1, 1, 1)

        for i in range(norm_images.shape[0]):
            S_data[i * self.num_steps : (i + 1) * self.num_steps] = spikegen.rate(
                norm_images[i],
                num_steps=self.num_steps,
                gain=self.gain,
                offset=self.offset,
                first_spike_time=self.first_spike_time,
                time_var_input=self.time_var_input,
            )

        # Reshape to match expected format
        S_data_corrected = S_data.squeeze(2).flatten(start_dim=2)
        S_data_reshaped = np.reshape(
            S_data_corrected,
            (
                S_data.shape[0] * S_data.shape[1],
                S_data_corrected.shape[2],
            ),
        )

        return S_data_reshaped.numpy()

    def get_total_samples(self):
        """Return total number of available samples."""
        return len(self.all_images)


class AudioDataStreamer:
    """
    Streams audio in batches without truncating/padding.
    You get variable-length waveforms; the encoder will handle time-normalization.
    """

    def __init__(self, data_path, target_sr=22050, batch_size=100):
        self.data_path = data_path
        self.target_sr = target_sr
        self.batch_size = batch_size

        self.audio_files, self.audio_labels = [], []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path {data_path} does not exist!")

        for root, _, files in os.walk(data_path):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue
                fp = os.path.join(root, file)
                # Expect filenames like "3_...wav"; adapt if yours differ
                label = int(file[0]) if file[0].isdigit() else None
                if label is not None and 0 <= label <= 9:
                    self.audio_files.append(fp)
                    self.audio_labels.append(label)

        self.indices = np.arange(len(self.audio_files))
        np.random.shuffle(self.indices)
        print(f"Found {len(self.audio_files)} audio files")
        print(f"Label distribution: {np.bincount(self.audio_labels)}")

    def get_batch(self, start_idx, num_samples):
        if start_idx >= len(self.audio_files):
            return None, None
        end_idx = min(start_idx + num_samples, len(self.audio_files))
        idxs = self.indices[start_idx:end_idx]

        audio_batch, labels_batch = [], []
        for i in idxs:
            fp = self.audio_files[i]
            try:
                # Load full file at target_sr (no duration cap)
                x, sr = librosa.load(fp, sr=self.target_sr, mono=True)
                audio_batch.append(x.astype(np.float32))
                labels_batch.append(self.audio_labels[i])
            except Exception as e:
                print(f"Error loading {fp}: {e}")
        if not audio_batch:
            return None, None
        return audio_batch, np.array(labels_batch, dtype=np.int64)

    def get_total_samples(self):
        return len(self.audio_files)


def load_audiomnist_data(data_path, target_sr=22050):
    """
    Loads all AudioMNIST into memory (if you really want that).
    Fixed: returns properly stacked padded data if you *choose* to pad.
    """
    all_data, all_labels = [], []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path {data_path} does not exist!")

    for root, _, files in tqdm(list(os.walk(data_path)), desc="Loading AudioMNIST"):
        for file in files:
            if not file.endswith(".wav"):
                continue
            fp = os.path.join(root, file)
            try:
                x, sr = librosa.load(fp, sr=target_sr, mono=True)
                label = int(file[0]) if file[0].isdigit() else None
                if label is not None and 0 <= label <= 9:
                    all_data.append(x.astype(np.float32))
                    all_labels.append(label)
            except Exception as e:
                print(f"Error loading {fp}: {e}")

    if not all_data:
        return None, None

    # Optional: pad to the max length (not used downstream if you use the encoder below)
    max_len = max(len(x) for x in all_data)
    padded = [np.pad(x, (0, max_len - len(x))) for x in all_data]
    data_array = np.stack(padded, axis=0)
    labels_array = np.array(all_labels, dtype=np.int64)
    print(f"Audio shape: {data_array.shape}  Labels shape: {labels_array.shape}")
    print(f"Label distribution: {np.bincount(labels_array)}")
    return data_array, labels_array


def load_audio_batch(
    audio_streamer,
    start_idx,
    batch_size,
    num_steps,
    num_input_neurons,
    plot_spectrograms=False,
    return_rates=False,
):
    """
    Load a batch of audio data and convert to spikes on-demand.

    Args:
        audio_streamer: AudioDataStreamer instance
        start_idx: Starting index for the batch
        batch_size: Number of samples to load
        num_steps: Number of time steps for spike generation
        num_input_neurons: Number of input neurons (frequency bands)

    Returns:
        (spike_data, labels) or (None, None) if no more data
    """
    # Load audio batch
    audio_batch, labels = audio_streamer.get_batch(start_idx, batch_size)

    if audio_batch is None:
        return None, None

    # Convert to spikes
    spike_data_3d = cochlear_to_spikes_1s(
        audio_batch,
        sr=getattr(audio_streamer, "target_sr", 22050),
        n_channels=int(num_input_neurons),
        win_ms=25,
        hop_ms=10,
        target_max_rate_hz=100.0,
        env_cutoff_hz=120.0,
        out_T_ms=num_steps,
        eps=1e-8,
    )  # (B, num_steps, num_input_neurons)

    # Flatten batch and time to match training pipeline expectations: (B*T, F)
    spike_data = spike_data_3d.reshape(-1, spike_data_3d.shape[-1])

    # Extend labels to match spike data (repeat each label for num_steps)
    spike_labels = labels.repeat(num_steps)

    # Optional visualization of spectrograms + spikes for this batch
    if plot_spectrograms:
        try:
            plot_audio_spectrograms_and_spikes(
                audio_data=audio_batch,
                spikes=spike_data,
                spike_labels=spike_labels,
                audio_labels=labels,
                num_steps=num_steps,
                sample_rate=getattr(audio_streamer, "target_sr", 22050),
            )
        except Exception as e:
            print(f"Warning: failed to plot spectrograms for this batch: {e}")

    return spike_data, spike_labels


def load_image_batch(
    image_streamer,
    start_idx,
    batch_size,
    num_steps,
    num_input_neurons,
):
    """
    Load a batch of image data and convert to spikes on-demand.

    Args:
        image_streamer: ImageDataStreamer instance
        start_idx: Starting index for the batch
        batch_size: Number of samples to load
        num_steps: Number of time steps for spike generation
        num_input_neurons: Number of input neurons (pixel size squared)

    Returns:
        (spike_data, labels) or (None, None) if no more data
    """
    # Load image batch
    spike_data, labels = image_streamer.get_batch(start_idx, batch_size)

    if spike_data is None:
        return None, None

    # Extend labels to match spike data (repeat each label for num_steps)
    spike_labels = labels.repeat(num_steps)

    return spike_data, spike_labels


def example_audio_streaming_usage():
    """
    Example of how to use the streaming audio data loader.
    This shows how to load audio data in small batches during training.
    """
    # Initialize audio streamer
    data_path = "/home/andreas/Documents/GitHub/AudioMNIST/data"
    audio_streamer = AudioDataStreamer(data_path, batch_size=50)

    # Example: Load batches during training loop
    batch_size = 50
    num_steps = 100
    num_input_neurons = 225  # 15x15 for example

    for epoch in range(10):  # 10 epochs
        print(f"Epoch {epoch + 1}")

        # Load batches sequentially
        start_idx = 0
        batch_num = 0

        while True:
            # Load audio batch and convert to spikes
            spike_data, labels = load_audio_batch(
                audio_streamer,
                start_idx,
                batch_size,
                num_steps,
                num_input_neurons,
                scaling_method="normalize",
            )

            if spike_data is None:  # No more data
                break

            print(
                f"  Batch {batch_num}: {spike_data.shape[0]} samples, {spike_data.shape[1]} neurons"
            )

            # Here you would train your SNN with spike_data and labels
            # train_snn_batch(spike_data, labels)

            start_idx += batch_size
            batch_num += 1

            # Optional: Break after a certain number of batches for demo
            if batch_num >= 5:  # Only process 5 batches per epoch for demo
                break

        print(f"  Processed {batch_num} batches in epoch {epoch + 1}")


def create_balanced_splits(
    data,
    labels,
    max_total_samples=30000,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    """
    Create balanced train/validation/test splits maintaining fixed ratios
    Ensures equal representation of each class (0-9) in each split
    """
    # Verify ratios sum to 1
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution before splitting: {dict(zip(unique_labels, counts))}")

    # Calculate actual sample counts based on available data and max limit
    available_samples = len(data)
    actual_max_samples = min(available_samples, max_total_samples)

    # Calculate samples per class based on actual available data
    samples_per_class = actual_max_samples // len(unique_labels)

    # Calculate split sizes based on ratios
    train_samples = int(actual_max_samples * train_ratio)
    val_samples = int(actual_max_samples * val_ratio)
    test_samples = (
        actual_max_samples - train_samples - val_samples
    )  # Ensure exact total

    # Calculate samples per class for each split
    train_per_class = train_samples // len(unique_labels)
    val_per_class = val_samples // len(unique_labels)
    test_per_class = test_samples // len(unique_labels)

    # First, limit total data to actual_max_samples if we have more
    if len(data) > actual_max_samples:
        # Sample equally from each class
        limited_data, limited_labels = [], []
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            np.random.shuffle(class_indices)
            # Take up to samples_per_class from each class
            selected_indices = class_indices[
                : min(samples_per_class, len(class_indices))
            ]
            limited_data.extend(data[selected_indices])
            limited_labels.extend(labels[selected_indices])

        data = np.array(limited_data)
        labels = np.array(limited_labels)
        print(f"Limited data to {len(data)} samples total")

    # Split data for each class
    train_data, train_labels = [], []
    val_data, val_labels = [], []
    test_data, test_labels = [], []

    for label in unique_labels:
        # Get indices for this class
        class_indices = np.where(labels == label)[0]
        np.random.shuffle(class_indices)

        # Split indices
        train_end = min(train_per_class, len(class_indices))
        val_end = train_end + min(val_per_class, len(class_indices) - train_end)
        test_end = val_end + min(test_per_class, len(class_indices) - val_end)

        train_indices = class_indices[:train_end]
        val_indices = class_indices[train_end:val_end]
        test_indices = class_indices[val_end:test_end]

        # Add to respective arrays
        train_data.extend(data[train_indices])
        train_labels.extend(labels[train_indices])
        val_data.extend(data[val_indices])
        val_labels.extend(labels[val_indices])
        test_data.extend(data[test_indices])
        test_labels.extend(labels[test_indices])

    # Convert back to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle each split
    train_indices = np.random.permutation(len(train_data))
    val_indices = np.random.permutation(len(val_data))
    test_indices = np.random.permutation(len(test_data))

    # Assign indices
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]
    val_data = val_data[val_indices]
    val_labels = val_labels[val_indices]
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]

    print(f"\nFinal split sizes:")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def combine_audio_image_data(audio_data, audio_labels, image_data, image_labels):
    """
    Combine audio and image data ensuring equal sample counts
    """
    # Ensure both have same number of samples
    min_samples = min(len(audio_data), len(image_data))

    # Truncate to minimum length
    audio_data = audio_data[:min_samples]
    audio_labels = audio_labels[:min_samples]
    image_data = image_data[:min_samples]
    image_labels = image_labels[:min_samples]

    print(f"Combined data: {min_samples} samples each for audio and image")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Image shape: {image_data.shape}")

    return audio_data, audio_labels, image_data, image_labels


def create_data(
    pixel_size,
    num_steps,
    gain,
    offset,
    first_spike_time,
    time_var_input,
    num_images_train,
    num_images_test,
    add_breaks,
    break_lengths,
    noisy_data,
    noise_level,
    plot_comparison,
    download,
    data_dir,
    use_validation_data=False,
    idx_train=0,
    idx_val=0,
    idx_test=0,
    val_split=0.2,
    train_split=0.6,
    test_split=0.2,
    audioMNIST=True,
    imageMNIST=False,
):
    """
    Optionally splits off a validation set from the training data if use_validation_data is True.
    Returns (train_data, train_labels, test_data, test_labels) as before, or
    (train_data, train_labels, val_data, val_labels, test_data, test_labels) if validation is enabled.
    When validation is enabled, the split is persistent and batches are indexed from the training set only.
    """

    # Initialize variables
    audio_train_data, audio_train_labels = None, None
    audio_val_data, audio_val_labels = None, None
    audio_test_data, audio_test_labels = None, None
    images_train, labels_train = None, None
    images_val, labels_val = None, None
    images_test, labels_test = None, None

    if audioMNIST:
        # Define data path
        data_path = "/home/andreas/Documents/GitHub/AudioMNIST/data"

        # Create audio streamer for efficient batch loading
        audio_streamer = AudioDataStreamer(data_path, batch_size=100)

        if audio_streamer.get_total_samples() > 0:
            print(
                f"Audio streamer initialized with {audio_streamer.get_total_samples()} total samples"
            )
            print("Audio data will be loaded in batches during training/testing")

            # Store streamer for later use instead of pre-loaded data
            audio_train_data = audio_streamer
            audio_train_labels = None  # Will be loaded per batch
            audio_val_data = audio_streamer
            audio_val_labels = None
            audio_test_data = audio_streamer
            audio_test_labels = None

    if imageMNIST:
        # set transform rule
        transform = transforms.Compose(
            [
                transforms.Grayscale(),  # Ensure single channel
                transforms.Resize((pixel_size)),  # Resize to specified pixels
                transforms.ToTensor(),  # Convert to tensor
            ]
        )

        # Fetch MNIST dataset
        mnist_train = datasets.MNIST(
            root=data_dir, train=True, transform=transform, download=download
        )
        mnist_test = datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=download
        )

        # Extract data and limit to same sample counts as audio
        np.random.seed(42)

        # Combine train and test data to create balanced splits
        all_samples = []
        all_labels = []

        # Add training samples
        for i in range(len(mnist_train)):
            all_samples.append(mnist_train[i][0])
            all_labels.append(mnist_train[i][1])

        # Add test samples
        for i in range(len(mnist_test)):
            all_samples.append(mnist_test[i][0])
            all_labels.append(mnist_test[i][1])

        # Convert to numpy arrays for processing
        all_images = torch.stack(all_samples)
        all_labels_array = np.array(all_labels)

        # Create balanced splits with same parameters as audio
        (
            (image_train_data, image_train_labels),
            (image_val_data, image_val_labels),
            (image_test_data, image_test_labels),
        ) = create_balanced_splits(
            all_images.numpy(),
            all_labels_array,
            max_total_samples=30000,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        # Convert back to tensors
        images_train = torch.tensor(image_train_data)
        labels_train = torch.tensor(image_train_labels)
        images_val = torch.tensor(image_val_data)
        labels_val = torch.tensor(image_val_labels)
        images_test = torch.tensor(image_test_data)
        labels_test = torch.tensor(image_test_labels)

        print(f"Image data loaded with equal sample counts:")
        print(f"Train: {len(images_train)} samples")
        print(f"Validation: {len(images_val)} samples")
        print(f"Test: {len(images_test)} samples")

    # Ensure equal sample counts between audio and image data if both are loaded
    if (
        audioMNIST
        and imageMNIST
        and audio_train_data is not None
        and images_train is not None
    ):
        print("\nEnsuring equal sample counts between audio and image data...")

        # Find minimum sample count across all splits
        min_train = min(len(audio_train_data), len(images_train))
        min_val = min(len(audio_val_data), len(images_val))
        min_test = min(len(audio_test_data), len(images_test))

        # Truncate to minimum counts
        audio_train_data = audio_train_data[:min_train]
        audio_train_labels = audio_train_labels[:min_train]
        audio_val_data = audio_val_data[:min_val]
        audio_val_labels = audio_val_labels[:min_val]
        audio_test_data = audio_test_data[:min_test]
        audio_test_labels = audio_test_labels[:min_test]

        images_train = images_train[:min_train]
        labels_train = labels_train[:min_train]
        images_val = images_val[:min_val]
        labels_val = labels_val[:min_val]
        images_test = images_test[:min_test]
        labels_test = labels_test[:min_test]

        print(f"Final equal sample counts:")
        print(f"Train: {len(audio_train_data)} audio, {len(images_train)} image")
        print(f"Validation: {len(audio_val_data)} audio, {len(images_val)} image")
        print(f"Test: {len(audio_test_data)} audio, {len(images_test)} image")

    # Check if we have any data to process
    if images_train is None and audio_train_data is None:
        print("Error: No data loaded!")
        return None, None, None, None

    # Process image data if available
    if images_train is not None:
        # Normalize spike intensity for each image
        target_sum = (pixel_size**2) * 0.1
        norm_images_train = torch.stack(
            [normalize_image(img=img, target_sum=target_sum) for img in images_train]
        )
        norm_images_test = torch.stack(
            [normalize_image(img=img, target_sum=target_sum) for img in images_test]
        )
        if use_validation_data and images_val is not None:
            norm_images_val = torch.stack(
                [normalize_image(img=img, target_sum=target_sum) for img in images_val]
            )
    else:
        # If no image data, create dummy data for compatibility
        print("No image data available, creating dummy data for compatibility")
        norm_images_train = torch.zeros((1, 1, pixel_size, pixel_size))
        norm_images_test = torch.zeros((1, 1, pixel_size, pixel_size))
        if use_validation_data:
            norm_images_val = torch.zeros((1, 1, pixel_size, pixel_size))
        labels_train = torch.zeros(1, dtype=torch.long)
        labels_test = torch.zeros(1, dtype=torch.long)
        if use_validation_data:
            labels_val = torch.zeros(1, dtype=torch.long)

        # Convert floats to poisson sequences
        S_data_train = torch.zeros(size=norm_images_train.shape)
        S_data_train = S_data_train.repeat(num_steps, 1, 1, 1)
        for i in range(norm_images_train.shape[0]):
            S_data_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                norm_images_train[i],
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )
        S_data_test = torch.zeros(size=norm_images_test.shape)
        S_data_test = S_data_test.repeat(num_steps, 1, 1, 1)
        for i in range(norm_images_test.shape[0]):
            S_data_test[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                norm_images_test[i],
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )
        if use_validation_data:
            S_data_val = torch.zeros(size=norm_images_val.shape)
            S_data_val = S_data_val.repeat(num_steps, 1, 1, 1)
            for i in range(norm_images_val.shape[0]):
                S_data_val[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                    norm_images_val[i],
                    num_steps=num_steps,
                    gain=gain,
                    offset=offset,
                    first_spike_time=first_spike_time,
                    time_var_input=time_var_input,
                )

    # Extend labels based on num_steps
    spike_labels_train = labels_train.numpy().repeat(num_steps)
    spike_labels_test = labels_test.numpy().repeat(num_steps)
    if use_validation_data:
        spike_labels_val = labels_val.numpy().repeat(num_steps)

    # Remove unnecessary dimensions
    S_data_train_corrected = S_data_train.squeeze(2).flatten(start_dim=2)
    S_data_test_corrected = S_data_test.squeeze(2).flatten(start_dim=2)
    if use_validation_data:
        S_data_val_corrected = S_data_val.squeeze(2).flatten(start_dim=2)

    # Merge second and first dimensions
    S_data_train = np.reshape(
        S_data_train_corrected,
        (
            S_data_train.shape[0] * S_data_train.shape[1],
            S_data_train_corrected.shape[2],
        ),
    )
    S_data_test = np.reshape(
        S_data_test_corrected,
        (
            S_data_test.shape[0] * S_data_test.shape[1],
            S_data_test_corrected.shape[2],
        ),
    )
    if use_validation_data:
        S_data_val = np.reshape(
            S_data_val_corrected,
            (
                S_data_val.shape[0] * S_data_val.shape[1],
                S_data_val_corrected.shape[2],
            ),
        )

    S_data_train = S_data_train.numpy()
    S_data_test = S_data_test.numpy()
    if use_validation_data:
        S_data_val = S_data_val.numpy()

    if plot_comparison:
        plot_floats_and_spikes(
            images_train, S_data_train, spike_labels_train, labels_train, num_steps
        )

    def add_breaks_to_data(
        S_data, spike_labels, num_images, num_steps, pixel_size, break_lengths
    ):
        data_list = []
        labels_list = []
        for img in range(num_images):
            image_slice = S_data[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels[img * num_steps : (img + 1) * num_steps]
            data_list.append(image_slice)
            labels_list.append(label_slice)
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)
            data_list.append(break_activity)
            labels_list.append(break_labels)
        S_data_out = np.concatenate(data_list, axis=0)
        spike_labels_out = np.concatenate(labels_list, axis=0)
        return S_data_out, spike_labels_out

    if add_breaks:
        S_data_train, spike_labels_train = add_breaks_to_data(
            S_data_train,
            spike_labels_train,
            num_images_train,
            num_steps,
            pixel_size,
            break_lengths,
        )
        S_data_test, spike_labels_test = add_breaks_to_data(
            S_data_test,
            spike_labels_test,
            num_images_test,
            num_steps,
            pixel_size,
            break_lengths,
        )
        if use_validation_data:
            S_data_val, spike_labels_val = add_breaks_to_data(
                S_data_val,
                spike_labels_val,
                num_images_test,
                num_steps,
                pixel_size,
                break_lengths,
            )

    if noisy_data:
        S_data_train = S_data_train.astype(int)
        break_activity = (
            np.random.random(size=S_data_train.shape) < noise_level
        ).astype(int)
        S_data_train = S_data_train | break_activity

    # Print final data statistics
    print(f"\n=== Final Data Summary ===")
    print(f"Training data: {len(S_data_train)} samples")
    print(f"Test data: {len(S_data_test)} samples")
    if use_validation_data:
        print(f"Validation data: {len(S_data_val)} samples")

    if audioMNIST and audio_train_data is not None:
        print(
            f"Audio data included: {len(audio_train_data)} train, {len(audio_val_data) if audio_val_data is not None else 0} val, {len(audio_test_data)} test"
        )

    if imageMNIST and images_train is not None:
        print(
            f"Image data included: {len(images_train)} train, {len(images_val) if images_val is not None else 0} val, {len(images_test)} test"
        )

    # Handle audio streaming data
    if audioMNIST and audio_train_data is not None:
        print(
            f"Audio data: Streaming mode (total samples: {audio_train_data.get_total_samples()})"
        )
        print("Audio data will be loaded in batches during training/testing")

    # Handle image data
    if imageMNIST and images_train is not None:
        print(
            f"Image data: {len(images_train)} train, {len(images_val) if images_val is not None else 0} val, {len(images_test)} test"
        )
    print(f"Data shape: {S_data_train.shape}")
    print(f"Labels shape: {spike_labels_train.shape}")

    print("=" * 30)

    if use_validation_data:
        return (
            S_data_val,
            spike_labels_val,
            S_data_test,
            spike_labels_test,
        )
    else:
        return (
            S_data_train,
            spike_labels_train,
            S_data_test,
            spike_labels_test,
        )
