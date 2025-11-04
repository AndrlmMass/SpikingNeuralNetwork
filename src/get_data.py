from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
import torch.nn.functional as F
from snntorch import spikegen
import librosa
from tqdm import tqdm
import warnings
import numpy as np
import torch
import os
import json

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
    target_max_rate_hz=150.0,  # upper bound for an active channel (balanced for audio-only networks)
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
        train_count=None,
        val_count=None,
        test_count=None,
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

        # Partition indices for train/val/test if counts provided
        total = len(self.indices)
        tc = train_count or total
        vc = val_count or 0
        tec = test_count or 0
        tc = min(tc, total)
        vc = min(vc, max(0, total - tc))
        tec = min(tec, max(0, total - tc - vc))

        self.train_indices = self.indices[:tc]
        self.val_indices = self.indices[tc : tc + vc]
        self.test_indices = self.indices[tc + vc : tc + vc + tec]
        self.ptr_train = 0
        self.ptr_val = 0
        self.ptr_test = 0

        print(f"Found {len(self.all_images)} image samples")
        print(f"Label distribution: {np.bincount(self.all_labels)}")

    def get_batch(self, start_idx, num_samples, partition="train"):
        """
        Load a batch of image samples from the specified partition.
        Returns (spike_data, labels) or (None, None) if no more data.
        """
        if partition == "train":
            pool = self.train_indices
            ptr = self.ptr_train
        elif partition == "val":
            pool = self.val_indices
            ptr = self.ptr_val
        else:
            pool = self.test_indices
            ptr = self.ptr_test

        if ptr >= len(pool):
            return None, None

        end_ptr = min(ptr + num_samples, len(pool))
        batch_indices = pool[ptr:end_ptr]

        # Get batch data
        batch_images = self.all_images[batch_indices]
        batch_labels = self.all_labels[batch_indices]

        # Convert to spikes
        spike_data = self._convert_images_to_spikes(batch_images)

        # Extend labels to match spike data (repeat each label for num_steps)
        extended_labels = batch_labels.repeat(self.num_steps)

        # advance pointer
        if partition == "train":
            self.ptr_train = end_ptr
        elif partition == "val":
            self.ptr_val = end_ptr
        else:
            self.ptr_test = end_ptr

        return spike_data, extended_labels

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
        """Return total number of available training samples."""
        return len(self.train_indices)

    def reset_partition(self, partition="all"):
        if partition in ("all", "train"):
            self.ptr_train = 0
        if partition in ("all", "val"):
            self.ptr_val = 0
        if partition in ("all", "test"):
            self.ptr_test = 0


class AudioDataStreamer:
    """
    Streams audio in batches without truncating/padding.
    You get variable-length waveforms; the encoder will handle time-normalization.
    """

    def __init__(
        self,
        data_path,
        target_sr=22050,
        batch_size=100,
        train_count=None,
        val_count=None,
        test_count=None,
    ):
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

        # store labels as numpy array for fast fancy indexing
        self.audio_labels = np.asarray(self.audio_labels, dtype=np.int64)
        self.indices = np.arange(len(self.audio_files))
        np.random.shuffle(self.indices)
        total = len(self.indices)
        tc = train_count or total
        vc = val_count or 0
        tec = test_count or 0
        tc = min(tc, total)
        vc = min(vc, max(0, total - tc))
        tec = min(tec, max(0, total - tc - vc))
        self.train_indices = self.indices[:tc]
        self.val_indices = self.indices[tc : tc + vc]
        self.test_indices = self.indices[tc + vc : tc + vc + tec]
        self.ptr_train = 0
        self.ptr_val = 0
        self.ptr_test = 0
        print(f"Found {len(self.audio_files)} audio files")
        print(f"Label distribution: {np.bincount(self.audio_labels)}")

    def get_batch(self, start_idx, num_samples, partition="train"):
        if partition == "train":
            pool = self.train_indices
            ptr = self.ptr_train
        elif partition == "val":
            pool = self.val_indices
            ptr = self.ptr_val
        else:
            pool = self.test_indices
            ptr = self.ptr_test
        if ptr >= len(pool):
            return None, None
        end_ptr = min(ptr + num_samples, len(pool))
        idxs = pool[ptr:end_ptr]

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
        # advance pointer
        if partition == "train":
            self.ptr_train = end_ptr
        elif partition == "val":
            self.ptr_val = end_ptr
        else:
            self.ptr_test = end_ptr
        return audio_batch, np.array(labels_batch, dtype=np.int64)

    def get_total_samples(self):
        return len(self.train_indices)

    def reset_partition(self, partition="all"):
        if partition in ("all", "train"):
            self.ptr_train = 0
        if partition in ("all", "val"):
            self.ptr_val = 0
        if partition in ("all", "test"):
            self.ptr_test = 0


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
        print("No valid audio files found!")
        return None, None

    # Convert to numpy arrays
    # Pad/truncate all audio to same length
    max_length = max(len(audio) for audio in all_data)
    padded_data = []

    for audio in all_data:
        if len(audio) < max_length:
            # Pad with zeros
            padded_audio = np.pad(audio, (0, max_length - len(audio)), "constant")
        else:
            # Truncate
            padded_audio = audio[:max_length]
        padded_data.append(padded_audio)

    data_array = np.array(padded_data)
    labels_array = np.array(all_labels)

    print(f"Loaded {len(data_array)} samples")
    print(f"Audio shape: {data_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
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
    partition="train",
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
    audio_batch, labels = audio_streamer.get_batch(
        start_idx, batch_size, partition=partition
    )

    if audio_batch is None:
        return None, None

    # Convert to spikes
    spike_data_3d = cochlear_to_spikes_1s(
        audio_batch,
        sr=getattr(audio_streamer, "target_sr", 22050),
        n_channels=int(num_input_neurons),
        win_ms=25,
        hop_ms=10,
        target_max_rate_hz=150.0,  # Balanced for audio-only (use 300 for multimodal)
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
    partition="train",
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
    # Load image batch (labels returned are already per-timestep)
    spike_data, labels = image_streamer.get_batch(
        start_idx, batch_size, partition=partition
    )

    if spike_data is None:
        return None, None

    # Do NOT repeat again; labels are already length = batch_size * num_steps
    return spike_data, labels


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
    train=22000,
    test=7900,
    val=100,
):
    # Set random seed for reproducibility
    np.random.seed(42)
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

    print(f"Total available samples: {available_samples}")
    print(f"Max samples limit: {max_total_samples}")
    print(f"Actual samples used: {actual_max_samples}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Train samples: {train_samples} ({train_ratio*100:.1f}%)")
    print(f"Val samples: {val_samples} ({val_ratio*100:.1f}%)")
    print(f"Test samples: {test_samples} ({test_ratio*100:.1f}%)")
    print(f"Train samples per class: {train_per_class}")
    print(f"Val samples per class: {val_per_class}")
    print(f"Test samples per class: {test_per_class}")

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


def convert_audio_to_spikes(audio_data, num_steps, pixel_size):
    """
    Convert audio waveforms to spike trains using a cochlea-inspired approach.
    Each neuron encodes a specific frequency band and spikes according to a Poisson process
    with mean rate proportional to the energy in that band at each time step.
    """
    import numpy as np
    import torch

    batch_size = len(audio_data)
    spike_data = []

    # Define frequency bands (one per neuron/pixel)
    n_freqs = pixel_size * pixel_size
    sr = 22050  # Assume AudioMNIST default sample rate

    # Center frequencies spaced on a mel scale for biological plausibility
    mel_min, mel_max = 0, 2595 * np.log10(1 + (sr // 2) / 700)
    mel_points = np.linspace(mel_min, mel_max, n_freqs + 2)[1:-1]
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    for i, audio_sample in enumerate(audio_data):
        if i % 1000 == 0:
            print(f"Converting audio sample {i+1}/{batch_size} to spikes...")

        # Pad or trim audio to fixed length
        audio = np.array(audio_sample)
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)))
        else:
            audio = audio[:sr]

        # Compute STFT
        hop_length = sr // num_steps
        stft = np.abs(np.fft.rfft(audio, n=sr))
        freqs = np.fft.rfftfreq(sr, 1 / sr)

        # For each time step, get the spectrum in a window
        spikes_t = []
        for t in range(num_steps):
            start = t * hop_length
            end = start + hop_length
            segment = audio[start:end] if end <= len(audio) else audio[start:]
            if len(segment) < 8:  # skip if too short
                segment = np.pad(segment, (0, 8 - len(segment)))
            spectrum = np.abs(np.fft.rfft(segment, n=hop_length))
            # For each frequency band, compute mean energy
            band_energies = []
            for f in hz_points:
                idx = np.argmin(np.abs(freqs - f))
                band_energies.append(spectrum[idx] if idx < len(spectrum) else 0)
            band_energies = np.array(band_energies)
            # Normalize energies to [0, 1]
            if band_energies.max() > 0:
                band_energies = band_energies / band_energies.max()
            # Poisson mean rate: scale to reasonable spike rate (e.g., max 1 per timestep)
            lam = band_energies
            # Generate Poisson spikes for each neuron (frequency band)
            spikes_flat = np.random.poisson(lam)
            # Reshape to (pixel_size, pixel_size)
            spikes_img = spikes_flat.reshape(pixel_size, pixel_size).astype(np.float32)
            spikes_t.append(spikes_img)
        sample_spike_array = np.stack(spikes_t, axis=0)
        spike_data.append(sample_spike_array)

    spike_tensor = torch.tensor(np.array(spike_data), dtype=torch.float32)
    return spike_tensor


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
    max_total_samples,
    train_ratio,
    test_ratio,
    val_ratio,
    audioMNIST=True,
    imageMNIST=False,
    use_validation_data=True,
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
        # Load data
        data_path = r"C:\Users\Andreas\Documents\GitHub\AudioMNIST\data"
        audio_data, audio_labels = load_audiomnist_data(data_path)

        if audio_data is not None:
            # Create balanced splits maintaining 60/20/20 ratio
            (
                (audio_train_data, audio_train_labels),
                (audio_val_data, audio_val_labels),
                (audio_test_data, audio_test_labels),
            ) = create_balanced_splits(
                audio_data,
                audio_labels,
                max_total_samples=max_total_samples,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            print(
                f"Audio data loaded: {len(audio_train_data)} train, {len(audio_val_data)} val, {len(audio_test_data)} test"
            )

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

        # Use the specified number of temporal steps
        temporal_steps = num_steps

    # Process audio data if available
    if audio_train_data is not None:
        print(
            f"Converting audio data to spike trains with {temporal_steps} temporal steps..."
        )

        # Convert audio to spike trains
        audio_spikes_train = convert_audio_to_spikes(
            audio_train_data, temporal_steps, pixel_size
        )
        audio_spikes_test = convert_audio_to_spikes(
            audio_test_data, temporal_steps, pixel_size
        )
        if use_validation_data and audio_val_data is not None:
            audio_spikes_val = convert_audio_to_spikes(
                audio_val_data, temporal_steps, pixel_size
            )

        print(f"Audio spike conversion complete:")
        print(f"Train audio spikes shape: {audio_spikes_train.shape}")
        print(f"Test audio spikes shape: {audio_spikes_test.shape}")

        # If no image data, use audio data for the main processing
        if images_train is None:
            print("Using audio data as primary data source")
            norm_images_train = audio_spikes_train
            norm_images_test = audio_spikes_test
            if use_validation_data and audio_val_data is not None:
                norm_images_val = audio_spikes_val
            # Use audio labels
            labels_train = torch.tensor(audio_train_labels)
            labels_test = torch.tensor(audio_test_labels)
            if use_validation_data and audio_val_labels is not None:
                labels_val = torch.tensor(audio_val_labels)

    # If no image data but we have audio data, create dummy image data
    if images_train is None and audio_train_data is None:
        print("No image data available, creating dummy data for compatibility")
        norm_images_train = torch.zeros((1, temporal_steps, pixel_size, pixel_size))
        norm_images_test = torch.zeros((1, temporal_steps, pixel_size, pixel_size))
        if use_validation_data:
            norm_images_val = torch.zeros((1, temporal_steps, pixel_size, pixel_size))
        labels_train = torch.zeros(1, dtype=torch.long)
        labels_test = torch.zeros(1, dtype=torch.long)
        if use_validation_data:
            labels_val = torch.zeros(1, dtype=torch.long)

    # Handle different data types for spike generation
    # Check if data is already in spike format (from audio conversion) or needs conversion (images)

    if (
        norm_images_train.shape[1] == num_steps
        and norm_images_train.shape[2] == pixel_size
    ):
        # Data is already in spike format (converted from audio)
        print("Using pre-converted spike data (audio or processed data)")
        S_data_train = norm_images_train
        S_data_test = norm_images_test
        if use_validation_data:
            S_data_val = norm_images_val
    else:
        # Data needs spike conversion (traditional image data)
        print("Converting image data to spike trains using Poisson process")
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

    # Combine audio and image data if both are present
    if (
        audioMNIST
        and imageMNIST
        and audio_train_data is not None
        and images_train is not None
    ):
        print("\nCombining audio and image data into unified dataset...")

        # Get the spike trains for both modalities
        audio_spikes_train = convert_audio_to_spikes(
            audio_train_data, num_steps, pixel_size
        )
        audio_spikes_test = convert_audio_to_spikes(
            audio_test_data, num_steps, pixel_size
        )

        # Combine by concatenating along channel/feature dimension
        # Original shape: (batch, time_steps, pixel_size, pixel_size)
        # Combined shape: (batch, time_steps, pixel_size, pixel_size * 2)

        # For now, we'll use image data as primary and add audio as additional channels
        # You could also implement different fusion strategies here
        S_data_train_combined = torch.cat(
            [
                S_data_train.view(-1, num_steps, pixel_size, pixel_size),
                audio_spikes_train,
            ],
            dim=-1,
        )
        S_data_test_combined = torch.cat(
            [
                S_data_test.view(-1, num_steps, pixel_size, pixel_size),
                audio_spikes_test,
            ],
            dim=-1,
        )

        # Update data references
        S_data_train = S_data_train_combined.view(-1, pixel_size * 2)
        S_data_test = S_data_test_combined.view(-1, pixel_size * 2)

        if use_validation_data:
            audio_spikes_val = convert_audio_to_spikes(
                audio_val_data, num_steps, pixel_size
            )
            S_data_val_combined = torch.cat(
                [
                    S_data_val.view(-1, num_steps, pixel_size, pixel_size),
                    audio_spikes_val,
                ],
                dim=-1,
            )
            S_data_val = S_data_val_combined.view(-1, pixel_size * 2)

        print(f"Combined modalities - New feature dimension: {pixel_size * 2}")

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

    print(f"Data shape: {S_data_train.shape}")
    print(f"Labels shape: {spike_labels_train.shape}")
    print(f"Temporal steps: {num_steps}")
    print(f"Spatial dimensions: {pixel_size}x{pixel_size}")
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


# ============================================================================
# Streaming Functions (moved from big_comb.py)
# ============================================================================


def load_audio_batch_streaming(
    audio_streamer,
    batch_size,
    current_train_idx,
    current_test_idx,
    all_train,
    all_test,
    num_steps,
    N_x,
    get_training_mode_func,
    is_training=True,
    plot_spectrograms=False,
    partition=None,
):
    """
    Load a batch of audio data and convert to spikes on-demand.

    Args:
        audio_streamer: The audio data streamer object
        batch_size: Number of samples to load
        current_train_idx: Current training index
        current_test_idx: Current test index
        all_train: Total training images/samples
        all_test: Total test images/samples
        num_steps: Number of time steps per sample
        N_x: Total number of input neurons
        get_training_mode_func: Function to get training mode
        is_training: If True, load training data; if False, load test data

    Returns:
        (spike_data, labels, new_train_idx, new_test_idx) or (None, None, current_train_idx, current_test_idx) if no more data
    """
    if audio_streamer is None:
        return None, None, current_train_idx, current_test_idx

    # Check if we've exceeded the total limit
    if is_training and current_train_idx >= all_train:
        return None, None, current_train_idx, current_test_idx
    if not is_training and current_test_idx >= all_test:
        return None, None, current_train_idx, current_test_idx

    # Determine current index
    start_idx = current_train_idx if is_training else current_test_idx

    # Load audio batch - determine correct number of neurons based on mode
    training_mode = get_training_mode_func()
    if training_mode == "multimodal":
        # In multimodal, each modality uses (floor((sqrt(base))/sqrt(2)))^2 where base = original single-modality sqrt
        # Here N_x is the total expected concatenated size; each side should be N_x // 2
        per_mod_total = N_x // 2
        base_dim = int(np.sqrt(per_mod_total * 2))  # reverse of 2*(shared_dim^2)
        shared_dim = int(np.floor(base_dim / np.sqrt(2)))
        num_audio_neurons = int(shared_dim**2)
    else:
        # Audio-only mode - use full N_x for audio
        num_audio_neurons = int(np.sqrt(N_x)) ** 2

    # print(f"Audio neurons: {num_audio_neurons}")  # Commented for performance
    spike_data, labels = load_audio_batch(
        audio_streamer,
        start_idx,
        batch_size,
        num_steps,
        num_audio_neurons,
        plot_spectrograms=plot_spectrograms,
        partition=(
            partition if partition is not None else ("train" if is_training else "val")
        ),
    )

    # Update indices
    new_train_idx = current_train_idx
    new_test_idx = current_test_idx
    if spike_data is not None:
        if is_training:
            new_train_idx = current_train_idx + batch_size
        else:
            new_test_idx = current_test_idx + batch_size

    return spike_data, labels, new_train_idx, new_test_idx


def load_multimodal_batch(
    audio_streamer,
    image_streamer,
    batch_size,
    current_train_idx,
    current_test_idx,
    all_train,
    all_test,
    num_steps,
    N_x,
    get_training_mode_func,
    load_prealigned_func,
    is_training=True,
    plot_spectrograms=False,
    partition=None,
):
    """
    Load a batch of multimodal data (image + audio) with synchronized labels.

    Args:
        audio_streamer: The audio data streamer object
        image_streamer: The image data streamer object
        batch_size: Number of samples to load
        current_train_idx: Current training index
        current_test_idx: Current test index
        all_train: Total training images/samples
        all_test: Total test images/samples
        num_steps: Number of time steps per sample
        N_x: Total number of input neurons
        get_training_mode_func: Function to get training mode
        load_prealigned_func: Function to load pre-aligned batch
        is_training: If True, load training data; if False, load test data

    Returns:
        (concatenated_spike_data, labels, new_train_idx, new_test_idx) or (None, None, current_train_idx, current_test_idx) if no more data
    """

    # Determine correct number of neurons based on mode
    training_mode = get_training_mode_func()
    if training_mode == "multimodal":
        # Each modality should contribute exactly N_x // 2 features using the same shared_dim rule
        per_mod_total = N_x // 2
        base_dim = int(np.sqrt(per_mod_total * 2))
        shared_dim = int(np.floor(base_dim / np.sqrt(2)))
        per_mod_features = int(shared_dim**2)
        # If rounding caused drift, clamp to per_mod_total
        if per_mod_features > per_mod_total:
            per_mod_features = per_mod_total
        num_image_neurons = per_mod_features
        num_audio_neurons = per_mod_features
    else:
        # Fallback - use full N_x
        num_image_neurons = int(np.sqrt(N_x)) ** 2
        num_audio_neurons = int(np.sqrt(N_x)) ** 2

    # Load pre-aligned multimodal batch
    multimodal_spikes, labels = load_prealigned_func(
        audio_streamer=audio_streamer,
        image_streamer=image_streamer,
        batch_size=batch_size,
        current_train_idx=current_train_idx,
        current_test_idx=current_test_idx,
        all_train=all_train,
        all_test=all_test,
        num_steps=num_steps,
        N_x=N_x,
        get_training_mode_func=get_training_mode_func,
        is_training=is_training,
        plot_spectrograms=plot_spectrograms,
        partition=partition,
        num_image_neurons=num_image_neurons,
        num_audio_neurons=num_audio_neurons,
    )

    if multimodal_spikes is None:
        return None, None, current_train_idx, current_test_idx

    # Advance the index after successful loading
    new_train_idx = current_train_idx + batch_size if is_training else current_train_idx
    new_test_idx = (
        current_test_idx + batch_size if not is_training else current_test_idx
    )

    return multimodal_spikes, labels, new_train_idx, new_test_idx


def load_prealigned_multimodal_batch(
    audio_streamer,
    image_streamer,
    batch_size,
    current_train_idx,
    current_test_idx,
    all_train,
    all_test,
    num_steps,
    N_x,
    get_training_mode_func,
    is_training=True,
    plot_spectrograms=False,
    partition=None,
    num_image_neurons=None,
    num_audio_neurons=None,
):
    """
    Load a batch from multimodal datasets with pre-aligned labels.
    Uses pre-aligned indices to ensure audio and image samples correspond to the same class.
    """
    # Determine partition
    data_partition = (
        partition if partition is not None else ("train" if is_training else "val")
    )

    # Get pre-aligned indices for this batch
    aligned_indices = get_prealigned_indices(
        audio_streamer=audio_streamer,
        image_streamer=image_streamer,
        batch_size=batch_size,
        current_idx=current_train_idx if is_training else current_test_idx,
        partition=data_partition,
        is_training=is_training,
    )

    if aligned_indices is None:
        return None, None

    audio_indices, image_indices = aligned_indices

    # Load audio batch using pre-aligned indices
    audio_spikes, audio_labels = load_audio_batch_with_indices(
        audio_streamer=audio_streamer,
        indices=audio_indices,
        num_steps=num_steps,
        num_input_neurons=num_audio_neurons,
        plot_spectrograms=plot_spectrograms,
        partition=data_partition,
    )

    if audio_spikes is None:
        return None, None

    # Load image batch using pre-aligned indices
    image_spikes, image_labels = load_image_batch_with_indices(
        image_streamer=image_streamer,
        indices=image_indices,
        num_steps=num_steps,
        num_input_neurons=num_image_neurons,
        partition=data_partition,
    )

    if image_spikes is None:
        return None, None

    # Verify alignment (should be perfect now)
    if len(audio_labels) > 0 and len(image_labels) > 0:
        if not np.array_equal(audio_labels, image_labels):
            print(f"Warning: Label mismatch detected!")
            print(f"Audio labels: {audio_labels}")
            print(f"Image labels: {image_labels}")
        else:
            print(f"✓ Perfect alignment: {len(audio_labels)} samples")

    # Concatenate image and audio features
    # Concatenate and if needed pad/truncate to exactly N_x columns expected by the network
    concatenated = np.concatenate([image_spikes, audio_spikes], axis=1)
    if concatenated.shape[1] < N_x:
        pad_cols = N_x - concatenated.shape[1]
        concatenated = np.pad(concatenated, ((0, 0), (0, pad_cols)), mode="constant")
    elif concatenated.shape[1] > N_x:
        concatenated = concatenated[:, :N_x]

    multimodal_spikes = concatenated

    # Labels are already extended by individual batch loaders
    multimodal_labels = audio_labels

    return multimodal_spikes, multimodal_labels


def get_prealigned_indices(
    audio_streamer,
    image_streamer,
    batch_size,
    current_idx,
    partition="train",
    is_training=True,
):
    """
    Get pre-aligned indices for audio and image data to ensure matching labels.
    Returns (audio_indices, image_indices) or None if alignment fails.
    """
    # Get available indices for both datasets
    if partition == "train":
        audio_pool = audio_streamer.train_indices
        image_pool = image_streamer.train_indices
    elif partition == "val":
        audio_pool = audio_streamer.val_indices
        image_pool = image_streamer.val_indices
    else:  # test
        audio_pool = audio_streamer.test_indices
        image_pool = image_streamer.test_indices

    # Check if we have enough data
    if current_idx >= len(audio_pool) or current_idx >= len(image_pool):
        return None

    # Get the requested batch size, limited by available data
    end_idx = min(current_idx + batch_size, len(audio_pool), len(image_pool))
    actual_batch_size = end_idx - current_idx

    if actual_batch_size <= 0:
        return None

    # Get indices for this batch
    audio_indices = audio_pool[current_idx:end_idx]
    image_indices = image_pool[current_idx:end_idx]

    # Get labels for verification (ensure numpy arrays for fancy indexing)
    audio_labels = np.asarray(audio_streamer.audio_labels)[audio_indices]
    image_labels = np.asarray(image_streamer.all_labels)[image_indices]

    # Verify alignment
    if not np.array_equal(audio_labels, image_labels):
        print(f"Warning: Label mismatch in batch alignment!")
        print(f"Audio labels: {audio_labels}")
        print(f"Image labels: {image_labels}")
        # For now, we'll continue but this indicates a problem with the pre-alignment

    return audio_indices, image_indices


def load_audio_batch_with_indices(
    audio_streamer,
    indices,
    num_steps,
    num_input_neurons,
    plot_spectrograms=False,
    partition="train",
):
    """
    Load audio data using specific indices instead of sequential loading.
    """
    if indices is None or len(indices) == 0:
        return None, None

    audio_batch, labels_batch = [], []
    for idx in indices:
        fp = audio_streamer.audio_files[idx]
        try:
            x, sr = librosa.load(fp, sr=audio_streamer.target_sr, mono=True)
            audio_batch.append(x.astype(np.float32))
            labels_batch.append(audio_streamer.audio_labels[idx])
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    if not audio_batch:
        return None, None

    # Convert to spikes
    spike_data_3d = cochlear_to_spikes_1s(
        audio_batch,
        sr=audio_streamer.target_sr,
        n_channels=int(num_input_neurons),
        win_ms=25,
        hop_ms=10,
        target_max_rate_hz=150.0,  # Balanced for audio-only (use 300 for multimodal)
        env_cutoff_hz=120.0,
        out_T_ms=num_steps,
        eps=1e-8,
    )  # (B, num_steps, num_input_neurons)

    # Flatten batch and time to match training pipeline expectations: (B*T, F)
    spike_data = spike_data_3d.reshape(-1, spike_data_3d.shape[-1])

    # Extend labels to match spike data (repeat each label for num_steps)
    spike_labels = np.array(labels_batch).repeat(num_steps)

    # Optional visualization
    if plot_spectrograms:
        try:
            plot_audio_spectrograms_and_spikes(
                audio_data=audio_batch,
                spikes=spike_data,
                spike_labels=spike_labels,
                audio_labels=np.array(labels_batch),
                num_steps=num_steps,
                sample_rate=audio_streamer.target_sr,
            )
        except Exception as e:
            print(f"Warning: failed to plot spectrograms for this batch: {e}")

    return spike_data, spike_labels


def load_image_batch_with_indices(
    image_streamer,
    indices,
    num_steps,
    num_input_neurons,
    partition="train",
):
    """
    Load image data using specific indices instead of sequential loading.
    """
    if indices is None or len(indices) == 0:
        return None, None

    # Get batch data using the specific indices
    batch_images = image_streamer.all_images[indices]
    batch_labels = image_streamer.all_labels[indices]

    # Convert to spikes
    spike_data = image_streamer._convert_images_to_spikes(batch_images)

    # Extend labels to match spike data (repeat each label for num_steps)
    spike_labels = batch_labels.repeat(num_steps)

    return spike_data, spike_labels


def create_prealigned_multimodal_datasets(
    audio_streamer,
    image_streamer,
    max_total_samples=30000,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    """
    Create pre-aligned multimodal datasets ensuring matching labels between audio and image data.
    This is the key fix for the multimodal architecture problem.

    Args:
        audio_streamer: AudioDataStreamer instance
        image_streamer: ImageDataStreamer instance
        max_total_samples: Maximum total samples to use
        train_ratio, val_ratio, test_ratio: Data split ratios

    Returns:
        (audio_streamer, image_streamer) with pre-aligned indices
    """
    print("Creating pre-aligned multimodal datasets...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Get all available labels from both datasets
    audio_labels = audio_streamer.audio_labels
    image_labels = image_streamer.all_labels

    # Find common classes
    audio_classes = set(audio_labels)
    image_classes = set(image_labels)
    common_classes = audio_classes.intersection(image_classes)

    if not common_classes:
        raise ValueError("No common classes found between audio and image datasets!")

    print(f"Common classes: {sorted(common_classes)}")

    # Create aligned indices for each class
    aligned_audio_indices = []
    aligned_image_indices = []

    for class_label in sorted(common_classes):
        # Get indices for this class in both datasets
        audio_class_indices = np.where(audio_labels == class_label)[0]
        image_class_indices = np.where(image_labels == class_label)[0]

        # Take the minimum count to ensure perfect alignment
        min_count = min(len(audio_class_indices), len(image_class_indices))

        if min_count == 0:
            continue

        # Randomly sample the same number from each dataset
        np.random.shuffle(audio_class_indices)
        np.random.shuffle(image_class_indices)

        aligned_audio_indices.extend(audio_class_indices[:min_count])
        aligned_image_indices.extend(image_class_indices[:min_count])

        print(f"Class {class_label}: {min_count} aligned samples")

    # Convert to numpy arrays
    aligned_audio_indices = np.array(aligned_audio_indices)
    aligned_image_indices = np.array(aligned_image_indices)

    # Shuffle the aligned pairs together to maintain correspondence
    shuffle_indices = np.random.permutation(len(aligned_audio_indices))
    aligned_audio_indices = aligned_audio_indices[shuffle_indices]
    aligned_image_indices = aligned_image_indices[shuffle_indices]

    # Limit total samples if needed
    if len(aligned_audio_indices) > max_total_samples:
        aligned_audio_indices = aligned_audio_indices[:max_total_samples]
        aligned_image_indices = aligned_image_indices[:max_total_samples]

    total_samples = len(aligned_audio_indices)
    print(f"Total aligned samples: {total_samples}")

    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # Split indices
    train_end = train_size
    val_end = train_size + val_size

    # Update streamer indices with aligned data
    audio_streamer.train_indices = aligned_audio_indices[:train_end]
    audio_streamer.val_indices = aligned_audio_indices[train_end:val_end]
    audio_streamer.test_indices = aligned_audio_indices[val_end:]

    image_streamer.train_indices = aligned_image_indices[:train_end]
    image_streamer.val_indices = aligned_image_indices[train_end:val_end]
    image_streamer.test_indices = aligned_image_indices[val_end:]

    # Reset pointers
    audio_streamer.ptr_train = 0
    audio_streamer.ptr_val = 0
    audio_streamer.ptr_test = 0

    image_streamer.ptr_train = 0
    image_streamer.ptr_val = 0
    image_streamer.ptr_test = 0

    print(f"Pre-aligned splits:")
    print(f"  Train: {len(audio_streamer.train_indices)} samples")
    print(f"  Val: {len(audio_streamer.val_indices)} samples")
    print(f"  Test: {len(audio_streamer.test_indices)} samples")

    # Verify alignment - ensure numpy arrays for advanced indexing
    audio_labels_array = np.asarray(audio_streamer.audio_labels)
    image_labels_array = np.asarray(image_streamer.all_labels)
    train_audio_labels = audio_labels_array[audio_streamer.train_indices[:10]]
    train_image_labels = image_labels_array[image_streamer.train_indices[:10]]

    if np.array_equal(train_audio_labels, train_image_labels):
        print("✓ Alignment verification passed!")
    else:
        print("✗ Alignment verification failed!")
        print(f"Audio labels: {train_audio_labels}")
        print(f"Image labels: {train_image_labels}")

    return audio_streamer, image_streamer


def sync_multimodal_datasets():
    """
    Synchronize multimodal datasets by ensuring they use the same random seed.
    This ensures that when both datasets are shuffled, they have similar label distributions.
    """
    print("Synchronizing multimodal datasets with shared random seed...")

    # Set a fixed seed for reproducibility across both datasets
    sync_seed = 42

    # The streamers should use this seed internally when shuffling
    # For now, we just ensure the seed is set globally
    np.random.seed(sync_seed)

    print(
        "Multimodal datasets synchronized! Both will use the same random seed for shuffling."
    )
    print(f"Sync seed: {sync_seed}")

    # Reset the random seed to avoid affecting other random operations
    np.random.seed()
