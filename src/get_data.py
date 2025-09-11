from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
import torch.nn.functional as F
from snntorch import spikegen
import numpy as np
import torch
import os
import json

os.environ["TQDM_DISABLE"] = "True"


def normalize_image(img, target_sum=1.0):
    current_sum = img.sum()
    return img * (target_sum / current_sum) if current_sum > 0 else img


import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

def normalize_audio_lufs(audio, sample_rate, target_lufs=-23.0):
    """
    Normalize audio to target LUFS (Loudness Units relative to Full Scale)
    This is the standard used by Spotify and other streaming platforms
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Convert RMS to LUFS approximation (simplified)
    # In practice, you'd use a proper LUFS meter, but this gives good results
    current_lufs = 20 * np.log10(rms) if rms > 0 else -60
    
    # Calculate gain needed
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized_audio = audio * gain_linear
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized_audio))
    if max_val > 1.0:
        normalized_audio = normalized_audio / max_val * 0.95
    
    return normalized_audio

def load_audiomnist_data(data_path, target_sr=22050, duration=1.0):
    """
    Load AudioMNIST data with proper normalization and labeling
    """
    all_data = []
    all_labels = []
    
    print("Loading AudioMNIST data...")
    
    if not os.path.exists(data_path):
        print(f"Error: Path {data_path} does not exist!")
        return None, None
    
    # Walk through all files
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                try:
                    # Load audio file
                    audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
                    
                    # Normalize using LUFS standard
                    normalized_audio = normalize_audio_lufs(audio, sr)
                    
                    # Extract label from filename (first character should be 0-9)
                    label = int(file[0]) if file[0].isdigit() else None
                    
                    if label is not None and 0 <= label <= 9:
                        all_data.append(normalized_audio)
                        all_labels.append(label)
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
    
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
            padded_audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
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

def create_balanced_splits(data, labels, max_total_samples=30000, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Create balanced train/validation/test splits maintaining fixed ratios
    Ensures equal representation of each class (0-9) in each split
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
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
    test_samples = actual_max_samples - train_samples - val_samples  # Ensure exact total
    
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
            selected_indices = class_indices[:min(samples_per_class, len(class_indices))]
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

def convert_audio_to_spikes(audio_data, num_steps, pixel_size, use_cochlea_model=True, sample_rate=None):
    """
    Convert audio waveforms to spike trains
    If use_cochlea_model=True: Uses cochlea-inspired frequency decomposition with Poisson processes
    If use_cochlea_model=False: Uses original Bernoulli process approach
    Ensures same temporal duration as image data

    Args:
        audio_data: List of audio waveforms
        num_steps: Number of temporal steps for spike generation
        pixel_size: Spatial dimension size (pixel_size x pixel_size)
        use_cochlea_model: Whether to use cochlea-inspired approach
        sample_rate: Audio sample rate (if None, auto-detects from data length)
    """
    batch_size = len(audio_data)
    spike_data = []

    for i, audio_sample in enumerate(audio_data):
        if i % 1000 == 0:
            print(f"Converting audio sample {i+1}/{batch_size} to spikes...")

        if use_cochlea_model:
            # Cochlea-inspired approach: Frequency decomposition with Poisson processes
            sample_spikes = convert_audio_cochlea_style(audio_sample, num_steps, pixel_size, sample_rate)
        else:
            # Original approach: Amplitude-based Bernoulli process
            sample_spikes = convert_audio_amplitude_style(audio_sample, num_steps, pixel_size)

        spike_data.append(sample_spikes)

    # Convert to tensor: (batch_size, num_steps, pixel_size, pixel_size)
    spike_tensor = torch.tensor(np.array(spike_data), dtype=torch.float32)

    return spike_tensor


def convert_audio_cochlea_style(audio_sample, num_steps, pixel_size, sample_rate=None):
    """
    Convert audio to spikes using cochlea-inspired frequency decomposition
    Each spatial position encodes a different frequency band
    Uses Poisson processes with frequency-specific rates

    Args:
        audio_sample: Audio waveform array
        num_steps: Number of temporal steps for spike generation
        pixel_size: Spatial dimension size (pixel_size x pixel_size)
        sample_rate: Audio sample rate (if None, tries to estimate from data)
    """
    # If sample_rate not provided, try to estimate from audio length
    # AudioMNIST uses 22050 Hz, but we'll make it flexible
    if sample_rate is None:
        # Estimate based on typical audio lengths and sample rates
        # For 1 second audio at 22050 Hz: ~22050 samples
        # For 1 second audio at 16000 Hz: ~16000 samples
        audio_length = len(audio_sample)
        if audio_length > 20000:  # Likely 22050 Hz
            sample_rate = 22050
        elif audio_length > 15000:  # Likely 16000 Hz
            sample_rate = 16000
        else:  # Shorter audio, assume higher sample rate
            sample_rate = 44100

    print(f"Using sample rate: {sample_rate} Hz for frequency analysis")

    # Create mel spectrogram for frequency decomposition (like cochlea)
    # Use more mel bins to ensure good frequency coverage
    n_mels = min(pixel_size * pixel_size, 80)  # More frequency bands for better coverage

    # Ensure n_fft is appropriate for the sample rate
    n_fft = min(2048, len(audio_sample) // 4)  # Larger FFT for better frequency resolution

    # Compute mel spectrogram with full frequency range
    mel_spec = librosa.feature.melspectrogram(
        y=audio_sample,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=max(1, len(audio_sample) // num_steps),
        fmin=20,  # Start from 20 Hz (human hearing threshold)
        fmax=min(sample_rate//2, 20000)  # Up to 20kHz or Nyquist, whichever is smaller
    )

    # Convert to decibels (log scale, like auditory system)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    # For each frequency band (mel bin), compute average activity over time
    # This becomes the Poisson rate for that frequency
    freq_rates = np.mean(mel_spec_norm, axis=1)  # Average across time for each frequency

    # Scale rates to reasonable Poisson firing rates (Hz)
    # Typical neuron firing rates: 0-100 Hz
    max_rate = 50.0  # Maximum firing rate in Hz
    poisson_rates = freq_rates * max_rate

    # Generate Poisson spikes for each frequency band over time
    sample_spikes = []
    dt = len(audio_sample) / sample_rate / num_steps  # Time per step in seconds

    for t in range(num_steps):
        # Create spatial spike pattern where each position corresponds to a frequency
        spatial_spikes = np.zeros((pixel_size, pixel_size), dtype=np.float32)

        # Map frequency bands to spatial positions
        for freq_idx in range(min(n_mels, pixel_size * pixel_size)):
            rate = poisson_rates[freq_idx]

            # Generate Poisson spike count for this time step
            spike_count = np.random.poisson(rate * dt)

            # Distribute spikes across the spatial dimension for this frequency
            if spike_count > 0:
                # Create a spatial pattern for this frequency band
                # Use a more sophisticated mapping to cover all spatial positions
                y_pos = freq_idx // pixel_size
                x_pos = freq_idx % pixel_size

                if y_pos < pixel_size and x_pos < pixel_size:
                    # Place spike at this spatial location
                    spatial_spikes[y_pos, x_pos] = 1.0

                    # If we have multiple spikes, spread them to neighboring positions
                    if spike_count > 1:
                        for extra_spike in range(spike_count - 1):
                            # Add some spatial jitter
                            jitter_y = np.random.randint(-1, 2)
                            jitter_x = np.random.randint(-1, 2)
                            new_y = np.clip(y_pos + jitter_y, 0, pixel_size - 1)
                            new_x = np.clip(x_pos + jitter_x, 0, pixel_size - 1)
                            spatial_spikes[new_y, new_x] = 1.0

        sample_spikes.append(spatial_spikes)

    return np.stack(sample_spikes, axis=0)


def convert_audio_amplitude_style(audio_sample, num_steps, pixel_size):
    """
    Original amplitude-based approach for comparison
    """
    # Get audio properties
    audio_length = len(audio_sample)

    # Resample audio to match temporal steps
    if audio_length > num_steps:
        # Downsample by taking every nth sample
        step_size = audio_length // num_steps
        resampled_audio = audio_sample[::step_size][:num_steps]
    else:
        # Upsample by repeating samples
        repeat_factor = num_steps // audio_length + 1
        resampled_audio = np.tile(audio_sample, repeat_factor)[:num_steps]

    # Convert audio amplitude to spike probability [0, 1]
    # Normalize to [0, 1] range
    audio_min = np.min(resampled_audio)
    audio_max = np.max(resampled_audio)
    if audio_max > audio_min:
        normalized_audio = (resampled_audio - audio_min) / (audio_max - audio_min)
    else:
        normalized_audio = np.zeros_like(resampled_audio)

    # Apply sigmoid to create smoother probability distribution
    spike_probabilities = 1 / (1 + np.exp(-5 * (normalized_audio - 0.5)))

    # Generate spike trains for each temporal step
    sample_spikes = []
    for t in range(num_steps):
        # Create spatial spike pattern based on audio amplitude
        prob = spike_probabilities[t]

        # Generate Bernoulli spikes across spatial dimensions
        # Higher amplitude = higher probability of spikes
        spatial_spikes = np.random.binomial(1, prob, size=(pixel_size, pixel_size)).astype(np.float32)

        sample_spikes.append(spatial_spikes)

    return np.stack(sample_spikes, axis=0)

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
            (audio_train_data, audio_train_labels), (audio_val_data, audio_val_labels), (audio_test_data, audio_test_labels) = create_balanced_splits(
                audio_data, audio_labels, max_total_samples=max_total_samples, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
            )
            print(f"Audio data loaded: {len(audio_train_data)} train, {len(audio_val_data)} val, {len(audio_test_data)} test")

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
        (image_train_data, image_train_labels), (image_val_data, image_val_labels), (image_test_data, image_test_labels) = create_balanced_splits(
            all_images.numpy(), all_labels_array, max_total_samples=30000, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
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
    if audioMNIST and imageMNIST and audio_train_data is not None and images_train is not None:
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
        print(f"Converting audio data to spike trains with {temporal_steps} temporal steps...")

        # Convert audio to spike trains
        audio_spikes_train = convert_audio_to_spikes(audio_train_data, temporal_steps, pixel_size, sample_rate=22050)
        audio_spikes_test = convert_audio_to_spikes(audio_test_data, temporal_steps, pixel_size, sample_rate=22050)
        if use_validation_data and audio_val_data is not None:
            audio_spikes_val = convert_audio_to_spikes(audio_val_data, temporal_steps, pixel_size, sample_rate=22050)

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

    if norm_images_train.shape[1] == num_steps and norm_images_train.shape[2] == pixel_size:
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
    if audioMNIST and imageMNIST and audio_train_data is not None and images_train is not None:
        print("\nCombining audio and image data into unified dataset...")

        # Get the spike trains for both modalities
        audio_spikes_train = convert_audio_to_spikes(audio_train_data, num_steps, pixel_size, sample_rate=22050)
        audio_spikes_test = convert_audio_to_spikes(audio_test_data, num_steps, pixel_size, sample_rate=22050)

        # Combine by concatenating along channel/feature dimension
        # Original shape: (batch, time_steps, pixel_size, pixel_size)
        # Combined shape: (batch, time_steps, pixel_size, pixel_size * 2)

        # For now, we'll use image data as primary and add audio as additional channels
        # You could also implement different fusion strategies here
        S_data_train_combined = torch.cat([S_data_train.view(-1, num_steps, pixel_size, pixel_size),
                                          audio_spikes_train], dim=-1)
        S_data_test_combined = torch.cat([S_data_test.view(-1, num_steps, pixel_size, pixel_size),
                                         audio_spikes_test], dim=-1)

        # Update data references
        S_data_train = S_data_train_combined.view(-1, pixel_size * 2)
        S_data_test = S_data_test_combined.view(-1, pixel_size * 2)

        if use_validation_data:
            audio_spikes_val = convert_audio_to_spikes(audio_val_data, num_steps, pixel_size, sample_rate=22050)
            S_data_val_combined = torch.cat([S_data_val.view(-1, num_steps, pixel_size, pixel_size),
                                           audio_spikes_val], dim=-1)
            S_data_val = S_data_val_combined.view(-1, pixel_size * 2)

        print(f"Combined modalities - New feature dimension: {pixel_size * 2}")

    # Print final data statistics
    print(f"\n=== Final Data Summary ===")
    print(f"Training data: {len(S_data_train)} samples")
    print(f"Test data: {len(S_data_test)} samples")
    if use_validation_data:
        print(f"Validation data: {len(S_data_val)} samples")

    if audioMNIST and audio_train_data is not None:
        print(f"Audio data included: {len(audio_train_data)} train, {len(audio_val_data) if audio_val_data is not None else 0} val, {len(audio_test_data)} test")

    if imageMNIST and images_train is not None:
        print(f"Image data included: {len(images_train)} train, {len(images_val) if images_val is not None else 0} val, {len(images_test)} test")

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


class AudioDataStreamer:
    """
    Streamer class for audio data that provides batch processing capabilities
    """
    def __init__(self, data_path, batch_size=32):
        """
        Initialize the audio data streamer

        Args:
            data_path: Path to audio data directory
            batch_size: Batch size for processing
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.total_samples = 0

        # Count total samples in the dataset
        if os.path.exists(data_path):
            audio_files = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.wav'):
                        audio_files.append(os.path.join(root, file))

            self.total_samples = len(audio_files)
            self.audio_files = audio_files
        else:
            print(f"Warning: Audio data path {data_path} does not exist")
            self.audio_files = []

    def get_total_samples(self):
        """
        Get the total number of audio samples in the dataset

        Returns:
            int: Total number of samples
        """
        return self.total_samples


class ImageDataStreamer:
    """
    Streamer class for image data that provides batch processing capabilities
    """
    def __init__(self, data_path, batch_size=32, pixel_size=28, num_steps=10,
                 gain=1.0, offset=0.0, first_spike_time=0, time_var_input=0):
        """
        Initialize the image data streamer

        Args:
            data_path: Path to image data directory
            batch_size: Batch size for processing
            pixel_size: Size of image pixels
            num_steps: Number of temporal steps
            gain: Gain parameter for spike generation
            offset: Offset parameter for spike generation
            first_spike_time: First spike time
            time_var_input: Time variance for input
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.num_steps = num_steps
        self.gain = gain
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input

        # For MNIST, we know there are 60000 training and 10000 test samples
        # But we'll set a reasonable default
        self.total_samples = 60000

    def get_total_samples(self):
        """
        Get the total number of image samples in the dataset

        Returns:
            int: Total number of samples
        """
        return self.total_samples
