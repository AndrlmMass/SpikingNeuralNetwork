from torchvision import datasets, transforms
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.get_data import (
    cochlear_to_spikes_1s,
    create_balanced_splits,
)
from src.get_data import AudioDataStreamer
from tqdm import tqdm
import torch
import numpy as np


def collect_imageMNIST(pixel_size=11, data_dir="data/MNIST", download=True):
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

    # Set random seed
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
        train=22000,
        test=7900,
        val=100,
    )

    # Convert back to tensors
    images_train = torch.tensor(image_train_data)
    labels_train = torch.tensor(image_train_labels)
    images_val = torch.tensor(image_val_data)
    labels_val = torch.tensor(image_val_labels)
    images_test = torch.tensor(image_test_data)
    labels_test = torch.tensor(image_test_labels)

    return images_train, labels_train, images_val, labels_val, images_test, labels_test


def load_audio_batch(
    start_idx,
    batch_size,
    num_steps,
    num_input_neurons,
    debug=False,
    force_recompute=False,
    data_path="/home/andreas/Documents/GitHub/AudioMNIST/data",
    stream_batch_size: int = 256,
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
    # Define audio streamer
    audio_streamer = AudioDataStreamer(
        data_path=data_path,
        target_sr=22050,
        batch_size=stream_batch_size,
    )

    # Build cache path for features/splits
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    target_sr = getattr(audio_streamer, "target_sr", 22050)
    cache_file = os.path.join(
        cache_dir,
        f"audio_features_sr{target_sr}_ch{int(num_input_neurons)}_T{int(num_steps)}.npz",
    )

    # A single-file precomputed cache of the full dataset (before splitting)
    precomputed_all = os.path.join(
        cache_dir,
        f"audio_all_sr{target_sr}_ch{int(num_input_neurons)}_T{int(num_steps)}.npz",
    )

    if os.path.exists(cache_file) and not force_recompute:
        data = np.load(cache_file)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
        )

    # If we have a precomputed full dataset and not forcing recompute, use it
    if os.path.exists(precomputed_all) and not force_recompute:
        data_all = np.load(precomputed_all)
        X_full = data_all["X_all"]
        y_full = data_all["y_all"]
    else:
        # Stream through the dataset in mini-batches and write to memmap to minimize RAM
        total = audio_streamer.get_total_samples()
        if total == 0:
            raise ValueError("No audio files found for precomputation")

        # Prepare on-disk memmaps
        mm_X_path = os.path.join(
            cache_dir,
            f"audio_all_sr{target_sr}_ch{int(num_input_neurons)}_T{int(num_steps)}.mmap",
        )
        mm_y_path = os.path.join(
            cache_dir,
            f"audio_all_labels_sr{target_sr}_ch{int(num_input_neurons)}_T{int(num_steps)}.mmap",
        )
        X_mm = np.memmap(
            mm_X_path,
            dtype=np.float32,
            mode="w+",
            shape=(total, int(num_input_neurons)),
        )
        y_mm = np.memmap(mm_y_path, dtype=np.int64, mode="w+", shape=(total,))

        pbar = tqdm(total=total, desc="Precomputing audio features", disable=False)
        cur = 0
        write_ptr = 0
        while True:
            audio_batch, labels = audio_streamer.get_batch(cur, stream_batch_size)
            if audio_batch is None:
                break

            _, rates_bt = cochlear_to_spikes_1s(
                audio_batch,
                sr=target_sr,
                n_channels=num_input_neurons,
                win_ms=25,
                hop_ms=10,
                target_max_rate_hz=100.0,
                env_cutoff_hz=120.0,
                out_T_ms=num_steps,
                eps=1e-8,
                return_rates=True,
                show_progress=False,
                debug=debug,
            )
            mean_rates = rates_bt.mean(axis=1).astype(np.float32)

            n = len(labels)
            X_mm[write_ptr : write_ptr + n] = mean_rates
            y_mm[write_ptr : write_ptr + n] = labels
            write_ptr += n

            pbar.update(n)
            cur += stream_batch_size
            if cur >= total:
                break

        pbar.close()

        # Flush memmap to disk and wrap into a single compressed npz for easy reuse
        del X_mm
        del y_mm

        X_full = np.memmap(
            mm_X_path, dtype=np.float32, mode="r", shape=(total, int(num_input_neurons))
        )
        y_full = np.memmap(mm_y_path, dtype=np.int64, mode="r", shape=(total,))
        np.savez_compressed(
            precomputed_all, X_all=np.array(X_full), y_all=np.array(y_full)
        )
        # Optionally remove memmap temporary files now that compressed file exists
        try:
            os.remove(mm_X_path)
            os.remove(mm_y_path)
        except Exception:
            pass

    # Split data into train, val, test
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_balanced_splits(
        X_full,
        y_full,
        max_total_samples=30000,
        train=22000,
        test=7900,
        val=100,
    )

    # Save cache for next runs
    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        meta=np.array(
            [target_sr, int(num_input_neurons), int(num_steps)], dtype=np.int64
        ),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data(
    pixel_size: int = 15,
    num_steps: int = 1000,
    batch_size: int = 30000,
    imageMNIST: bool = True,
    audioMNIST: bool = True,
    debug: bool = False,
    force_recompute: bool = False,
    combined: bool = True,
):
    """
    Prepare mean-rate features for AudioMNIST for a linear classifier.

    - Streams audio using AudioDataStreamer
    - Encodes with cochlear_to_spikes_1s (rates path) to 1000 steps
    - Averages rates over time per channel to get one feature vector per sample
    - Returns balanced train/val/test splits

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """

    images_train = images_val = images_test = None
    image_labels_train = image_labels_val = image_labels_test = None
    audio_train = audio_val = audio_test = None
    audio_labels_train = audio_labels_val = audio_labels_test = None
    combined_train = combined_val = combined_test = None
    combined_labels_train = combined_labels_val = combined_labels_test = None

    if imageMNIST:
        # get image data
        (
            images_train,
            image_labels_train,
            images_val,
            image_labels_val,
            images_test,
            image_labels_test,
        ) = collect_imageMNIST()
    if audioMNIST:
        # get audio data
        (
            audio_train,
            audio_labels_train,
            audio_val,
            audio_labels_val,
            audio_test,
            audio_labels_test,
        ) = load_audio_batch(
            start_idx=0,
            batch_size=batch_size,
            num_steps=num_steps,
            num_input_neurons=int(pixel_size**2),
            debug=debug,
            force_recompute=force_recompute,
        )

    # Convert images to flat feature arrays for classifier if available
    def _flatten_images(x):
        if x is None:
            return None
        arr = x.numpy()
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)

    image_train_feats = _flatten_images(images_train) if imageMNIST else None
    image_val_feats = _flatten_images(images_val) if imageMNIST else None
    image_test_feats = _flatten_images(images_test) if imageMNIST else None

    # Optionally build balanced combined audio+image features
    if (
        combined
        and imageMNIST
        and audioMNIST
        and image_train_feats is not None
        and audio_train is not None
    ):
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        img_f = int(pixel_size**2)
        aud_f = int(pixel_size**2)
        comb_cache = os.path.join(
            cache_dir,
            f"combined_features_img{img_f}_aud{aud_f}_T{int(num_steps)}.npz",
        )

        if os.path.exists(comb_cache) and not force_recompute:
            data = np.load(comb_cache)
            combined_train = data["X_train_comb"]
            combined_labels_train = data["y_train_comb"]
            combined_val = data["X_val_comb"]
            combined_labels_val = data["y_val_comb"]
            combined_test = data["X_test_comb"]
            combined_labels_test = data["y_test_comb"]
        else:

            def _align_and_concat(X_img, y_img, X_aud, y_aud):
                # reduce size of X_img and X_aud to sum to pixel_size**2 with random sampling
                size = int(np.floor(pixel_size // (np.sqrt(2))) ** 2)
                X_img = X_img[:, :size]
                X_aud = X_aud[:, :size]

                classes_img = np.unique(y_img)
                classes_aud = np.unique(y_aud)
                classes = sorted(set(classes_img.tolist()) & set(classes_aud.tolist()))
                if not classes:
                    raise ValueError(
                        "No overlapping classes between image and audio labels"
                    )
                # enforce balanced per-class count
                per_class_min = [
                    min(np.sum(y_img == c), np.sum(y_aud == c)) for c in classes
                ]
                k_common = int(min(per_class_min))
                if k_common <= 0:
                    raise ValueError(
                        "Insufficient overlapping samples to build combined set"
                    )
                X_list, y_list = [], []
                for c in classes:
                    idx_img = np.where(y_img == c)[0][:k_common]
                    idx_aud = np.where(y_aud == c)[0][:k_common]
                    Xi = X_img[idx_img]
                    Xa = X_aud[idx_aud]
                    # Concatenate features
                    Xc = np.concatenate([Xi, Xa], axis=1)
                    yc = np.full(k_common, c, dtype=y_img.dtype)
                    # sanity checks
                    assert Xi.shape[0] == Xa.shape[0] == k_common
                    X_list.append(Xc)
                    y_list.append(yc)
                X = np.vstack(X_list)
                y = np.concatenate(y_list)
                # shuffle
                perm = np.random.permutation(len(y))
                return X[perm], y[perm]

            combined_train, combined_labels_train = _align_and_concat(
                image_train_feats,
                image_labels_train.numpy(),
                audio_train,
                audio_labels_train,
            )
            combined_val, combined_labels_val = _align_and_concat(
                image_val_feats, image_labels_val.numpy(), audio_val, audio_labels_val
            )
            combined_test, combined_labels_test = _align_and_concat(
                image_test_feats,
                image_labels_test.numpy(),
                audio_test,
                audio_labels_test,
            )

            np.savez_compressed(
                comb_cache,
                X_train_comb=combined_train,
                y_train_comb=combined_labels_train,
                X_val_comb=combined_val,
                y_val_comb=combined_labels_val,
                X_test_comb=combined_test,
                y_test_comb=combined_labels_test,
            )

    return (
        image_train_feats,
        image_labels_train,
        image_val_feats,
        image_labels_val,
        image_test_feats,
        image_labels_test,
        audio_train,
        audio_labels_train,
        audio_val,
        audio_labels_val,
        audio_test,
        audio_labels_test,
        combined_train,
        combined_labels_train,
        combined_val,
        combined_labels_val,
        combined_test,
        combined_labels_test,
    )


if __name__ == "__main__":
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        audio_train,
        labels_train,
        audio_val,
        labels_val,
        audio_test,
        labels_test,
        Xc_train,
        yc_train,
        Xc_val,
        yc_val,
        Xc_test,
        yc_test,
    ) = get_data(imageMNIST=True, audioMNIST=True, combined=True, debug=True)

    def safe_shape(x):
        return getattr(x, "shape", None)

    print("Results (None if not computed):")
    print("X_train:", safe_shape(X_train), "y_train:", safe_shape(y_train))
    print("X_val:", safe_shape(X_val), "y_val:", safe_shape(y_val))
    print("X_test:", safe_shape(X_test), "y_test:", safe_shape(y_test))
    print(
        "audio_train:",
        safe_shape(audio_train),
        "labels_train:",
        safe_shape(labels_train),
    )
    print("audio_val:", safe_shape(audio_val), "labels_val:", safe_shape(labels_val))
    print(
        "audio_test:", safe_shape(audio_test), "labels_test:", safe_shape(labels_test)
    )
    print(
        "combined_train:",
        safe_shape(Xc_train),
        "combined_labels:",
        safe_shape(yc_train),
    )
    print("combined_val:", safe_shape(Xc_val), "combined_labels:", safe_shape(yc_val))
    print(
        "combined_test:", safe_shape(Xc_test), "combined_labels:", safe_shape(yc_test)
    )
