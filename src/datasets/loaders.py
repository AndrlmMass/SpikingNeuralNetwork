from torchvision import datasets, transforms
from snntorch import spikegen
from tqdm import tqdm
from annotated_types import Gt
import tempfile
import shutil
import warnings
import numpy as np
import torch
import os
import json
from PIL import Image
from .geomfig import gen_triangle, gen_circle, gen_square, gen_x, load_or_create_geomfig_data
import hashlib

os.environ["TQDM_DISABLE"] = "True"
warnings.filterwarnings("ignore")

class ImageDataStreamer:
    """
    Eagerly loads and preprocesses the configured dataset into RAM so the
    training loop can request batches without incurring per-iteration
    torchvision/disk overhead.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        pixel_size: int,
        num_steps: int,
        num_classes: int,
        which_classes: list,
        gain: float,
        offset: int,
        first_spike_time: int,
        time_var_input: bool,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        rng: np.random.Generator,
        noise_var: float,
        jitter: bool,
        jitter_amount: float,
        force_recreate: bool,
        num_workers: int,
        seed: int,
        dataset: str,
        tri_size: float,
        tri_thick: int,
        cir_size: float,
        cir_thick: int,
        sqr_size: float,
        sqr_thick: int,
        x_size: float,
        x_thick: int,
        clamp_min: float,
        clamp_max: float,
    ):
        # make arguments global
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.which_classes = which_classes
        self.rng = rng
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input
        self.dataset = (dataset or "mnist").lower()

        # Load image dataset
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((pixel_size, pixel_size)),
                transforms.ToTensor(),
            ]
        )

        ds_map = {
            "mnist": datasets.MNIST,
            "kmnist": datasets.KMNIST,
            "fmnist": datasets.FashionMNIST,
        }
        if self.dataset == "notmnist":
            (
                self.train_images,
                self.train_labels,
                self.val_images,
                self.val_labels,
                self.test_images,
                self.test_labels,
            ) = _load_notmnist_deeplake(transform, self.rng, train_ratio, val_ratio, test_ratio)
            self.len_train = len(self.train_images)
            self.len_val = len(self.val_images)
            self.len_test = len(self.test_images)

        elif self.dataset == "geomfig":
            (
                self.train_images,
                self.train_labels,
                self.val_images,
                self.val_labels,
                self.test_images,
                self.test_labels,
            ) = load_or_create_geomfig_data(
                cache_root=self.data_dir,
                batch_size=self.batch_size,
                pixel_size=self.pixel_size,
                num_steps=self.num_steps,
                gain=self.gain,
                offset=self.offset,
                first_spike_time=self.first_spike_time,
                time_var_input=self.time_var_input,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                rng=self.rng,
                num_classes=self.num_classes,
                which_classes=self.which_classes,
                noise_var=noise_var,
                tri_size=self.tri_size,
                tri_thick=self.tri_thick,
                cir_size=self.cir_size,
                cir_thick=self.cir_thick,
                sqr_size=self.sqr_size,
                sqr_thick=self.sqr_thick,
                x_size=self.x_size,
                x_thick=self.x_thick,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
                jitter=jitter,
                jitter_amount=jitter_amount,
                force_recreate=force_recreate,
                num_workers=num_workers,
                seed=seed,
            )
            # Get lengths of each partition (will be combined in unified partitioning)
            self.len_train = len(self.train_images)
            self.len_val = len(self.val_images)
            self.len_test = len(self.test_images)

        else:
            if self.dataset not in ds_map:
                raise ValueError(f"Unsupported image dataset: {self.dataset}")
            ds_cls = ds_map[self.dataset]

            # Use a stable on-disk cache for torchvision datasets to avoid re-downloads
            torch_root = os.path.join("data", "torchvision")
            os.makedirs(torch_root, exist_ok=True)

            mnist_train = ds_cls(
                root=torch_root, train=True, transform=transform, download=True
            )
            mnist_test = ds_cls(
                root=torch_root, train=False, transform=transform, download=True
            )

            # subset validation data from train set by percentage
            mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_ratio, val_ratio], generator=rng)

            # Try to load from image cache first
            train_images, train_labels = self._load_images_cache("train")
            val_images, val_labels = self._load_images_cache("val")
            test_images, test_labels = self._load_images_cache("test")
            
            if train_images is not None and val_images is not None and test_images is not None:
                print(f"Using cached {self.dataset.upper()} images")
                self.train_images, self.train_labels = train_images, train_labels
                self.val_images, self.val_labels = val_images, val_labels
                self.test_images, self.test_labels = test_images, test_labels
            else:
                # Load all splits into RAM
                self.len_train = len(mnist_train)
                self.len_val = len(mnist_val)
                self.len_test = len(mnist_test)

                def _load_split(dataset):
                    """Helper to load a dataset split into RAM."""
                    images, labels = zip(*[dataset[i] for i in range(len(dataset))])
                    return torch.stack(images), np.array([int(lbl) for lbl in labels], dtype=np.int64)

                self.train_images, self.train_labels = _load_split(mnist_train)
                self.val_images, self.val_labels = _load_split(mnist_val)
                self.test_images, self.test_labels = _load_split(mnist_test)
                
                # Save images to cache
                self._save_images_cache("train", self.train_images, self.train_labels)
                self._save_images_cache("val", self.val_images, self.val_labels)
                self._save_images_cache("test", self.test_images, self.test_labels)
            
            # Set lengths
            self.len_train = len(self.train_images)
            self.len_val = len(self.val_images)
            self.len_test = len(self.test_images)

        # Convert images to spikes (with disk caching)
        self.train_spikes, self.train_labels = self._load_or_convert_spikes(
            "train", self.train_images, self.train_labels
        )
        self.val_spikes, self.val_labels = self._load_or_convert_spikes(
            "val", self.val_images, self.val_labels
        )
        self.test_spikes, self.test_labels = self._load_or_convert_spikes(
            "test", self.test_images, self.test_labels
        )

        # Create and shuffle indices for each partition
        self.train_indices = np.arange(self.len_train)
        self.val_indices = np.arange(self.len_val)
        self.test_indices = np.arange(self.len_test)
        self.rng.shuffle(self.train_indices)
        self.rng.shuffle(self.val_indices)
        self.rng.shuffle(self.test_indices)

        self.ptr_train = 0
        self.ptr_val = 0
        self.ptr_test = 0 

    def get_batch(self, num_samples, partition="train"):
        """
        Load a batch of spike samples from the specified partition.
        Returns (spike_data, labels) or (None, None) if no more data.
        """
        if partition == "train":
            pool = self.train_indices
            ptr = self.ptr_train
            spikes = self.train_spikes
            labels = self.train_labels
        elif partition == "val":
            pool = self.val_indices
            ptr = self.ptr_val
            spikes = self.val_spikes
            labels = self.val_labels
        else:
            pool = self.test_indices
            ptr = self.ptr_test
            spikes = self.test_spikes
            labels = self.test_labels

        if ptr >= len(pool):
            return None, None

        end_ptr = min(ptr + num_samples, len(pool))
        batch_indices = pool[ptr:end_ptr]

        # Extract spike data: each sample has num_steps rows in the spike array
        # batch_indices are sample indices, need to map to spike row indices
        spike_rows = []
        batch_labels = []
        for sample_idx in batch_indices:
            start_row = int(sample_idx) * self.num_steps
            end_row = start_row + self.num_steps
            spike_rows.append(spikes[start_row:end_row])
            batch_labels.append(labels[int(sample_idx)])

        # Stack spike data: shape will be (num_samples * num_steps, features)
        spike_data = np.vstack(spike_rows)
        
        # Extend labels to match spike data (repeat each label for num_steps)
        extended_labels = np.repeat(batch_labels, self.num_steps)

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
        target_sum = (self.pixel_size**2) * 0.1 # very dubious stuff
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

    def _get_image_params(self, partition):
        """Get parameters that affect image generation (dataset-specific)."""
        params = {
            "dataset": self.dataset,
            "pixel_size": self.pixel_size,
            "partition": partition,
        }
        # Add dataset-specific parameters
        if self.dataset == "geomfig":
            params.update({
                "num_classes": self.num_classes,
                "which_classes": sorted(self.which_classes) if self.which_classes else None,
                "noise_var": getattr(self, "noise_var", None),
                "tri_size": getattr(self, "tri_size", None),
                "tri_thick": getattr(self, "tri_thick", None),
                "cir_size": getattr(self, "cir_size", None),
                "cir_thick": getattr(self, "cir_thick", None),
                "sqr_size": getattr(self, "sqr_size", None),
                "sqr_thick": getattr(self, "sqr_thick", None),
                "x_size": getattr(self, "x_size", None),
                "x_thick": getattr(self, "x_thick", None),
                "clamp_min": getattr(self, "clamp_min", None),
                "clamp_max": getattr(self, "clamp_max", None),
                "jitter": getattr(self, "jitter", None),
                "jitter_amount": getattr(self, "jitter_amount", None),
            })
        return params

    def _get_spike_params(self, partition):
        """Get parameters that affect spike generation."""
        params = {
            "dataset": self.dataset,
            "pixel_size": self.pixel_size,
            "num_steps": self.num_steps,
            "gain": self.gain,
            "offset": self.offset,
            "first_spike_time": self.first_spike_time,
            "time_var_input": self.time_var_input,
            "partition": partition,
        }
        return params

    def _get_image_cache_path(self, partition):
        """Generate cache file path for image data based on dataset parameters."""
        cache_dir = os.path.join(self.data_dir, "image_cache")
        os.makedirs(cache_dir, exist_ok=True)
        params = self._get_image_params(partition)
        cache_hash = params_hash(params)
        return os.path.join(cache_dir, f"{self.dataset}_{partition}_{cache_hash}.npz")

    def _get_spike_cache_path(self, partition):
        """Generate cache file path for spike data based on dataset parameters."""
        cache_dir = os.path.join(self.data_dir, "spike_cache")
        os.makedirs(cache_dir, exist_ok=True)
        params = self._get_spike_params(partition)
        cache_hash = params_hash(params)
        return os.path.join(cache_dir, f"{self.dataset}_{partition}_{cache_hash}.npz")

    def _save_images_cache(self, partition, images, labels):
        """Save images and labels to cache."""
        cache_path = self._get_image_cache_path(partition)
        try:
            # Convert torch tensors to numpy for saving
            images_np = images.numpy() if isinstance(images, torch.Tensor) else images
            np.savez_compressed(cache_path, images=images_np, labels=labels)
            print(f"Saved {partition} images and labels to cache: {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save image cache: {e}")

    def _load_images_cache(self, partition):
        """Load images and labels from cache if available."""
        cache_path = self._get_image_cache_path(partition)
        if not os.path.exists(cache_path):
            return None, None
        
        try:
            data = np.load(cache_path)
            images_np = data["images"]
            labels = data["labels"]
            # Convert back to torch tensors
            images = torch.from_numpy(images_np) if isinstance(images_np, np.ndarray) else images_np
            print(f"Loaded {partition} images and labels from cache: {cache_path}")
            return images, labels
        except Exception as e:
            print(f"Failed to load image cache: {e}")
            return None, None

    def _load_or_convert_spikes(self, partition, images, labels):
        """
        Load spikes with priority: cached spikes → cached images → regenerate.
        Returns (spikes, labels).
        """
        spike_cache_path = self._get_spike_cache_path(partition)
        expected_rows = len(images) * self.num_steps
        
        # Priority 1: Try to load spikes from cache
        if os.path.exists(spike_cache_path):
            try:
                data = np.load(spike_cache_path)
                cached_spikes = data["spikes"]
                cached_labels = data["labels"]
                
                # Verify both spikes and labels match
                if (cached_spikes.shape[0] == expected_rows and 
                    len(cached_labels) == len(labels)):
                    print(f"Using cached {partition} spikes")
                    return cached_spikes, cached_labels
                else:
                    print(f"Spike cache mismatch, checking image cache...")
            except Exception as e:
                print(f"Failed to load spike cache: {e}, checking image cache...")
        
        # Priority 2: Try to load images from cache and convert to spikes
        cached_images, cached_labels = self._load_images_cache(partition)
        if cached_images is not None and len(cached_images) == len(images):
            print(f"Using cached {partition} images, converting to spikes...")
            spikes = self._convert_images_to_spikes(cached_images)
            # Save the newly generated spikes
            try:
                np.savez_compressed(spike_cache_path, spikes=spikes, labels=cached_labels)
                print(f"Saved {partition} spikes to cache")
            except Exception as e:
                print(f"Warning: Failed to save spike cache: {e}")
            return spikes, cached_labels
        
        # Priority 3: Convert provided images to spikes (images not in cache)
        print(f"Converting {partition} images to spikes (no cache found)...")
        spikes = self._convert_images_to_spikes(images)
        
        # Save images to cache for future use
        self._save_images_cache(partition, images, labels)
        
        # Save spikes to cache
        try:
            np.savez_compressed(spike_cache_path, spikes=spikes, labels=labels)
            print(f"Saved {partition} spikes to cache")
        except Exception as e:
            print(f"Warning: Failed to save spike cache: {e}")
        
        return spikes, labels

    def reset_partition(self, partition="all"):
        if partition in ("all", "train"):
            self.ptr_train = 0
        if partition in ("all", "val"):
            self.ptr_val = 0
        if partition in ("all", "test"):
            self.ptr_test = 0


def load_image_batch(
    streamer,
    batch_size,
    partition="train",
):
    """
    Wrapper to load a batch from the streamer.
    start_idx and n_x are provided for compatibility but handled by the streamer.
    """
    return streamer.get_batch(batch_size, partition=partition)

def params_hash(params: dict) -> str:
    j = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(j.encode("utf-8")).hexdigest()


def normalize_image(img, target_sum=1.0): # maybe remove this function
    current_sum = img.sum()
    return img * (target_sum / current_sum) if current_sum > 0 else img


def _load_notmnist_deeplake(transform, train_ratio, val_ratio, test_ratio, rng):
    """
    Load NotMNIST from Deeplake and return torch tensors for train/test splits.
    """
    import deeplake
    ds = deeplake.load("hub://activeloop/not-mnist-small", read_only=True)

    dataset = []

    for sample in ds:
        try:
            img_np = sample["images"].numpy()
            lbl = sample["labels"].numpy()
        except KeyError as exc:  # pragma: no cover - dataset schema check
            raise KeyError(
                "NotMNIST sample missing expected 'images' or 'labels' tensors."
            ) from exc
        label_int = int(np.asarray(lbl).flatten()[0])

        img_np = np.asarray(img_np)
        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]

        img_pil = Image.fromarray(img_np.astype(np.uint8), mode="L")
        img_tensor = transform(img_pil)
        dataset.append((img_tensor, label_int))
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
    n_test  = n - n_train - n_val 
    # split train set into train and val by ratio
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(*zip(*dataset)),
        [n_train, n_val, n_test],
        generator=rng
    )
    return train_ds.dataset["images"], train_ds.dataset["labels"], val_ds.dataset["images"], val_ds.dataset["labels"], test_ds.dataset["images"], test_ds.dataset["labels"]