from torchvision import datasets, transforms
from snntorch import spikegen
from tqdm import tqdm
import warnings
import numpy as np
import torch
import os
import json
from PIL import Image
from .geomfig import gen_triangle, gen_circle, gen_square, gen_x
import hashlib

os.environ["TQDM_DISABLE"] = "True"


def normalize_image(img, target_sum=1.0):
    current_sum = img.sum()
    return img * (target_sum / current_sum) if current_sum > 0 else img


warnings.filterwarnings("ignore")



def _normalize_split_value(split_value):
    """Convert Deeplake split tensor values to lowercase strings."""
    if split_value is None:
        return "train"
    if isinstance(split_value, bytes):
        return split_value.decode("utf-8").lower()
    if isinstance(split_value, str):
        return split_value.lower()
    if hasattr(split_value, "tolist"):
        return _normalize_split_value(split_value.tolist())
    if isinstance(split_value, (list, tuple)) and split_value:
        return _normalize_split_value(split_value[0])
    return str(split_value).lower()


def _load_notmnist_deeplake(transform):
    """
    Load NotMNIST from Deeplake and return torch tensors for train/test splits.
    """
    try:
        import deeplake
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading NotMNIST requires the 'deeplake' package. "
            "Install it via 'pip install deeplake'."
        ) from exc

    try:
        ds = deeplake.load("hub://activeloop/not-mnist-small", read_only=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load NotMNIST from Deeplake ({exc})") from exc

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for sample in ds:
        try:
            img_np = sample["images"].numpy()
            lbl = sample["labels"].numpy()
        except KeyError as exc:  # pragma: no cover - dataset schema check
            raise KeyError(
                "NotMNIST sample missing expected 'images' or 'labels' tensors."
            ) from exc

        split_value = None
        if "split" in sample:
            try:
                split_value = sample["split"].numpy()
            except Exception:
                split_value = None

        split_name = _normalize_split_value(split_value)
        label_int = int(np.asarray(lbl).flatten()[0])

        img_np = np.asarray(img_np)
        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]
        if img_np.ndim != 2:
            raise ValueError(
                f"Unexpected NotMNIST image shape {img_np.shape}; expected (H, W)."
            )

        img_pil = Image.fromarray(img_np.astype(np.uint8), mode="L")
        img_tensor = transform(img_pil)

        if split_name in {"test", "validation", "val"}:
            test_images.append(img_tensor)
            test_labels.append(label_int)
        else:
            train_images.append(img_tensor)
            train_labels.append(label_int)

    if not train_images:
        raise RuntimeError("Failed to load any NotMNIST samples from Deeplake.")

    if not test_images:
        # Dataset may not include a split tensor; perform an internal split.
        total_samples = len(train_images)
        if total_samples < 2:
            raise RuntimeError(
                "NotMNIST dataset too small to create train/test splits."
            )
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(total_samples)
        test_size = max(1, int(0.1 * total_samples))
        test_indices = perm[:test_size]
        train_indices = perm[test_size:]

        test_images = [train_images[idx] for idx in test_indices]
        test_labels = [train_labels[idx] for idx in test_indices]
        train_images = [train_images[idx] for idx in train_indices]
        train_labels = [train_labels[idx] for idx in train_indices]

    train_images = torch.stack(train_images)
    test_images = torch.stack(test_images)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    return train_images, train_labels, test_images, test_labels


class ImageDataStreamer:
    """
    Eagerly loads and preprocesses the configured dataset into RAM so the
    training loop can request batches without incurring per-iteration
    torchvision/disk overhead.
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
        dataset: str = "mnist",
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.num_steps = num_steps
        self.gain = gain
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
            "fashionmnist": datasets.FashionMNIST,
            "fmnist": datasets.FashionMNIST,
            "fashion": datasets.FashionMNIST,
        }
        if self.dataset == "notmnist":
            (
                self.train_images,
                self.train_labels,
                self.test_images,
                self.test_labels,
            ) = _load_notmnist_deeplake(transform)
            self.len_train = len(self.train_images)
            self.len_test = len(self.test_images)
            train_label_targets = self.train_labels
            test_label_targets = self.test_labels
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

            # Eagerly materialize tensors for faster repeated access
            self.len_train = len(mnist_train)
            self.len_test = len(mnist_test)
            print(f"Loading {self.dataset.upper()} train set into RAM...")
            # Enable progress bars for this block
            try:
                os.environ["TQDM_DISABLE"] = "False"
            except Exception:
                pass
            self.train_images = torch.zeros(
                (self.len_train, 1, pixel_size, pixel_size), dtype=torch.float32
            )
            self.train_labels = np.zeros(self.len_train, dtype=np.int64)
            for idx in tqdm(
                range(self.len_train),
                total=self.len_train,
                desc=f"Loading {self.dataset} train",
                leave=False,
            ):
                img, lbl = mnist_train[idx]
                self.train_images[idx] = img
                self.train_labels[idx] = int(lbl)

            print(f"Loading {self.dataset.upper()} test set into RAM...")
            self.test_images = torch.zeros(
                (self.len_test, 1, pixel_size, pixel_size), dtype=torch.float32
            )
            self.test_labels = np.zeros(self.len_test, dtype=np.int64)
            for idx in tqdm(
                range(self.len_test),
                total=self.len_test,
                desc=f"Loading {self.dataset} test",
                leave=False,
            ):
                img, lbl = mnist_test[idx]
                self.test_images[idx] = img
                self.test_labels[idx] = int(lbl)

            train_label_targets = np.array(getattr(mnist_train, "targets", []))
            test_label_targets = np.array(getattr(mnist_test, "targets", []))

        total = self.len_train + self.len_test

        # Shuffle indices for random access over the combined (train+test) space
        self.indices = np.arange(total)
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

        # Lightweight summary (use dataset targets if available)
        print(f"Found {total} image samples")
        try:
            tr_t = np.array(train_label_targets)
            te_t = np.array(test_label_targets)
            if tr_t.size or te_t.size:
                lab_dist = np.bincount(np.concatenate([tr_t, te_t]).astype(int))
                print(f"Label distribution: {lab_dist}")
        except Exception:
            pass

    def show_preview(self, num_samples: int = 9, save_path: str | None = None):
        """
        Display a small grid of cached images so the user can confirm the dataset.
        """
        if self.train_images is None or len(self.train_images) == 0:
            print("No cached images available for preview.")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Matplotlib unavailable; skipping dataset preview ({exc})")
            return

        num = max(1, min(int(num_samples), len(self.train_images)))
        images = self.train_images[:num].cpu().numpy().squeeze(1)
        labels = self.train_labels[:num]

        cols = int(np.ceil(np.sqrt(num)))
        rows = int(np.ceil(num / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = np.atleast_2d(axes)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx < num:
                ax.imshow(images[idx], cmap="gray")
                ax.set_title(f"Label {labels[idx]}")
            ax.axis("off")

        fig.suptitle(f"{self.dataset.upper()} preview ({num} samples)", fontsize=14)
        plt.tight_layout()

        try:
            if save_path is None:
                os.makedirs("plots", exist_ok=True)
                save_path = os.path.join("plots", f"{self.dataset}_preview.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            print(f"Dataset preview saved to {save_path}")
        except Exception as exc:
            print(f"Failed to save dataset preview ({exc})")

        plt.show()
        plt.close(fig)

    def get_batch(self, num_samples, partition="train"):
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

        # Pull batch data from cached tensors in RAM
        images_list = []
        labels_list = []
        for gi in batch_indices:
            if gi < self.len_train:
                images_list.append(self.train_images[int(gi)])
                labels_list.append(self.train_labels[int(gi)])
            else:
                idx = int(gi - self.len_train)
                images_list.append(self.test_images[idx])
                labels_list.append(self.test_labels[idx])

        batch_images = torch.stack(images_list)
        batch_labels = np.asarray(labels_list, dtype=np.int64)

        # Convert to spikes
        spike_data = self._convert_images_to_spikes(batch_images)

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


class GeomfigDataStreamer:
    """
    Streamer for pre-generated geomfig spike data.
    Provides batched access to train/val/test splits.
    """

    def __init__(
        self,
        data_train,
        labels_train,
        data_val,
        labels_val,
        data_test,
        labels_test,
        batch_size=100,
        num_steps=100,
    ):
        """
        Initialize with pre-generated spike arrays.

        Args:
            data_train: (T_train, N_x) numpy array of spikes
            labels_train: (T_train,) numpy array of labels
            data_val: (T_val, N_x) numpy array of spikes
            labels_val: (T_val,) numpy array of labels
            data_test: (T_test, N_x) numpy array of spikes
            labels_test: (T_test,) numpy array of labels
            batch_size: Default batch size (for compatibility)
            num_steps: Number of timesteps per image sample (needed to convert num_samples to timesteps)
        """
        # Determine N_x from first available array
        n_x = 225  # default
        if data_train is not None and len(data_train) > 0:
            n_x = data_train.shape[1]
        elif data_val is not None and len(data_val) > 0:
            n_x = data_val.shape[1]
        elif data_test is not None and len(data_test) > 0:
            n_x = data_test.shape[1]

        self.data_train = (
            data_train
            if data_train is not None and len(data_train) > 0
            else np.zeros((0, n_x), dtype=np.int8)
        )
        self.labels_train = (
            labels_train
            if labels_train is not None and len(labels_train) > 0
            else np.zeros((0,), dtype=np.int32)
        )
        self.data_val = (
            data_val
            if data_val is not None and len(data_val) > 0
            else np.zeros((0, n_x), dtype=np.int8)
        )
        self.labels_val = (
            labels_val
            if labels_val is not None and len(labels_val) > 0
            else np.zeros((0,), dtype=np.int32)
        )
        self.data_test = (
            data_test
            if data_test is not None and len(data_test) > 0
            else np.zeros((0, n_x), dtype=np.int8)
        )
        self.labels_test = (
            labels_test
            if labels_test is not None and len(labels_test) > 0
            else np.zeros((0,), dtype=np.int32)
        )
        self.batch_size = batch_size
        self.num_steps = num_steps

        # Pointers for each partition
        self.ptr_train = 0
        self.ptr_val = 0
        self.ptr_test = 0

    def get_batch(self, num_samples, partition="train"):
        """
        Load a batch of spike data from the specified partition.
        Returns (spike_data, labels) or (None, None) if no more data.

        Args:
            start_idx: Starting index (ignored, uses internal pointer)
            num_samples: Number of image samples to return (will be converted to timesteps)
            partition: "train", "val", or "test"

        Returns:
            (spike_data, labels) where:
                spike_data: (num_samples * num_steps, N_x) numpy array
                labels: (num_samples * num_steps,) numpy array
        """
        if partition == "train":
            data = self.data_train
            labels = self.labels_train
            ptr = self.ptr_train
        elif partition == "val":
            data = self.data_val
            labels = self.labels_val
            ptr = self.ptr_val
        else:  # test
            data = self.data_test
            labels = self.labels_test
            ptr = self.ptr_test

        if ptr >= len(data):
            return None, None

        # Convert num_samples (image samples) to timesteps
        num_timesteps = num_samples * self.num_steps
        end_ptr = min(ptr + num_timesteps, len(data))
        batch_data = data[ptr:end_ptr]
        batch_labels = labels[ptr:end_ptr]

        # Update pointer
        if partition == "train":
            self.ptr_train = end_ptr
        elif partition == "val":
            self.ptr_val = end_ptr
        else:
            self.ptr_test = end_ptr

        return batch_data, batch_labels

    def get_total_samples(self):
        """Return total number of available training timesteps."""
        return len(self.data_train)

    def reset_partition(self, partition="all"):
        """Reset the pointer for the specified partition(s)."""
        if partition in ("all", "train"):
            self.ptr_train = 0
        if partition in ("all", "val"):
            self.ptr_val = 0
        if partition in ("all", "test"):
            self.ptr_test = 0



# ============================================================================
# Geometric Figures (geomfig) dataset generation
# ============================================================================


def _geomfig_generate_one(
    cls_id: int,
    pixel_size: int,
    noise_var: float,
    jitter: bool = False,
    jitter_amount: float = 0.05,
    worker_seed: int | None = None,
) -> np.ndarray:
    """
    Generate a single geometric figure image in [0,1] of shape (pixel_size, pixel_size).
    Class mapping: 0=triangle, 1=circle, 2=square, 3=x.
    """
    # Optional per-task seeding for parallel generation reproducibility
    if worker_seed is not None:
        try:
            np.random.seed(int(worker_seed))
        except Exception:
            pass
    # Base params from legacy usage
    tri_size, tri_thick = 0.7, 250
    cir_size, cir_thick = 0.7, 3
    sqr_size, sqr_thick = 0.6, 4
    x_size, x_thick = 0.8, 350

    def j(val, rel=True, clamp_min=None, clamp_max=None):
        if not jitter:
            return val
        delta = (np.random.rand() * 2 - 1) * jitter_amount
        out = val * (1.0 + delta) if rel else val + delta
        if clamp_min is not None:
            out = max(clamp_min, out)
        if clamp_max is not None:
            out = min(clamp_max, out)
        return out

    if cls_id == 0:  # triangle
        img = gen_triangle(
            input_dims=pixel_size,
            triangle_size=float(j(tri_size, rel=True, clamp_min=0.05, clamp_max=0.95)),
            triangle_thickness=int(max(1, j(tri_thick, rel=True))),
            noise_rand=True,
            noise_variance=float(max(0.0, noise_var)),
        )
    elif cls_id == 1:  # circle
        img = gen_circle(
            input_dims=pixel_size,
            circle_size=float(j(cir_size, rel=True, clamp_min=0.05, clamp_max=0.95)),
            circle_thickness=int(max(1, j(cir_thick, rel=True))),
            noise_rand=True,
            noise_variance=float(max(0.0, noise_var)),
        )
    elif cls_id == 2:  # square
        img = gen_square(
            input_dims=pixel_size,
            square_size=float(j(sqr_size, rel=True, clamp_min=0.05, clamp_max=0.95)),
            square_thickness=int(max(1, j(sqr_thick, rel=True))),
            noise_rand=True,
            noise_variance=float(max(0.0, noise_var)),
        )
    else:  # 3: x
        img = gen_x(
            input_dims=pixel_size,
            x_size=float(j(x_size, rel=True, clamp_min=0.05, clamp_max=0.95)),
            x_thickness=int(max(1, j(x_thick, rel=True))),
            noise_rand=True,
            noise_variance=float(max(0.0, noise_var)),
        )
    # Ensure numeric array in [0,1]
    img = np.asarray(img, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def create_geomfig_data(
    pixel_size: int,
    num_steps: int,
    gain: float,
    train_count: int,
    val_count: int,
    test_count: int,
    noise_var: float = 0.02,
    jitter: bool = False,
    jitter_amount: float = 0.05,
    seed: int = 42,
    num_workers: int | None = None,
):
    """
    Generate geometric figures dataset and convert to spike trains.
    Returns time-series arrays shaped (total_timesteps, pixel_size*pixel_size)
    with per-timestep integer labels.
    """
    rng = np.random.default_rng(seed)
    n_classes = 4

    def make_split(total_samples: int, split_name: str):
        # Round down to multiple of 4 for class balance
        total = max(0, (int(total_samples) // n_classes) * n_classes)
        if total == 0:
            return np.zeros((0, pixel_size * pixel_size), dtype=np.int8), np.zeros(
                (0,), dtype=np.int32
            )
        imgs = np.zeros((total, pixel_size, pixel_size), dtype=np.float32)
        labels = np.zeros((total,), dtype=np.int32)
        # Cycle through classes 0..3
        cls_seq = np.array([i % n_classes for i in range(total)], dtype=np.int32)
        labels[:] = cls_seq

        # Enable progress bars for generation
        try:
            os.environ["TQDM_DISABLE"] = "False"
        except Exception:
            pass

        # Decide on parallel vs sequential
        use_workers = 1
        try:
            if num_workers is not None:
                use_workers = max(1, int(num_workers))
        except Exception:
            use_workers = 1

        if use_workers <= 1:
            for i in tqdm(
                range(total),
                total=total,
                desc=f"Generating geomfig {split_name}",
                leave=False,
            ):
                imgs[i] = _geomfig_generate_one(
                    cls_id=int(cls_seq[i]),
                    pixel_size=pixel_size,
                    noise_var=noise_var,
                    jitter=jitter,
                    jitter_amount=jitter_amount,
                )
        else:
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed
            except Exception:
                ProcessPoolExecutor = None
                as_completed = None

            if ProcessPoolExecutor is None:
                # Fallback to sequential if concurrent.futures unavailable
                for i in tqdm(
                    range(total),
                    total=total,
                    desc=f"Generating geomfig {split_name}",
                    leave=False,
                ):
                    imgs[i] = _geomfig_generate_one(
                        cls_id=int(cls_seq[i]),
                        pixel_size=pixel_size,
                        noise_var=noise_var,
                        jitter=jitter,
                        jitter_amount=jitter_amount,
                    )
            else:
                # Pre-draw independent seeds for each sample to avoid identical substreams across forks
                seeds = rng.integers(0, 2**31 - 1, size=total, dtype=np.int64)
                with ProcessPoolExecutor(max_workers=use_workers) as ex:
                    futures = {}
                    for i in range(total):
                        futures[
                            ex.submit(
                                _geomfig_generate_one,
                                int(cls_seq[i]),
                                int(pixel_size),
                                float(noise_var),
                                bool(jitter),
                                float(jitter_amount),
                                int(seeds[i]),
                            )
                        ] = i
                    for fut in tqdm(
                        as_completed(futures),
                        total=total,
                        desc=f"Generating geomfig {split_name} (x{use_workers})",
                        leave=False,
                    ):
                        i = futures[fut]
                        try:
                            imgs[i] = fut.result()
                        except Exception:
                            # In case of worker failure, fallback to inline generation for this sample
                            imgs[i] = _geomfig_generate_one(
                                cls_id=int(cls_seq[i]),
                                pixel_size=pixel_size,
                                noise_var=noise_var,
                                jitter=jitter,
                                jitter_amount=jitter_amount,
                            )
        # Shuffle for randomness
        idx = rng.permutation(total)
        imgs = imgs[idx]
        labels = labels[idx]

        # Poisson/Bernoulli rate coding to spikes: (total * num_steps, N_x)
        N_x = pixel_size * pixel_size
        probs = np.clip(imgs.reshape(total, N_x) * float(gain), 0.0, 1.0)
        spikes = (rng.random((total, num_steps, N_x)) < probs[:, None, :]).astype(
            np.int8
        )
        spikes_ts = spikes.reshape(total * num_steps, N_x)
        labels_ts = np.repeat(labels, num_steps).astype(np.int32)
        return spikes_ts, labels_ts

    data_train, labels_train = make_split(train_count, "train")
    data_val, labels_val = make_split(val_count, "val")
    data_test, labels_test = make_split(test_count, "test")

    return (
        data_train,
        labels_train,
        data_val,
        labels_val,
        data_test,
        labels_test,
    )


def load_or_create_geomfig_data(
    pixel_size: int,
    num_steps: int,
    gain: float,
    train_count: int,
    val_count: int,
    test_count: int,
    noise_var: float = 0.02,
    noise_mean: float = 0.0,
    jitter: bool = False,
    jitter_amount: float = 0.05,
    seed: int = 42,
    force_recreate: bool = False,
    cache_root: str = os.path.join("data", "cache", "geomfig"),
    num_workers: int | None = None,
):
    """
    Load geomfig spike data from disk cache if available; otherwise generate and cache it.
    """
    os.makedirs(cache_root, exist_ok=True)
    meta = {
        "pixel_size": int(pixel_size),
        "num_steps": int(num_steps),
        "gain": float(gain),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "noise_var": float(noise_var),
        "noise_mean": float(noise_mean),
        "jitter": bool(jitter),
        "jitter_amount": float(jitter_amount),
        "seed": int(seed),
        "version": 1,
    }
    key_json = json.dumps(meta, sort_keys=True)
    key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()  # nosec - cache key
    cache_path = os.path.join(cache_root, f"geomfig_{key_hash}.npz")

    if (not force_recreate) and os.path.exists(cache_path):
        try:
            npz = np.load(cache_path, allow_pickle=False)
            data_train = npz["data_train"]
            labels_train = npz["labels_train"]
            data_val = npz["data_val"]
            labels_val = npz["labels_val"]
            data_test = npz["data_test"]
            labels_test = npz["labels_test"]
            print(f"Geomfig cache loaded: {os.path.basename(cache_path)}")
            return (
                data_train,
                labels_train,
                data_val,
                labels_val,
                data_test,
                labels_test,
            )
        except Exception as exc:
            print(f"Warning: failed to load geomfig cache ({exc}); regenerating...")

    # Generate fresh and cache
    (
        data_train,
        labels_train,
        data_val,
        labels_val,
        data_test,
        labels_test,
    ) = create_geomfig_data(
        pixel_size=pixel_size,
        num_steps=num_steps,
        gain=gain,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        noise_var=noise_var,
        jitter=jitter,
        jitter_amount=jitter_amount,
        seed=seed,
        num_workers=num_workers,
    )

    try:
        np.savez_compressed(
            cache_path,
            data_train=data_train.astype(np.int8, copy=False),
            labels_train=labels_train.astype(np.int32, copy=False),
            data_val=data_val.astype(np.int8, copy=False),
            labels_val=labels_val.astype(np.int32, copy=False),
            data_test=data_test.astype(np.int8, copy=False),
            labels_test=labels_test.astype(np.int32, copy=False),
            meta=key_json,
        )
        print(f"Geomfig cache saved: {os.path.basename(cache_path)}")
    except Exception as exc:
        raise UserWarning(f"Failed to save geomfig cache ({exc})")

    return (
        data_train,
        labels_train,
        data_val,
        labels_val,
        data_test,
        labels_test,
    )


def load_image_batch(
    streamer,
    start_idx,
    batch_size,
    num_steps,
    n_x,
    partition="train",
):
    """
    Wrapper to load a batch from the streamer.
    start_idx and n_x are provided for compatibility but handled by the streamer.
    """
    return streamer.get_batch(batch_size, partition=partition)
