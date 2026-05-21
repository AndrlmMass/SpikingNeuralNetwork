from torchvision import datasets, transforms
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import warnings
import numpy as np
import torch
import os
from PIL import Image

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


def load_fcx1():
    train = np.load("data/fcx1/X_fcx1_train.npz")
    test = np.load("data/fcx1/X_fcx1_test.npz")

    X_train = train["x"]
    y_train = train["y"]

    X_test = test["x"]
    y_test = test["y"]

    return X_train, y_train, X_test, y_test


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
        max_rate_hz=67.0,
        train_count=None,
        val_count=None,
        test_count=None,
        random_seed=0,
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
        self.max_rate_hz = max_rate_hz
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
        elif self.dataset == "fcx1":
            # load spike data
            (
                self.X_train_spikes,
                self.y_train_spikes,
                self.X_test_spikes,
                self.y_test_spikes,
            ) = load_fcx1()

            # save data specs
            self.len_train = self.X_train_spikes.shape[0]
            self.len_test = self.X_test_spikes.shape[0]
            train_label_targets = self.y_train_spikes
            test_label_targets = self.y_test_spikes

            # reshape data to match 10x10 resolution
            self.X_train_spikes = self.X_train_spikes[:, :100]
            self.X_test_spikes = self.X_test_spikes[:, :100]
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
        self.indices = np.arange(total)

        if self.dataset != "fcx1":
            # Shuffle indices for random access over the combined (train+test) space
            # Use a seeded RNG so the train/val/test split is reproducible
            _split_rng = np.random.default_rng(random_seed)
            _split_rng.shuffle(self.indices)

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
        else:
            # initate random seed
            rng = np.random.default_rng(random_seed)

            # intiate partition start index and step size
            self.prt_train = 0
            self.prt_test = 0
            self.partition_increment = int(self.num_steps * self.batch_size)

            # create arrays of indices
            ind_train = np.arange(self.len_train)
            ind_test = np.arange(self.len_test)

            # define indices
            train_indices = ind_train // self.partition_increment
            test_indices = ind_test // self.partition_increment

            # get unique groups
            train_uniq = np.unique(train_indices)
            test_uniq = np.unique(test_indices)

            # permutate the unique list
            perm_train = rng.permutation(train_uniq)
            perm_test = rng.permutation(test_uniq)

            # extend again based on length per sequence variable
            self.train_indices = np.concatenate(
                [np.where(train_indices == g)[0] for g in perm_train]
            )
            self.test_indices = np.concatenate(
                [np.where(test_indices == g)[0] for g in perm_test]
            )

            # shuffle data based on new indices
            self.X_train_spikes = self.X_train_spikes[self.train_indices]
            self.y_train_spikes = self.y_train_spikes[self.train_indices]
            self.X_test_spikes = self.X_test_spikes[self.test_indices]
            self.y_test_spikes = self.y_test_spikes[self.test_indices]

    def show_preview(self, num_samples: int = 9, save_path: str | None = None):
        """
        Display a small grid of cached images so the user can confirm the dataset.
        """
        if self.dataset == "fcx1":
            print("image preview is not available")
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

    def preview_pipeline(self, digit_idx: int = 0, out_dir: str | None = None) -> None:
        """Save three separate PDFs showing one digit at each pipeline stage."""
        if self.dataset == "fcx1":
            print("preview_pipeline is not available for fcx1")
            return

        import matplotlib.pyplot as plt

        if out_dir is None:
            out_dir = os.path.join("results", "data", "MNIST")
        os.makedirs(out_dir, exist_ok=True)

        img_tensor = self.train_images[digit_idx]  # (1, H, W)
        img_batch = img_tensor.unsqueeze(0)  # (1, 1, H, W)

        def _save_image(arr, fname, cmap, vmin=None, vmax=None):
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.axis("off")
            ax.set_xlabel("Time (ms)", fontsize=8, color="white")
            ax.set_ylabel("Neuron", fontsize=8, color="white")
            fig.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.99)
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved → {path}")

        # 1 — raw
        _save_image(img_tensor.squeeze().numpy(), "1_raw.pdf", "gray", 0, 1)

        # 2 — gabor
        with torch.no_grad():
            gabor = self.gabor_pack_quadrants(img_batch)
        _save_image(gabor[0, 0].numpy(), "2_gabor.pdf", "gray", 0, 1)

        # 3 — spike raster
        torch.manual_seed(0)
        with torch.no_grad():
            p = (gabor * self.max_rate_hz * (1.0 / 1000.0)).clamp(0.0, 1.0)
            spikes = (torch.rand((self.num_steps,) + p.shape) < p).float()
            spikes = spikes.squeeze(2).squeeze(1)  # (T, H, W)
            spike_matrix = spikes.view(self.num_steps, -1).numpy()  # (T, N_pixels)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(
            spike_matrix.T,
            aspect="auto",
            cmap="binary_r",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Neuron", fontsize=16)
        ax.tick_params(
            axis="both", which="both", length=0, labelbottom=False, labelleft=False
        )
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.99)
        path = os.path.join(out_dir, "3_spikes.pdf")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved → {path}")

    def get_batch(self, start_idx, num_samples, partition="train"):
        """
        Load a batch of image samples from the specified partition.
        Returns (spike_data, labels) or (None, None) if no more data.
        """

        if self.dataset != "fcx1":
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
        else:
            # identify desired data
            if partition == "train":
                spike_data = self.X_train_spikes
                extended_labels = self.y_train_spikes
            else:
                spike_data = self.X_test_spikes
                extended_labels = self.y_test_spikes

            # define start and stop
            start = self.partition_increment
            stop = start + self.batch_size * self.num_steps

            # fetch partition
            spike_data = spike_data[start:stop]
            extended_labels = extended_labels[start:stop]

            # update partition
            self.partition_increment = stop

            return spike_data, extended_labels

    def _convert_images_to_spikes(self, images):
        """
        Convert image batch to Poisson spike trains.
        Assumes input intensities in range [0, 1].
        """
        # apply gabor filters
        images = self.gabor_pack_quadrants(images)

        # Compute spike probability per timestep
        # r_max = 67 Hz
        # dt assumed 1 ms
        dt_ms = 1.0
        p = images * self.max_rate_hz * (dt_ms / 1000.0)  # (B, 1, H, W) or (B, H, W)
        p = p.clamp(0.0, 1.0)  # safety cap

        # Generate Poisson spikes
        # Output shape: (T, B, C, H, W) -> where C = 1 -> (100,100,1,15,15)
        spikes = (torch.rand((self.num_steps,) + p.shape, device=p.device) < p).float()

        # Remove channel dim if present -> (100,100,15,15)
        spikes = spikes.squeeze(2)

        # Flatten spatial dims
        spikes_flat = spikes.flatten(start_dim=2)  # (T, B, N_pixels) -> (100,100,225)

        # Reshape to match your expected format (T*B, N_pixels) -> (10000,225)
        spikes = spikes_flat.transpose(0, 1).reshape(
            spikes_flat.shape[1] * spikes_flat.shape[0], spikes_flat.shape[2]
        )

        return spikes.cpu().numpy()

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

    def gabor_pack_quadrants(self, images):
        import math

        """
        images: (B,1,H,W) in [0,1]
        returns: (B,1,H,W) with 4 gabor maps packed into quadrants
        """

        assert images.dim() == 4 and images.shape[1] == 1
        B, _, H, W = images.shape
        device, dtype = images.device, images.dtype

        # # plot first image before gabor process
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(images[0].cpu().numpy().squeeze(0))
        # plt.show()
        # plt.close(fig)

        # --- resolution-aware scaling ---
        S = int(math.sqrt(H * W))  # auto adapts (28 for MNIST)
        ksize = max(5, S // 4)  # kernel ~ 1/4 of image width
        if ksize % 2 == 0:
            ksize += 1  # force odd

        sigma = ksize / 3
        lambd = ksize / 2
        gamma = 0.5
        psi = 0.0

        # --- build gabor kernels ---
        r = ksize // 2
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij",
        )

        def gabor(theta):
            ct, st = math.cos(theta), math.sin(theta)
            x_theta = x * ct + y * st
            y_theta = -x * st + y * ct

            gauss = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2))
            wave = torch.cos(2 * math.pi * x_theta / lambd + psi)
            k = gauss * wave
            k = k - k.mean()
            k = k / (k.norm() + 1e-8)
            return k

        thetas = [0.0, math.pi / 2, math.pi / 4, 3 * math.pi / 4]
        kernels = torch.stack([gabor(t) for t in thetas], dim=0)
        kernels = kernels[:, None, :, :]  # (4,1,k,k)

        # --- convolve ---
        pad = ksize // 2
        resp = F.conv2d(images, kernels, padding=pad)  # (B,4,H,W)

        # unsigned response
        resp = resp.abs()

        # Normalize per channel
        rmin = resp.amin(dim=(2, 3), keepdim=True)  # only spatial dims
        rmax = resp.amax(dim=(2, 3), keepdim=True)
        resp = (resp - rmin) / (rmax - rmin + 1e-8)

        # --- pack into quadrants ---
        h1 = H // 2
        h2 = H - h1
        w1 = W // 2
        w2 = W - w1

        packed = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        tl = F.interpolate(
            resp[:, 0:1], size=(h1, w1), mode="bilinear", align_corners=False
        )
        tr = F.interpolate(
            resp[:, 1:2], size=(h1, w2), mode="bilinear", align_corners=False
        )
        bl = F.interpolate(
            resp[:, 2:3], size=(h2, w1), mode="bilinear", align_corners=False
        )
        br = F.interpolate(
            resp[:, 3:4], size=(h2, w2), mode="bilinear", align_corners=False
        )

        packed[:, :, :h1, :w1] = tl
        packed[:, :, :h1, w1:] = tr
        packed[:, :, h1:, :w1] = bl
        packed[:, :, h1:, w1:] = br

        # plot same image after gabor processing
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(packed[0].cpu().numpy().squeeze(0))
        # plt.show()
        # plt.close(fig)

        return packed


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

    def get_batch(self, start_idx, num_samples, partition="train"):
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
