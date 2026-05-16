import hashlib
import json
import os

import numpy as np


class ImageDataStreamer:
    """Loads image/spike datasets and streams Poisson-encoded batches.

    For image datasets (MNIST, KMNIST, FashionMNIST, geomfig) pixel values
    are converted to Poisson spike trains on first use and cached to disk.
    For spike datasets (fcx1) the data is already binary and is streamed
    directly without re-encoding.

    Parameters
    ----------
    data_dir : str
        Root data directory (contains torchvision/, fcx1/, cache/ etc.)
    pixel_size : int
        Images are resized to pixel_size x pixel_size before encoding.
    num_steps : int
        Number of timesteps per image / per sample window.
    max_rate_hz : float
        Maximum Poisson firing rate in Hz (maps pixel intensity 1.0 → this rate).
    gain : float
        Multiplicative gain applied to spike probabilities.
    train_count : int
        Number of training images/windows to use.
    val_count : int
        Number of validation images/windows to use.
    test_count : int
        Number of test images/windows to use.
    dataset : str
        One of 'mnist', 'kmnist', 'fmnist', 'fashionmnist', 'notmnist',
        'geomfig', 'fcx1'.
    seed : int
        RNG seed for spike generation and sampling.
    noise_var : float
        Additive Gaussian noise variance applied to geomfig images.
    jitter : bool
        Whether to apply spatial jitter to geomfig images.
    jitter_amount : float
        Magnitude of spatial jitter for geomfig.
    """

    def __init__(
        self,
        data_dir: str,
        pixel_size: int,
        num_steps: int,
        max_rate_hz: float,
        gain: float,
        train_count: int,
        val_count: int,
        test_count: int,
        dataset: str,
        seed: int = 42,
        noise_var: float = 0.02,
        jitter: bool = False,
        jitter_amount: float = 0.05,
    ):
        self.data_dir = data_dir
        self.pixel_size = pixel_size
        self.num_steps = num_steps
        self.max_rate_hz = max_rate_hz
        self.gain = gain
        self.train_count = train_count
        self.val_count = val_count
        self.test_count = test_count
        self.dataset = dataset.lower()
        self.seed = seed
        self.noise_var = noise_var
        self.jitter = jitter
        self.jitter_amount = jitter_amount

        self._data = self._load_or_generate()
        self._cursors = {"train": 0, "val": 0, "test": 0}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset_partition(self, partition: str):
        self._cursors[partition] = 0

    def get_batch(self, batch_size: int, partition: str):
        data = self._data[f"data_{partition}"]
        labels = self._data[f"labels_{partition}"]

        cursor = self._cursors[partition]
        start = cursor * self.num_steps
        end = start + batch_size * self.num_steps

        if start >= len(data):
            return None, None

        end = min(end, len(data))
        self._cursors[partition] += batch_size
        return data[start:end], labels[start:end]

    def show_preview(self, num_samples: int = 9, save_path: str = None):
        import matplotlib.pyplot as plt

        data = self._data["data_train"]
        labels = self._data["labels_train"]

        unique_labels = np.unique(labels[: self.num_steps * min(500, len(labels) // self.num_steps)])
        n = min(num_samples, len(unique_labels))

        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        if n == 1:
            axes = [axes]

        for i, lbl in enumerate(unique_labels[:n]):
            # show average spike rate over first occurrence of this label
            idx = np.where(labels == lbl)[0][0]
            # round down to nearest num_steps boundary
            start = (idx // self.num_steps) * self.num_steps
            img = data[start : start + self.num_steps].mean(axis=0)
            axes[i].imshow(img.reshape(self.pixel_size, self.pixel_size), cmap="gray")
            axes[i].set_title(f"Class {lbl}", fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path)
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _meta(self) -> dict:
        return dict(
            dataset=self.dataset,
            gain=self.gain,
            jitter=self.jitter,
            jitter_amount=self.jitter_amount,
            noise_var=self.noise_var,
            num_steps=self.num_steps,
            pixel_size=self.pixel_size,
            seed=self.seed,
            test_count=self.test_count,
            train_count=self.train_count,
            val_count=self.val_count,
            version=1,
        )

    def _cache_path(self) -> str:
        meta_str = json.dumps(self._meta(), sort_keys=True)
        key = hashlib.sha1(meta_str.encode()).hexdigest()
        cache_dir = os.path.join(self.data_dir, "cache", self.dataset)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.dataset}_{key}.npz")

    def _load_or_generate(self) -> dict:
        path = self._cache_path()
        if os.path.exists(path):
            print(f"\rdata loaded from cache ({self.dataset})", end="")
            d = np.load(path, allow_pickle=True)
            return {k: d[k] for k in ["data_train", "labels_train", "data_val", "labels_val", "data_test", "labels_test"]}

        print(f"\rgenerating spike data ({self.dataset})...", end="")
        result = self._generate()
        np.savez_compressed(path, meta=json.dumps(self._meta(), sort_keys=True), **result)
        print(f"\rdata cached → {path}", end="")
        return result

    # ------------------------------------------------------------------
    # Dataset-specific loaders
    # ------------------------------------------------------------------

    def _generate(self) -> dict:
        if self.dataset == "fcx1":
            return self._load_fcx1()
        else:
            return self._generate_image_dataset()

    def _generate_image_dataset(self) -> dict:
        rng = np.random.default_rng(self.seed)
        total_train = self.train_count + self.val_count

        imgs_tr, lbls_tr = self._load_raw_images("train", total_train, rng)
        imgs_te, lbls_te = self._load_raw_images("test", self.test_count, rng)

        imgs_val = imgs_tr[self.train_count:]
        lbls_val = lbls_tr[self.train_count:]
        imgs_tr = imgs_tr[: self.train_count]
        lbls_tr = lbls_tr[: self.train_count]

        d_tr, l_tr = self._to_spikes(imgs_tr, lbls_tr, rng)
        d_val, l_val = self._to_spikes(imgs_val, lbls_val, rng)
        d_te, l_te = self._to_spikes(imgs_te, lbls_te, rng)

        return dict(
            data_train=d_tr, labels_train=l_tr,
            data_val=d_val, labels_val=l_val,
            data_test=d_te, labels_test=l_te,
        )

    def _load_raw_images(self, split: str, count: int, rng) -> tuple:
        if self.dataset == "geomfig":
            return self._generate_geomfig(count, rng)
        return self._load_torchvision(split, count, rng)

    def _load_torchvision(self, split: str, count: int, rng) -> tuple:
        import torchvision
        import torchvision.transforms as T

        name_map = {
            "mnist": "MNIST",
            "kmnist": "KMNIST",
            "fmnist": "FashionMNIST",
            "fashionmnist": "FashionMNIST",
            "fashion": "FashionMNIST",
            "notmnist": "MNIST",
        }
        tv_name = name_map.get(self.dataset, "MNIST")
        tv_class = getattr(torchvision.datasets, tv_name)

        transform = T.Compose([
            T.Resize((self.pixel_size, self.pixel_size)),
            T.ToTensor(),
        ])
        ds = tv_class(
            root=os.path.join(self.data_dir, "torchvision"),
            train=(split == "train"),
            download=True,
            transform=transform,
        )

        n = min(count, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        images = np.zeros((n, self.pixel_size ** 2), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int32)
        for i, j in enumerate(idx):
            img, lbl = ds[int(j)]
            images[i] = img.numpy().flatten()
            labels[i] = int(lbl)
        return images, labels

    def _generate_geomfig(self, count: int, rng) -> tuple:
        n_classes = 4
        per_class = count // n_classes
        images, labels = [], []
        for cls in range(n_classes):
            for _ in range(per_class):
                img = self._geomfig_one(cls, rng)
                images.append(img.flatten())
                labels.append(cls)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        idx = rng.permutation(len(images))
        return images[idx], labels[idx]

    def _geomfig_one(self, cls_id: int, rng) -> np.ndarray:
        p = self.pixel_size
        img = np.zeros((p, p), dtype=np.float32)
        cx, cy, r = p // 2, p // 2, p // 3

        if self.jitter and self.jitter_amount > 0:
            cx = int(np.clip(cx + rng.uniform(-self.jitter_amount, self.jitter_amount) * p, r, p - r - 1))
            cy = int(np.clip(cy + rng.uniform(-self.jitter_amount, self.jitter_amount) * p, r, p - r - 1))

        if cls_id == 0:   # triangle
            for row in range(p):
                half = max(1, int((row / p) * r))
                for col in range(max(0, cx - half), min(p, cx + half + 1)):
                    if row < p:
                        img[row, col] = 1.0
        elif cls_id == 1:  # circle
            for row in range(p):
                for col in range(p):
                    if (row - cy) ** 2 + (col - cx) ** 2 <= r ** 2:
                        img[row, col] = 1.0
        elif cls_id == 2:  # square
            img[max(0, cy - r):min(p, cy + r), max(0, cx - r):min(p, cx + r)] = 1.0
        elif cls_id == 3:  # X
            for i in range(-r, r + 1):
                for t in [-1, 1]:
                    row, col = cy + i, cx + t * i
                    if 0 <= row < p and 0 <= col < p:
                        img[row, col] = 1.0

        if self.noise_var > 0:
            img = np.clip(img + rng.normal(0, self.noise_var ** 0.5, (p, p)).astype(np.float32), 0, 1)
        return img

    def _to_spikes(self, images: np.ndarray, labels: np.ndarray, rng) -> tuple:
        n, n_px = images.shape
        T = n * self.num_steps
        prob = np.clip(images * (self.max_rate_hz / 1000.0) * self.gain, 0, 1)

        data = np.zeros((T, n_px), dtype=np.int8)
        spike_labels = np.zeros(T, dtype=np.int32)
        for i in range(n):
            start, end = i * self.num_steps, (i + 1) * self.num_steps
            data[start:end] = (rng.random((self.num_steps, n_px)) < prob[i]).astype(np.int8)
            spike_labels[start:end] = labels[i]
        return data, spike_labels

    # ------------------------------------------------------------------
    # FCX1 (already spike data — no Poisson encoding)
    # ------------------------------------------------------------------

    def _load_fcx1(self) -> dict:
        def _load(split: str, count: int):
            path = os.path.join(self.data_dir, "fcx1", f"X_fcx1_{split}.npz")
            d = np.load(path, allow_pickle=True)
            X = d["x"].astype(np.int8)   # already binary spikes
            y = d["y"].astype(np.int32)
            n_windows = min(count, len(X) // self.num_steps)
            end = n_windows * self.num_steps
            return X[:end], y[:end]

        total_train = self.train_count + self.val_count
        X_tr, y_tr = _load("train", total_train)
        X_te, y_te = _load("test", self.test_count)

        split = self.train_count * self.num_steps
        return dict(
            data_train=X_tr[:split], labels_train=y_tr[:split],
            data_val=X_tr[split:], labels_val=y_tr[split:],
            data_test=X_te, labels_test=y_te,
        )
