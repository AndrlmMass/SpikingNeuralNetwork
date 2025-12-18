"""
MNIST and related datasets.

This module provides utilities for loading MNIST, KMNIST, FMNIST, and NotMNIST.
"""

import numpy as np
import torch
import os
from torchvision import datasets
from PIL import Image
from skimage.draw import circle_perimeter, line
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Dataset identifiers
SUPPORTED_DATASETS = ['mnist', 'kmnist', 'fmnist', 'notmnist']

__all__ = [
    'load_mnist_family',
    'load_notmnist_deeplake',
    'SUPPORTED_DATASETS',
]

class GEOMFIG_DATASET:
    def __init__(self, pixel_size: int, train_ratio: float, val_ratio: float, test_ratio: float, noise_var: float, jitter: bool, jitter_amount: float, n_classes: int, num_samples: int, rng: np.random.Generator, which_classes: np.array, num_workers: int, tri_size: float, tri_thick: int, cir_size: float, cir_thick: int, sqr_size: float, sqr_thick: int, x_size: float, x_thick: int, clamp_min: float, clamp_max: float):
        self.pixel_size = pixel_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.noise_var = noise_var
        self.jitter = jitter
        self.jitter_amount = jitter_amount
        self.n_classes = n_classes
        self.num_samples = num_samples
        self.rng = rng
        self.which_classes = which_classes
        self.num_workers = num_workers
        self.tri_size = tri_size
        self.tri_thick = tri_thick
        self.cir_size = cir_size
        self.cir_thick = cir_thick
        self.sqr_size = sqr_size
        self.sqr_thick = sqr_thick
        self.x_size = x_size
        self.x_thick = x_thick
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def generate_normal_value(self, mean, variance):
        std_dev = np.sqrt(variance)
        value = self.rng.normal(loc=mean, scale=std_dev)
        return value

    def draw_line_high_res(high_res_grid, v0, v1, thickness, subgrid_resolution):
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        length = np.sqrt(dx**2 + dy**2)
        normal_unit = np.array([-dy / length, dx / length])

        for offset in range(-thickness // 2, thickness // 2):
            offset_vector = offset * normal_unit
            v0_offset = v0 + offset_vector / subgrid_resolution
            v1_offset = v1 + offset_vector / subgrid_resolution
            rr, cc = line(
                int(v0_offset[1] * subgrid_resolution),
                int(v0_offset[0] * subgrid_resolution),
                int(v1_offset[1] * subgrid_resolution),
                int(v1_offset[0] * subgrid_resolution),
            )
            for r, c in zip(rr, cc):
                if 0 <= r < high_res_grid.shape[0] and 0 <= c < high_res_grid.shape[1]:
                    high_res_grid[r, c] = 1


    def draw_circle_with_thickness(
        high_res_grid, center, radius, thickness, subgrid_resolution
    ):
        min_radius = int(radius - thickness // 2)
        max_radius = int(radius + thickness // 2)
        for r in range(min_radius, max_radius + 1):
            rr, cc = circle_perimeter(
                int(center * subgrid_resolution),
                int(center * subgrid_resolution),
                int(r * subgrid_resolution),
            )
            for i in range(len(rr)):
                if (
                    0 <= rr[i] < high_res_grid.shape[0]
                    and 0 <= cc[i] < high_res_grid.shape[1]
                ):
                    high_res_grid[rr[i], cc[i]] = 1


    def add_noise(self, input_space, noise_rand, noise_variance):
        if noise_rand:
            for j in range(input_space.shape[0]):
                for l in range(input_space.shape[1]):
                    mean, variance = input_space[j, l], noise_variance
                    fuzzy_val = self.rng.normal(loc=mean, scale=variance**2)
                    fuzzy_val = np.clip(fuzzy_val, 0, 1)
                    input_space[j, l] = fuzzy_val
        return input_space


    def gen_triangle(
        self, input_dims, triangle_size, triangle_thickness, noise_rand, noise_variance
    ):
        input_space = np.zeros((input_dims, input_dims))
        center = input_dims / 2
        base_length = triangle_size * input_dims
        triangle_height = (np.sqrt(3) / 2) * base_length
        top_vertex = (center, center + triangle_height / 2)
        bottom_left_vertex = (center - base_length / 2, center - triangle_height / 2)
        bottom_right_vertex = (center + base_length / 2, center - triangle_height / 2)

        subgrid_resolution = 100
        subgrid_size = subgrid_resolution**2
        high_res_triangle = np.zeros(
            (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
        )

        vertices = np.array([top_vertex, bottom_left_vertex, bottom_right_vertex])
        for i in range(3):
            self.draw_line_high_res(
                high_res_triangle,
                vertices[i],
                vertices[(i + 1) % 3],
                triangle_thickness,
                subgrid_resolution,
            )

        for i in range(input_dims):
            for j in range(input_dims):
                top = i * subgrid_resolution
                left = j * subgrid_resolution
                bottom = (i + 1) * subgrid_resolution
                right = (j + 1) * subgrid_resolution
                subgrid = high_res_triangle[top:bottom, left:right]
                coverage = np.sum(subgrid) / subgrid_size
                input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

        max_value = np.mean(input_space)
        input_space_normalized = input_space / max_value if max_value > 0 else input_space
        input_space_flipped = np.flipud(input_space_normalized)
        input_space_flipped = np.clip(input_space_flipped, a_min=0, a_max=1)

        return self.add_noise(input_space_flipped, noise_rand, noise_variance)


    def gen_square(
        self, input_dims, square_size, square_thickness, noise_rand, noise_variance
    ):
        input_space = np.zeros((input_dims, input_dims))
        square_length = int(square_size * input_dims)
        thickness = square_thickness
        start_idx = (input_dims - square_length) // 2
        end_idx = start_idx + square_length

        for i in range(thickness):
            input_space[start_idx + i, start_idx:end_idx] = 1
            input_space[end_idx - 1 - i, start_idx:end_idx] = 1

        for i in range(thickness):
            input_space[start_idx:end_idx, start_idx + i] = 1
            input_space[start_idx:end_idx, end_idx - 1 - i] = 1

        return self.add_noise(input_space, noise_rand, noise_variance)


    def gen_x(self, input_dims, x_size, x_thickness, noise_rand, noise_variance):
        if not (0 <= x_size <= 1):
            raise ValueError("X size must be between 0 and 1.")
        if x_thickness <= 0:
            raise ValueError("X thickness must be greater than 0.")

        center = input_dims / 2
        half_diagonal = input_dims * x_size / 2
        line1_start = (center - half_diagonal, center + half_diagonal)
        line1_end = (center + half_diagonal, center - half_diagonal)
        line2_start = (center + half_diagonal, center + half_diagonal)
        line2_end = (center - half_diagonal, center - half_diagonal)

        input_space = np.zeros((input_dims, input_dims))
        subgrid_resolution = 100
        subgrid_size = subgrid_resolution**2
        high_res_x = np.zeros(
            (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
        )

        self.draw_line_high_res(
            high_res_x, line1_start, line1_end, x_thickness, subgrid_resolution
        )
        self.draw_line_high_res(
            high_res_x, line2_start, line2_end, x_thickness, subgrid_resolution
        )

        for i in range(input_dims):
            for j in range(input_dims):
                top = i * subgrid_resolution
                left = j * subgrid_resolution
                bottom = (i + 1) * subgrid_resolution
                right = (j + 1) * subgrid_resolution
                subgrid = high_res_x[top:bottom, left:right]
                coverage = np.sum(subgrid) / subgrid_size
                input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

        max_value = np.mean(input_space)
        input_space_normalized = input_space / max_value if max_value > 0 else input_space
        input_space_normalized = np.clip(input_space_normalized, a_min=0, a_max=1)

        return self.add_noise(input_space_normalized, noise_rand, noise_variance)


    def gen_circle(self, input_dims, circle_size, circle_thickness, noise_rand, noise_variance):
        if not (0 <= circle_size <= 1):
            raise ValueError("Circle size must be between 0 and 1.")
        if circle_thickness <= 0:
            raise ValueError("Circle thickness must be greater than 0.")

        input_space = np.zeros((input_dims, input_dims))
        center = input_dims // 2
        radius = int(circle_size * input_dims / 2)
        subgrid_resolution = 100
        subgrid_size = subgrid_resolution**2
        high_res_circle = np.zeros(
            (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
        )

        self.draw_circle_with_thickness(
            high_res_circle, center, radius, circle_thickness, subgrid_resolution
        )

        for i in range(input_dims):
            for j in range(input_dims):
                top = i * subgrid_resolution
                left = j * subgrid_resolution
                bottom = (i + 1) * subgrid_resolution
                right = (j + 1) * subgrid_resolution
                subgrid = high_res_circle[top:bottom, left:right]
                coverage = np.sum(subgrid) / subgrid_size
                input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

        max_value = np.mean(input_space)
        input_space_normalized = input_space / max_value if max_value > 0 else input_space
        input_space_normalized = np.clip(input_space_normalized, a_min=0, a_max=1)

        return self.add_noise(input_space_normalized, noise_rand, noise_variance)


    def _geomfig_generate_one(
        self,
        cls_id: int,
        pixel_size: int,
        noise_var: float,
        jitter: bool,
        jitter_amount: float,
        tri_size: float,
        tri_thick: int,
        cir_size: float,
        cir_thick: int,
        sqr_size: float,
        sqr_thick: int,
        x_size: float,
        clamp_min: float,
        clamp_max: float,
        x_thick: int,
    ) -> np.ndarray:
        """
        Generate a single geometric figure image in [0,1] of shape (pixel_size, pixel_size).
        Class mapping: 0=triangle, 1=circle, 2=square, 3=x.
        """
        def j(val, rel, clamp_min, clamp_max):
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
            img = self.gen_triangle(
                input_dims=pixel_size,
                triangle_size=float(
                    j(tri_size, rel=True, clamp_min=clamp_min, clamp_max=clamp_max)
                ),
                triangle_thickness=int(max(1, j(tri_thick, rel=True))),
                noise_rand=True,
                noise_variance=float(max(0.0, noise_var)),
            )
        elif cls_id == 1:  # circle
            img = self.gen_circle(
                input_dims=pixel_size,
                circle_size=float(
                    j(cir_size, rel=True, clamp_min=clamp_min, clamp_max=clamp_max)
                ),
                circle_thickness=int(max(1, j(cir_thick, rel=True))),
                noise_rand=True,
                noise_variance=float(max(0.0, noise_var)),
            )
        elif cls_id == 2:  # square
            img = self.gen_square(
                input_dims=pixel_size,
                square_size=float(
                    j(sqr_size, rel=True, clamp_min=clamp_min, clamp_max=clamp_max)
                ),
                square_thickness=int(max(1, j(sqr_thick, rel=True))),
                noise_rand=True,
                noise_variance=float(max(0.0, noise_var)),
            )
        else:  # 3: x
            img = self.gen_x(
                input_dims=pixel_size,
                x_size=float(j(x_size, rel=True, clamp_min=clamp_min, clamp_max=clamp_max)),
                x_thickness=int(max(1, j(x_thick, rel=True))),
                noise_rand=True,
                noise_variance=float(max(0.0, noise_var)),
            )   
        # Ensure numeric array in [0,1]
        img = np.asarray(img, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img


    def create_geomfig_data(
        self,
        pixel_size: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        noise_var: float,
        jitter: bool,
        jitter_amount: float,
        n_classes: int,
        num_samples: int,
        rng: np.random.Generator,
        which_classes: np.array,
        num_workers: int,
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
        """
        Generate geometric figures dataset as images.
        Returns torch tensors of shape (N, 1, H, W) and labels.
        """

        def make_split(total_samples: int, split_name: str):
            if which_classes.shape[0] != n_classes:
                raise ValueError(f"Number of classes in which_classes ({which_classes.shape[0]}) does not match n_classes ({n_classes})")
            # Round down to optimal total samples for even split
            total = max(0, (int(total_samples) // n_classes) * n_classes)
            if total == 0:
                raise ValueError(
                    f"Number of classes are too low, or number of samples for {split_name}"
                )
            imgs = np.zeros((total, pixel_size, pixel_size), dtype=np.float32)
            labels = np.array([which_classes[i % len(which_classes)] for i in range(total)], dtype=np.int32)

            # Enable progress bars for generation
            os.environ["TQDM_DISABLE"] = "False"

            # set num workers
            use_workers = 1
            if num_workers is not None:
                use_workers = max(1, int(num_workers))

            # Predefine balanced class sequence (even repeats of 0..n_classes-1)
            # `total` was already rounded to be divisible by `n_classes` above.
            cls_seq = np.tile(which_classes, total // len(which_classes))
            # Deterministically shuffle class order using provided RNG
            cls_seq = rng.permutation(cls_seq)

            if use_workers == 1:
                for i in tqdm(
                    range(total),
                    total=total,
                    desc=f"Generating geomfig {split_name}",
                    leave=False,
                ):
                    cls_id = int(cls_seq[i])
                    imgs[i] = self._geomfig_generate_one(
                        cls_id=cls_id,
                        pixel_size=pixel_size,
                        noise_var=noise_var,
                        jitter=jitter,
                        jitter_amount=jitter_amount,
                        tri_size=tri_size,
                        tri_thick=tri_thick,
                        cir_size=cir_size,
                        cir_thick=cir_thick,
                        sqr_size=sqr_size,
                        sqr_thick=sqr_thick,
                        x_size=x_size,
                        clamp_min=clamp_min,
                        clamp_max=clamp_max,
                        x_thick=x_thick,
                    )
            else:
                # Pre-draw independent seeds for each sample to avoid identical substreams across forks
                seeds = rng.integers(0, 2**31 - 1, size=total, dtype=np.int64)
                with ProcessPoolExecutor(max_workers=use_workers) as ex:
                    futures = {}
                    for i in range(total):
                        cls_id = int(cls_seq[i])
                        futures[
                            ex.submit(
                                self._geomfig_generate_one,
                                    cls_id=cls_id,
                                    pixel_size=pixel_size,
                                    noise_var=noise_var,
                                    jitter=jitter,
                                    jitter_amount=jitter_amount,
                                    tri_size=tri_size,
                                    tri_thick=tri_thick,
                                    cir_size=cir_size,
                                    cir_thick=cir_thick,
                                    sqr_size=sqr_size,
                                    sqr_thick=sqr_thick,
                                    x_size=x_size,
                                    clamp_min=clamp_min,
                                    clamp_max=clamp_max,
                                    x_thick=x_thick,
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
                            cls_id = int(cls_seq[i])
                            imgs[i] = self._geomfig_generate_one(
                                cls_id=cls_id,
                                pixel_size=pixel_size,
                                noise_var=noise_var,
                                jitter=jitter,
                                jitter_amount=jitter_amount,
                                tri_size=tri_size,
                                tri_thick=tri_thick,
                                cir_size=cir_size,
                                cir_thick=cir_thick,
                                sqr_size=sqr_size,
                                sqr_thick=sqr_thick,
                                x_size=x_size,
                                clamp_min=clamp_min,
                                clamp_max=clamp_max,
                                x_thick=x_thick,
                            ) 

            # Return images (not spikes) - spike conversion handled by ImageDataStreamer
            # Convert to torch tensors with channel dimension: (N, H, W) -> (N, 1, H, W)
            imgs_torch = torch.from_numpy(imgs).unsqueeze(1)  # Add channel dimension
            
            return imgs_torch, labels
        total_samples = num_samples
        data_train, labels_train = make_split(int(train_ratio*total_samples), "train")
        data_val, labels_val = make_split(int(val_ratio*total_samples), "val")
        data_test, labels_test = make_split(int(test_ratio*total_samples), "test")

        return (
            data_train,
            labels_train,
            data_val,
            labels_val,
            data_test,
            labels_test,
        )

class MNIST_DATASET:
    def __init__(self, transform, train_ratio, val_ratio, rng, which_classes=None, load_cache_fn=None, save_cache_fn=None):
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.rng = rng
        self.which_classes = which_classes
        self.load_cache_fn = load_cache_fn
        self.save_cache_fn = save_cache_fn

    def load_mnist_family(
        self,
        dataset: str,
        transform,
        train_ratio: float,
        val_ratio: float,
        rng,
        which_classes=None,
        load_cache_fn=None,
        save_cache_fn=None,
    ):
        """
        Load MNIST-family datasets (MNIST, KMNIST, FMNIST) with caching support.
        
        Args:
            dataset: Dataset name ('mnist', 'kmnist', 'fmnist')
            transform: Image transform to apply
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            rng: Random number generator
            which_classes: Optional list of class indices to filter
            load_cache_fn: Optional function to load from cache (partition) -> (images, labels) or (None, None)
            save_cache_fn: Optional function to save to cache (partition, images, labels)
        
        Returns:
            (train_images, train_labels, val_images, val_labels, test_images, test_labels)
        """
        ds_map = {
            "mnist": datasets.MNIST,
            "kmnist": datasets.KMNIST,
            "fmnist": datasets.FashionMNIST,
        }
        
        if dataset not in ds_map:
            raise ValueError(f"Unsupported image dataset: {dataset}")
        ds_cls = ds_map[dataset]

        # Use a stable on-disk cache for torchvision datasets to avoid re-downloads
        torch_root = os.path.join("data", "torchvision")
        os.makedirs(torch_root, exist_ok=True)

        mnist_train = ds_cls(
            root=torch_root, train=True, transform=transform, download=True
        )
        mnist_test = ds_cls(
            root=torch_root, train=False, transform=transform, download=True
        )

        # Respect which_classes by filtering datasets if provided
        if which_classes:
            target_classes = set(which_classes)

            def _filter_dataset(ds):
                indices = [i for i in range(len(ds)) if int(ds[i][1]) in target_classes]
                return torch.utils.data.Subset(ds, indices)

            mnist_train = _filter_dataset(mnist_train)
            mnist_test = _filter_dataset(mnist_test)

        # subset validation data from train set by percentage
        mnist_train, mnist_val = torch.utils.data.random_split(
            mnist_train, [train_ratio, val_ratio], generator=rng
        )

        # Try to load from image cache first
        if load_cache_fn:
            train_images, train_labels = load_cache_fn("train")
            val_images, val_labels = load_cache_fn("val")
            test_images, test_labels = load_cache_fn("test")
            
            if train_images is not None and val_images is not None and test_images is not None:
                print(f"Using cached {dataset.upper()} images")
                return train_images, train_labels, val_images, val_labels, test_images, test_labels

        # Load all splits into RAM
        def _load_split(dataset):
            """Helper to load a dataset split into RAM."""
            images, labels = zip(*[dataset[i] for i in range(len(dataset))])
            return torch.stack(images), np.array([int(lbl) for lbl in labels], dtype=np.int64)

        train_images, train_labels = _load_split(mnist_train)
        val_images, val_labels = _load_split(mnist_val)
        test_images, test_labels = _load_split(mnist_test)
        
        # Save to cache if function provided
        if save_cache_fn:
            save_cache_fn("train", train_images, train_labels)
            save_cache_fn("val", val_images, val_labels)
            save_cache_fn("test", test_images, test_labels)

        return train_images, train_labels, val_images, val_labels, test_images, test_labels


    def load_notmnist_deeplake(transform, train_ratio, val_ratio, test_ratio, rng, which_classes=None):
        """
        Load NotMNIST from Deeplake and return torch tensors for train/val/test splits.
        
        Args:
            transform: Image transform to apply
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            rng: Random number generator
            which_classes: Optional list of class indices to filter
        
        Returns:
            (train_images, train_labels, val_images, val_labels, test_images, test_labels)
        """
        import deeplake
        ds = deeplake.load("hub://activeloop/not-mnist-small", read_only=True)

        dataset = []
        use_classes = set(which_classes) if which_classes else None

        for sample in ds:
            try:
                img_np = sample["images"].numpy()
                lbl = sample["labels"].numpy()
            except KeyError as exc:
                raise KeyError(
                    "NotMNIST sample missing expected 'images' or 'labels' tensors."
                ) from exc
            label_int = int(np.asarray(lbl).flatten()[0])

            # Filter by desired classes if provided
            if use_classes is not None and label_int not in use_classes:
                continue

            img_np = np.asarray(img_np)
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np[0]

            img_pil = Image.fromarray(img_np.astype(np.uint8), mode="L")
            img_tensor = transform(img_pil)
            dataset.append((img_tensor, label_int))
        
        n = len(dataset)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        n_test = int(test_ratio * n)
        
        # Create tensors from dataset
        images = torch.stack([img for img, _ in dataset])
        labels = np.array([lbl for _, lbl in dataset], dtype=np.int64)
        
        # Split into train/val/test
        dataset_tensor = torch.utils.data.TensorDataset(images, torch.from_numpy(labels))
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset_tensor,
            [n_train, n_val, n_test],
            generator=rng
        )
        
        # Extract data from splits
        train_indices = train_ds.indices
        val_indices = val_ds.indices
        test_indices = test_ds.indices
        
        train_images = images[train_indices]
        train_labels = labels[train_indices]
        val_images = images[val_indices]
        val_labels = labels[val_indices]
        test_images = images[test_indices]
        test_labels = labels[test_indices]
        
        return train_images, train_labels, val_images, val_labels, test_images, test_labels

