import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from snntorch import spikeplot as splt
from snntorch import spikegen
import snntorch as snn
import itertools
from tqdm import tqdm
import argparse
import os
import sys
import json
import math
from datetime import datetime
from PIL import Image
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute path to this script's directory (anchor outputs here by default)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

lif1 = snn.Leaky(beta=0.9)
lif2 = snn.Leaky(beta=0.9)

# dataloader arguments
batch_size = 128
data_path = "./data/mnist"

# define a transform
dtype = torch.float

# define a transform to preprocess the data
transform = transforms.Compose(
    [
        transforms.Resize((15, 15)),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

# Data loaders and dataset are created after CLI parsing
train_loader = None
test_loader = None


def _load_notmnist_deeplake(transform):
    import deeplake

    ds = deeplake.load("hub://activeloop/not-mnist-small", read_only=True)

    if "images" in ds.tensors:
        image_tensor = ds["images"]
    elif "image" in ds.tensors:
        image_tensor = ds["image"]
    else:
        raise KeyError(
            "Neither 'images' nor 'image' tensor found in the NOTMNIST DeepLake dataset."
        )

    if "labels" in ds.tensors:
        label_tensor_ds = ds["labels"]
    elif "label" in ds.tensors:
        label_tensor_ds = ds["label"]
    else:
        raise KeyError(
            "Neither 'labels' nor 'label' tensor found in the NOTMNIST DeepLake dataset."
        )

    raw_images = image_tensor.numpy()
    raw_labels = np.array(label_tensor_ds.numpy()).astype(np.int64).reshape(-1)

    processed_images = []
    for img in raw_images:
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        if img_np.ndim == 2:
            pil_img = Image.fromarray(img_np.astype(np.uint8), mode="L")
        else:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
        if transform is not None:
            img_tensor = transform(pil_img)
        else:
            img_tensor = torch.from_numpy(
                np.array(pil_img, dtype=np.float32) / 255.0
            ).unsqueeze(0)
        processed_images.append(img_tensor)

    data_tensor = torch.stack(processed_images)
    label_tensor = torch.from_numpy(raw_labels).long()

    dataset = TensorDataset(data_tensor, label_tensor)
    dataset.classes = list("ABCDEFGHIJ")
    dataset.dataset_name = "NOTMNIST"
    return dataset


def preview_dataset_samples(dataset, num_samples: int, dataset_name: str) -> None:
    """Plot a random subset of samples from the provided dataset."""
    if num_samples <= 0:
        return

    capped_num_samples = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:capped_num_samples]

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[int(idx)]
        images.append(img)
        if isinstance(label, torch.Tensor):
            labels.append(int(label.item()))
        elif isinstance(label, np.ndarray):
            labels.append(int(label.item()))
        else:
            labels.append(int(label))

    num_cols = min(5, capped_num_samples)
    num_rows = math.ceil(capped_num_samples / num_cols)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5)
    )
    axes = np.atleast_2d(axes)

    class_names = getattr(dataset, "classes", None)
    if class_names is None and hasattr(dataset, "dataset"):
        class_names = getattr(dataset.dataset, "classes", None)

    for plot_idx, ax in enumerate(axes.flat):
        if plot_idx < capped_num_samples:
            current_img = images[plot_idx]
            if isinstance(current_img, torch.Tensor):
                img = current_img.squeeze().cpu().numpy()
            else:
                img = np.array(current_img)
            if img.ndim == 2:
                # Normalize to [0, 1] for display without altering source tensors
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))

            if class_names and labels[plot_idx] < len(class_names):
                ax.set_title(class_names[labels[plot_idx]])
            else:
                ax.set_title(f"Label: {int(labels[plot_idx])}")
        ax.axis("off")

    fig.suptitle(f"{dataset_name} Preview ({capped_num_samples} samples)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Network architecture (num_inputs inferred after dataset init)
num_inputs = None
num_hidden = 1000
num_outputs = 10

# Temporal dynamics
num_steps = 100
beta = 0.95

# ---------------------- Sleep / STDP Regularization Config ----------------------
# STDP hyperparameters for sleep phase
tau_pre = 20.0
tau_post = 20.0
A_plus = 0.01
A_minus = 0.012
stdp_lr1 = 1e-3
stdp_lr2 = 1e-3

# Sleep activation thresholds relative to a baseline total |W|
sleep_trigger_ratio = 1.1
sleep_restart_ratio = 1.0

# Noisy membrane potential and random input during sleep
sleep_mem_noise_std = 3
sleep_input_rate = 0  # Bernoulli firing rate for random spikes

# Power-law mapping toward target weight magnitude
sleep_target_weight = 0.2
sleep_lambda = 0.999

# Guard to avoid excessively long sleep cycles in one activation
sleep_max_iters = 500


def compute_total_abs_weight(model: nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += module.weight.data.abs().sum().item()
    return total


def apply_powerlaw_weight_mapping_inplace(
    param: torch.Tensor, target_mag: float, lam: float
) -> None:
    # w_new = w_target * (|w_old|/w_target)^lambda with sign preserved
    with torch.no_grad():
        eps = 1e-12
        w = param.data
        sign = torch.sign(w)
        abs_w = w.abs()
        ratio = abs_w / max(target_mag, eps)
        mapped = target_mag * torch.pow(torch.clamp(ratio, min=0.0), lam)
        param.data = sign * mapped


def run_sleep_phase(
    model: nn.Module,
    device: torch.device,
    num_steps: int,
    batch_size_sleep: int,
    input_rate: float,
    mem_noise_std: float,
    target_mag: float,
    lam: float,
    stdp_lr1: float,
    stdp_lr2: float,
    tau_pre: float,
    tau_post: float,
    max_iters: int,
) -> int:
    """Run a sleep regularization cycle with random spikes, membrane noise,
    and pair-based STDP updates. Returns the number of sleep iterations performed.
    """
    # Precompute decays
    pre_decay = torch.exp(torch.tensor(-1.0 / tau_pre, device=device))
    post_decay = torch.exp(torch.tensor(-1.0 / tau_post, device=device))

    I = model.fc1.in_features
    H = model.fc1.out_features
    O = model.fc2.out_features

    sleep_iters = 0
    with torch.no_grad():
        while sleep_iters < max_iters:
            # Initialize traces and weight deltas
            pre_trace_in = torch.zeros((batch_size_sleep, I), device=device)
            post_trace_h = torch.zeros((batch_size_sleep, H), device=device)
            pre_trace_h = torch.zeros((batch_size_sleep, H), device=device)
            post_trace_o = torch.zeros((batch_size_sleep, O), device=device)

            dW1 = torch.zeros((I, H), device=device)
            dW2 = torch.zeros((H, O), device=device)

            # Initialize LIF states
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()

            for t in range(num_steps):
                # No external input during sleep
                pre_spk_in = torch.zeros((batch_size_sleep, I), device=device)

                # Inject Gaussian noise into membrane potentials
                if isinstance(mem1, torch.Tensor):
                    mem1 = mem1 + mem_noise_std * torch.randn_like(mem1)
                if isinstance(mem2, torch.Tensor):
                    mem2 = mem2 + mem_noise_std * torch.randn_like(mem2)

                # Forward one step
                cur1 = model.fc1(pre_spk_in)
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.fc2(spk1)
                spk2, mem2 = model.lif2(cur2, mem2)

                # Decay traces
                pre_trace_in = pre_trace_in * pre_decay
                post_trace_h = post_trace_h * post_decay
                pre_trace_h = pre_trace_h * pre_decay
                post_trace_o = post_trace_o * post_decay

                # Accumulate traces
                pre_trace_in = pre_trace_in + pre_spk_in
                post_trace_h = post_trace_h + spk1
                pre_trace_h = pre_trace_h + spk1
                post_trace_o = post_trace_o + spk2

                # STDP updates
                pot1 = torch.einsum("bi,bj->ij", pre_trace_in, spk1) / batch_size_sleep
                dep1 = (
                    torch.einsum("bi,bj->ij", pre_spk_in, post_trace_h)
                    / batch_size_sleep
                )
                dW1 += A_plus * pot1 - A_minus * dep1

                pot2 = torch.einsum("bi,bj->ij", pre_trace_h, spk2) / batch_size_sleep
                dep2 = torch.einsum("bi,bj->ij", spk1, post_trace_o) / batch_size_sleep
                dW2 += A_plus * pot2 - A_minus * dep2

            # Apply STDP updates
            model.fc1.weight.data += stdp_lr1 * dW1.t()
            model.fc2.weight.data += stdp_lr2 * dW2.t()

            # Power-law mapping toward target magnitude
            apply_powerlaw_weight_mapping_inplace(model.fc1.weight, target_mag, lam)
            apply_powerlaw_weight_mapping_inplace(model.fc2.weight, target_mag, lam)

            sleep_iters += 1

            # Stop early if below the external caller's restart threshold; caller will check
            # We break only on caller's signal to keep control; so continue here

    return sleep_iters


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


# Helper to create a fresh network on the configured device
def create_model():
    return Net().to(device)


# pass data into the network, sum the spikes over time
# and compare the neuron witht he highest number of spikes
# with the target


def calculate_accuracy(model, data, targets):
    """Calculate accuracy for a batch"""
    B = data.size(0)
    output, _ = model(data.view(B, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


def evaluate_model(model, data_loader, loss_fn, device, num_steps, batch_size):
    """Evaluate model on a dataset and return average loss and accuracy"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            B = data.size(0)
            spk_rec, mem_rec = model(data.view(B, -1))

            # Calculate loss
            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(spk_rec[step], targets)
            total_loss += loss_val.item()

            # Calculate accuracy
            _, idx = spk_rec.sum(dim=0).max(1)
            total_correct += (targets == idx).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


loss = nn.CrossEntropyLoss()


# ---------------------- Checkpointing Utilities ----------------------
def save_checkpoint(model: nn.Module, path: str, meta=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, map_location=None):
    chk = torch.load(path, map_location=map_location or device)
    model.load_state_dict(chk["model_state"])
    return chk.get("meta", {})


# ---------------------- Argparse: eval-only / load ----------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--eval-only",
    action="store_true",
    help="Load checkpoint and evaluate on test set, then exit",
)
parser.add_argument("--load", type=str, default="", help="Path to checkpoint to load")
parser.add_argument(
    "--sleep-trigger-ratio",
    type=float,
    default=None,
    help="Override sleep trigger ratio (default=1.1)",
)
parser.add_argument(
    "--sleep-restart-ratio",
    type=float,
    default=None,
    help="Override sleep restart ratio (default=0.9)",
)
parser.add_argument(
    "--sleep-log",
    action="store_true",
    help="Print sleep activation/deactivation events",
)
parser.add_argument(
    "--optimizer-mode",
    type=str,
    default="both",
    choices=["adam", "sleep", "both", "none"],
    help="adam (Adam only), sleep (STDP-only), both (Adam+Sleep), none (no updates)",
)
parser.add_argument(
    "--sleep-interval-pct",
    type=float,
    nargs="+",
    default=[0.1],
    help="One or more fractions for sleep frequency (0.0-1.0). Example: 0.05 0.1 0.2",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    choices=["MNIST", "KMNIST", "FMNIST", "NOTMNIST"],
    help="Dataset to use",
)
parser.add_argument(
    "--early-stopping", action="store_true", help="Enable early stopping"
)
parser.add_argument(
    "--patience",
    type=float,
    default=None,
    help="Early stopping patience as a fraction of total epochs (0.0-1.0, default: 0.1)",
)
parser.add_argument(
    "--min-delta",
    type=float,
    default=0.001,
    help="Minimum change to qualify as an improvement for early stopping (default: 0.001)",
)
parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Repeat full training this many times (default: 1)",
)
parser.add_argument(
    "--results-json",
    type=str,
    default=os.path.join(SCRIPT_DIR, "results", "sleep_results.json"),
    help="Path to JSON file where results are appended per run",
)
parser.add_argument(
    "--preview-samples",
    type=int,
    default=0,
    help="Number of training samples to preview before training starts (0 to disable)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: None)",
)
parser.add_argument(
    "--track-excel",
    action="store_true",
    default=False,
    help="track results in GLM/Results_.xlsx file after each run",
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Disable plotting of training/validation curves",
)
parser.add_argument(
    "--balanced",
    action="store_true",
    help="Use balanced per-class subsampling when limiting dataset sizes",
)
parser.add_argument(
    "--train-size",
    type=int,
    default=None,
    help="If set, limit the training set to this many samples (used with --balanced)",
)
parser.add_argument(
    "--test-size",
    type=int,
    default=None,
    help="If set, limit the test set to this many samples (used with --balanced)",
)
parser.add_argument("--help", action="help", help="Show help and exit")
cli_args, _ = parser.parse_known_args()


# ---------------------- Seeding ----------------------
def set_global_seed(seed: int) -> None:
    if seed is None:
        return
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Favor determinism where possible
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


set_global_seed(cli_args.seed)


# ---------------------- Balanced Subsample Utilities ----------------------
def _get_labels_array(ds) -> np.ndarray:
    """Best-effort extraction of labels from a dataset or subset without iterating all samples."""
    # Subset wrapper
    if isinstance(ds, torch.utils.data.Subset):
        base_labels = _get_labels_array(ds.dataset)
        idx = np.array(ds.indices, dtype=int)
        return base_labels[idx]
    # torchvision datasets expose targets/labels
    for attr in ["targets", "labels"]:
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()
            return np.array(arr)
    # TensorDataset: assume (data, labels)
    if hasattr(ds, "tensors") and len(ds.tensors) >= 2:
        labels = ds.tensors[1]
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy()
        return np.array(labels)
    # Fallback: iterate (slower)
    lbls = []
    for i in range(len(ds)):
        _, y = ds[i]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        else:
            y = int(y)
        lbls.append(y)
    return np.array(lbls, dtype=int)


def _balanced_indices(labels: np.ndarray, total: int, seed: int) -> np.ndarray:
    """Return indices for a balanced subset across classes totaling 'total' samples."""
    rng = np.random.RandomState(seed if seed is not None else 0)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    num_classes = len(classes)
    if total is None or total <= 0:
        return np.arange(labels.shape[0], dtype=int)
    per_class_base = total // num_classes
    remainder = total % num_classes
    # Build pool per class
    class_to_indices = {c: np.where(labels == c)[0] for c in classes}
    # Shuffle within class
    for c in classes:
        rng.shuffle(class_to_indices[c])
    # First pass: take base count per class (limited by availability)
    selected = []
    for c in classes:
        take = min(per_class_base, len(class_to_indices[c]))
        selected.append(class_to_indices[c][:take])
        class_to_indices[c] = class_to_indices[c][take:]
    selected = [
        idx
        for arr in selected
        for idx in (arr if isinstance(arr, np.ndarray) else np.array([], dtype=int))
    ]
    # Distribute remainder from classes with leftover
    remaining_needed = total - len(selected)
    if remaining_needed > 0:
        # flatten remaining pools
        for c in classes:
            if remaining_needed <= 0:
                break
            pool = class_to_indices[c]
            if len(pool) == 0:
                continue
            add = min(
                remaining_needed, len(pool), 1 if remainder > 0 else 0
            )  # one per class until remainder exhausted
            if add > 0:
                selected.extend(pool[:add].tolist())
                class_to_indices[c] = pool[add:]
                remainder -= add
                remaining_needed -= add
    # If still short, fill from any remaining pools
    if remaining_needed > 0:
        pools = (
            np.concatenate(
                [class_to_indices[c] for c in classes if len(class_to_indices[c]) > 0]
            )
            if any(len(class_to_indices[c]) > 0 for c in classes)
            else np.array([], dtype=int)
        )
        if len(pools) > 0:
            selected.extend(pools[:remaining_needed].tolist())
    return np.array(selected, dtype=int)


def make_balanced_subset(ds, total: int, seed: int):
    """Create a balanced torch.utils.data.Subset of size up to 'total'."""
    if total is None:
        return ds
    labels = _get_labels_array(ds)
    if total > len(labels):
        total = len(labels)
    idx = _balanced_indices(labels, total, seed)
    return torch.utils.data.Subset(ds, idx.tolist())


def balanced_split_from_full(full_ds, train_total: int, test_total: int, seed: int):
    """Split a single full dataset into balanced train/test subsets without overlap."""
    labels = _get_labels_array(full_ds)
    rng = np.random.RandomState(seed if seed is not None else 0)
    classes = np.unique(labels)
    class_to_indices = {c: rng.permutation(np.where(labels == c)[0]) for c in classes}
    num_classes = len(classes)
    train_pc = train_total // num_classes if train_total else 0
    test_pc = test_total // num_classes if test_total else 0
    train_sel = []
    test_sel = []
    for c in classes:
        pool = class_to_indices[c]
        t_take = min(train_pc, len(pool))
        train_sel.append(pool[:t_take])
        pool = pool[t_take:]
        v_take = min(test_pc, len(pool))
        test_sel.append(pool[:v_take])
        class_to_indices[c] = pool[v_take:]
    train_sel = np.concatenate(train_sel) if train_sel else np.array([], dtype=int)
    test_sel = np.concatenate(test_sel) if test_sel else np.array([], dtype=int)
    return torch.utils.data.Subset(
        full_ds, train_sel.tolist()
    ), torch.utils.data.Subset(full_ds, test_sel.tolist())


# ---------------------- Dataset Selection & Loaders ----------------------
dataset_name = cli_args.dataset.upper()
folder_map = {
    "MNIST": "mnist",
    "KMNIST": "kmnist",
    "FMNIST": "fmnist",
    "NOTMNIST": "notmnist",
}
data_path = f"./data/{folder_map.get(dataset_name, 'mnist')}"

if dataset_name == "NOTMNIST":
    full_dataset = _load_notmnist_deeplake(transform)
    if cli_args.balanced and (
        cli_args.train_size is not None or cli_args.test_size is not None
    ):
        train_total = (
            cli_args.train_size
            if cli_args.train_size is not None
            else len(full_dataset)
        )
        test_total = cli_args.test_size if cli_args.test_size is not None else 0
        train_dataset, test_dataset = balanced_split_from_full(
            full_dataset,
            train_total,
            test_total,
            seed=cli_args.seed if cli_args.seed is not None else 0,
        )
        print(
            f"Balanced NOTMNIST split: train={len(train_dataset)}, test={len(test_dataset)}"
        )
    else:
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(
                cli_args.seed if cli_args.seed is not None else 42
            ),
        )
    dataset_source = "hub://activeloop/not-mnist-small"
else:
    if dataset_name == "MNIST":
        TrainDS = datasets.MNIST
    elif dataset_name == "KMNIST":
        TrainDS = datasets.KMNIST
    elif dataset_name == "FMNIST":
        TrainDS = datasets.FashionMNIST
    else:
        TrainDS = datasets.MNIST

    train_dataset = TrainDS(data_path, train=True, download=True, transform=transform)
    test_dataset = TrainDS(data_path, train=False, download=True, transform=transform)
    if cli_args.balanced and (
        cli_args.train_size is not None or cli_args.test_size is not None
    ):
        if cli_args.train_size is not None:
            train_dataset = make_balanced_subset(
                train_dataset,
                cli_args.train_size,
                seed=cli_args.seed if cli_args.seed is not None else 0,
            )
            print(f"Balanced train subset: {len(train_dataset)} samples")
        if cli_args.test_size is not None:
            test_dataset = make_balanced_subset(
                test_dataset,
                cli_args.test_size,
                seed=cli_args.seed if cli_args.seed is not None else 0,
            )
            print(f"Balanced test subset: {len(test_dataset)} samples")
    dataset_source = os.path.abspath(data_path)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)

resolved_dataset = (
    train_dataset.dataset
    if isinstance(train_dataset, torch.utils.data.Subset)
    else train_dataset
)
print(
    f"Dataset resolved to {type(resolved_dataset).__name__} (source={dataset_source})"
)

if cli_args.preview_samples > 0:
    print(
        f"Previewing {min(cli_args.preview_samples, len(train_dataset))} samples from the {dataset_name} training set..."
    )
    preview_dataset_samples(
        train_dataset, cli_args.preview_samples, dataset_name=dataset_name
    )

# Infer input size from transformed sample
_sample_x, _ = train_dataset[0]
num_inputs = int(np.prod(_sample_x.shape))

if cli_args.eval_only:
    # Load checkpoint
    ckpt_path = cli_args.load
    if ckpt_path == "":
        # try default locations
        if os.path.exists("checkpoints/best.pt"):
            ckpt_path = "checkpoints/best.pt"
        elif os.path.exists("checkpoints/last.pt"):
            ckpt_path = "checkpoints/last.pt"
        else:
            print("No checkpoint found at default locations. Provide --load <path>.")
            sys.exit(1)

    model_for_eval = create_model()
    meta = load_checkpoint(model_for_eval, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path} | meta: {meta}")
    # Evaluate on test set
    final_test_loss, final_test_acc = evaluate_model(
        model_for_eval, test_loader, loss, device, num_steps, batch_size
    )
    print(f"Test Loss: {final_test_loss:.4f} | Test Acc: {final_test_acc*100:.2f}%")
    sys.exit(0)

# Apply CLI overrides for sleep thresholds
if cli_args.sleep_trigger_ratio is not None:
    sleep_trigger_ratio = cli_args.sleep_trigger_ratio
if cli_args.sleep_restart_ratio is not None:
    sleep_restart_ratio = cli_args.sleep_restart_ratio


def train_once(
    run_index: int, cli_args, sleep_interval_pct: float, enable_plot: bool
) -> float:
    # Fresh model and optimizer per run
    net = create_model()

    # Establish baseline |W| sum from randomly initialized network
    baseline_weight_sum = compute_total_abs_weight(net)
    sleep_active = False
    sleep_iters_run = 0

    # Determine training mode: adam / sleep / both / none
    use_adam = cli_args.optimizer_mode in ["adam", "both"]
    use_sleep = cli_args.optimizer_mode in ["sleep", "both"]

    # Calculate sleep interval (run sleep every N iterations)
    if use_sleep and sleep_interval_pct > 0:
        sleep_every_n = max(1, int(1.0 / sleep_interval_pct))
    else:
        sleep_every_n = float("inf")  # Never run sleep

    # Configure Adam optimizer if enabled
    if use_adam:
        optimizer = torch.optim.Adam(
            list(net.fc1.parameters()) + list(net.fc2.parameters()),
            lr=5e-4,
            betas=(0.9, 0.999),
        )
    else:
        optimizer = None

    num_epochs = 10
    loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    sleep_acc_deltas = (
        []
    )  # Track per-sleep-iteration change in train acc (post-Adam pre-sleep vs post-sleep)

    best_val_acc = 0.0
    global_iter_counter = 0  # Track iterations across all epochs

    # Early stopping configuration
    early_stopping_enabled = cli_args.early_stopping
    if early_stopping_enabled:
        # Patience as a fraction of total epochs
        if cli_args.patience is None:
            patience_frac = 0.1
        else:
            patience_frac = max(0.0, min(1.0, float(cli_args.patience)))
        patience = max(1, int(patience_frac * num_epochs))
        min_delta = cli_args.min_delta
        patience_counter = 0
    else:
        patience = 0
        min_delta = 0
        patience_counter = 0

    # Print training configuration
    print("\n" + "=" * 60)
    print(f"TRAINING CONFIGURATION (Run {run_index})")
    print("=" * 60)
    print(f"Optimizer Mode: {cli_args.optimizer_mode.upper()}")
    if use_adam:
        print(f"  - Adam optimizer: ENABLED (lr=5e-4)")
    else:
        print(f"  - Adam optimizer: DISABLED")
    if use_sleep:
        print(f"  - Sleep STDP: ENABLED")
        print(
            f"    * Sleep interval: every {sleep_every_n} iterations ({sleep_interval_pct*100:.1f}%)"
        )
        print(f"    * STDP learning rates: fc1={stdp_lr1}, fc2={stdp_lr2}")
        print(f"    * Membrane noise std: {sleep_mem_noise_std}")
        print(f"    * Target weight magnitude: {sleep_target_weight}")
    else:
        print(f"  - Sleep STDP: DISABLED")
    if not (use_adam or use_sleep):
        print(f"  - No updates: model weights remain fixed")
    print(f"Epochs: {num_epochs}")
    if early_stopping_enabled:
        print(
            f"Early Stopping: ENABLED (patience={patience} epochs ~ {patience_frac*100:.1f}%, min_delta={min_delta})"
        )
    else:
        print(f"Early Stopping: DISABLED")
    print(f"Batch size: {batch_size}")
    print(f"Time steps: {num_steps}")
    print(f"Dataset: {cli_args.dataset}")
    print("=" * 60 + "\n")

    # Outer training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_batch = iter(train_loader)

        # Create progress bar for training iterations
        pbar = tqdm(train_batch, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        # Minibatch training loop
        for data, targets in pbar:
            data = data.to(device)
            targets = targets.to(device)

            # Determine if this iteration should use sleep
            is_sleep_iter = (
                use_sleep
                and (sleep_interval_pct > 0)
                and (global_iter_counter % sleep_every_n == 0)
            )

            # forward pass
            net.train()
            B = data.size(0)
            spk_rec, mem_rec = net(data.view(B, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(spk_rec[step], targets)

            # Adam update always runs if enabled (even on sleep iterations)
            if use_adam:
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Calculate training accuracy (post-Adam update, pre-sleep)
            train_acc = calculate_accuracy(net, data, targets)
            train_acc_hist.append(train_acc)
            acc_before_sleep = train_acc if is_sleep_iter else None

            # Run sleep phase if this is a sleep iteration
            if is_sleep_iter:
                sleep_iters = run_sleep_phase(
                    model=net,
                    device=device,
                    num_steps=num_steps,
                    batch_size_sleep=batch_size,
                    input_rate=sleep_input_rate,
                    mem_noise_std=sleep_mem_noise_std,
                    target_mag=sleep_target_weight,
                    lam=sleep_lambda,
                    stdp_lr1=stdp_lr1,
                    stdp_lr2=stdp_lr2,
                    tau_pre=tau_pre,
                    tau_post=tau_post,
                    max_iters=1,
                )
                sleep_iters_run += sleep_iters
                if cli_args.sleep_log:
                    tqdm.write(
                        f"[sleep] ran at iteration {global_iter_counter} (epoch {epoch+1})"
                    )

                # Measure training accuracy again after sleep on the same batch
                acc_after_sleep = calculate_accuracy(net, data, targets)
                delta_acc = float(
                    acc_after_sleep
                    - (
                        acc_before_sleep
                        if acc_before_sleep is not None
                        else acc_after_sleep
                    )
                )
                sleep_acc_deltas.append(delta_acc)

            # Update progress bar with all available metrics
            metrics = {
                "Train Loss": f"{loss_val.item():.3f}",
                "Train Acc": f"{train_acc*100:.1f}%",
            }

            # Add validation metrics if available
            if test_loss_hist:
                metrics["Val Loss"] = f"{test_loss_hist[-1]:.3f}"
            if val_acc_hist:
                metrics["Val Acc"] = f"{val_acc_hist[-1]*100:.1f}%"

            # Show mode
            if use_adam and use_sleep:
                metrics["Mode"] = "Both"
            elif use_adam:
                metrics["Mode"] = "Adam"
            elif use_sleep:
                metrics["Mode"] = "Sleep"
            else:
                metrics["Mode"] = "None"

            # Show current optimizer being used
            if is_sleep_iter:
                metrics["Current"] = "Adam+Sleep" if use_adam else "Sleep"
                if sleep_acc_deltas:
                    metrics["ΔAcc@Sleep"] = f"{sleep_acc_deltas[-1]*100:+.2f}%"
            else:
                metrics["Current"] = (
                    "Adam" if use_adam else ("Sleep" if use_sleep else "None")
                )

            pbar.set_postfix(metrics)

            global_iter_counter += 1

        # Validation after each epoch
        val_loss, val_acc = evaluate_model(
            net, test_loader, loss, device, num_steps, batch_size
        )
        test_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Early stopping logic
        if early_stopping_enabled:
            if val_acc > best_val_acc + min_delta:
                # Significant improvement found
                best_val_acc = float(val_acc)
                patience_counter = 0
                save_checkpoint(
                    net,
                    "checkpoints/best.pt",
                    meta={
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                    },
                )
                tqdm.write(
                    f"[Early Stop] New best validation accuracy: {val_acc:.4f} (epoch {epoch+1})"
                )
            else:
                # No improvement
                patience_counter += 1
                tqdm.write(
                    f"[Early Stop] No improvement for {patience_counter}/{patience} epochs (current: {val_acc:.4f}, best: {best_val_acc:.4f})"
                )

                if patience_counter >= patience:
                    tqdm.write(
                        f"[Early Stop] Training stopped early at epoch {epoch+1} due to no improvement for {patience} epochs"
                    )
                    tqdm.write(
                        f"[Early Stop] Best validation accuracy: {best_val_acc:.4f}"
                    )
                    break
        else:
            # Original checkpointing behavior (no early stopping)
            save_checkpoint(
                net,
                "checkpoints/last.pt",
                meta={
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                },
            )
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                save_checkpoint(
                    net,
                    "checkpoints/best.pt",
                    meta={
                        "epoch": epoch,
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc),
                    },
                )

        # Always save last checkpoint
        save_checkpoint(
            net,
            "checkpoints/last.pt",
            meta={
                "epoch": epoch,
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            },
        )

    # Final comprehensive test evaluation
    final_test_loss, final_test_acc = evaluate_model(
        net, test_loader, loss, device, num_steps, batch_size
    )
    print(
        f"Final Test Loss: {final_test_loss:.4f} | Final Test Acc: {final_test_acc*100:.2f}%"
    )

    # Average change in training accuracy due to sleep (only computed on sleep iterations)
    avg_sleep_acc_delta = float(np.mean(sleep_acc_deltas)) if sleep_acc_deltas else None
    if avg_sleep_acc_delta is not None:
        print(
            f"Average Δ train acc after sleep: {avg_sleep_acc_delta*100:+.2f}% over {len(sleep_acc_deltas)} sleep iterations"
        )

    # Append results to JSON as we go
    results_entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_index": run_index,
        "optimizer_mode": cli_args.optimizer_mode,
        "dataset": cli_args.dataset,
        "seed": cli_args.seed,
        "sleep_interval_pct": float(sleep_interval_pct),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "final_test_loss": float(final_test_loss),
        "final_test_acc": float(final_test_acc),
        "avg_train_acc_delta_after_sleep": avg_sleep_acc_delta,
    }
    results_path = cli_args.results_json
    # Normalize to absolute path (so message and file align with actual location)
    results_path = os.path.abspath(results_path)
    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    # Read existing list (if any)
    existing = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception as e:
            print(
                f"Warning: failed to read existing results JSON '{results_path}': {e}"
            )
            existing = []
    existing.append(results_entry)
    # Write must succeed; propagate any exception to stop training
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved results to {results_path}")

    # Plot Loss and Accuracy (controlled by orchestrator)
    if enable_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(loss_hist, label="Train Loss")
        ax1.plot(
            [i * len(train_loader) for i in range(len(test_loss_hist))],
            test_loss_hist,
            "o-",
            label="Val Loss",
        )
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot([acc * 100 for acc in train_acc_hist], label="Train Acc")
        ax2.plot(
            [i * len(train_loader) for i in range(len(val_acc_hist))],
            [acc * 100 for acc in val_acc_hist],
            "o-",
            label="Val Acc",
        )
        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return float(final_test_acc)


# Excel tracking functions
excel_path = "../GLM/Results_.xlsx"
model_name = "snntorch"
lambda_value = 0.99997

def get_next_run_number():
    """Get the next run number for the model from existing Excel file."""
    excel_abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, excel_path))
    if not os.path.exists(excel_abs_path):
        return 1
    try:
        df = pd.read_excel(excel_abs_path, engine="openpyxl")
        if "Model" not in df.columns or "Run" not in df.columns:
            return 1
        model_runs = df[df["Model"] == model_name]
        if model_runs.empty:
            return 1
        return int(model_runs["Run"].max()) + 1
    except Exception as e:
        print(f"WARNING: Could not read Excel file to determine run number: {e}")
        return 1

def save_to_excel(sleep_interval_pct, run_index, test_accuracy, run_number, dataset, seed_val):
    """Append a new row to the Excel file after a run completes."""
    if not getattr(cli_args, "track_excel", False):
        return
    try:
        if test_accuracy is None:
            print("WARNING: Test accuracy is None, skipping Excel update")
            return

        excel_abs_path = os.path.abspath(os.path.join(SCRIPT_DIR, excel_path))
        sleep_duration = float(sleep_interval_pct)
        dataset_name_val = dataset

        new_row = {
            "Sleep_duration": sleep_duration,
            "Model": model_name,
            "Run": run_number,
            "Lambda": lambda_value,
            "Seed": seed_val if seed_val is not None else run_index,
            "Dataset": dataset_name_val.lower(),
            "Accuracy": float(test_accuracy),
        }

        if os.path.exists(excel_abs_path) and os.path.getsize(excel_abs_path) > 0:
            try:
                df = pd.read_excel(excel_abs_path, engine="openpyxl")
                required_columns = [
                    "Sleep_duration",
                    "Model",
                    "Run",
                    "Lambda",
                    "Seed",
                    "Dataset",
                    "Accuracy",
                ]
                if df.empty or not all(col in df.columns for col in required_columns):
                    df = pd.DataFrame(columns=required_columns)
            except Exception:
                df = pd.DataFrame(
                    columns=[
                        "Sleep_duration",
                        "Model",
                        "Run",
                        "Lambda",
                        "Seed",
                        "Dataset",
                        "Accuracy",
                    ]
                )
        else:
            df = pd.DataFrame(
                columns=[
                    "Sleep_duration",
                    "Model",
                    "Run",
                    "Lambda",
                    "Seed",
                    "Dataset",
                    "Accuracy",
                ]
            )

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        os.makedirs(os.path.dirname(excel_abs_path), exist_ok=True)
        df.to_excel(excel_abs_path, index=False, engine="openpyxl")
        print(f"Results saved to {excel_abs_path}")
    except Exception as e:
        print(f"WARNING: Could not save to Excel file: {e}")


# Orchestrate single or multiple sleep rates and runs
rates = (
    cli_args.sleep_interval_pct
    if isinstance(cli_args.sleep_interval_pct, list)
    else [cli_args.sleep_interval_pct]
)

# Determine run number for Excel tracking (same for all runs in this execution)
excel_run_number = get_next_run_number() if getattr(cli_args, "track_excel", False) else None

if len(rates) == 1 and cli_args.runs <= 1:
    should_plot = not cli_args.no_plot
    acc = train_once(
        run_index=1,
        cli_args=cli_args,
        sleep_interval_pct=rates[0],
        enable_plot=should_plot,
    )
    if excel_run_number is not None:
        save_to_excel(rates[0], 1, acc, excel_run_number, cli_args.dataset, cli_args.seed)
else:
    for rate in rates:
        accuracies = []
        print(
            f"\n===== SLEEP RATE {rate} ({rate*100:.1f}%) — {cli_args.runs} run(s) ====="
        )
        for r in range(cli_args.runs):
            print(f"\n  RUN {r+1}/{cli_args.runs}")
            acc = train_once(
                run_index=r + 1,
                cli_args=cli_args,
                sleep_interval_pct=rate,
                enable_plot=False,
            )
            accuracies.append(acc)
            if excel_run_number is not None:
                save_to_excel(rate, r + 1, acc, excel_run_number, cli_args.dataset, cli_args.seed)
        acc_pct = [a * 100.0 for a in accuracies]
        print("\n" + "-" * 60)
        print(f"Sleep rate {rate} — final test accuracies (%):")
        print(", ".join([f"{a:.2f}" for a in acc_pct]))
        print(
            f"Mean: {np.mean(acc_pct):.2f}% | Std: {np.std(acc_pct):.2f}% | Best: {np.max(acc_pct):.2f}%"
        )
