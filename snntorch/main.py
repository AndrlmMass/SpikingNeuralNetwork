import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from snntorch import spikeplot as splt
import pandas as pd
from snntorch import spikegen
import snntorch as snn
import itertools
from tqdm import tqdm
import argparse
import os
import sys
import json
from datetime import datetime

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
val_loader = None

# Network architecture (num_inputs inferred after dataset init)
num_inputs = None
num_hidden = 1000
num_outputs = 10

# Temporal dynamics
num_steps = 25
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
sleep_mem_noise_std = 0.5
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
    "---track-excel", action="store_true", help="Store results in excel sheet for GLMM"
)
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
    nargs="+",
    default=["MNIST"],
    choices=["MNIST", "KMNIST", "FMNIST"],
    help="Dataset(s) to use. Can specify multiple: --dataset MNIST KMNIST FMNIST",
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
    "--train-size",
    type=int,
    default=None,
    help="Number of training samples to use (evenly split across classes). If None, use full dataset.",
)
parser.add_argument(
    "--test-size",
    type=int,
    default=None,
    help="Number of test samples to use (evenly split across classes). If None, use full dataset.",
)
parser.add_argument(
    "--val-size",
    type=int,
    default=None,
    help="Number of validation samples to use (evenly split across classes). If specified, splits from training set. If None, test set is used for validation.",
)
parser.add_argument("--help", action="help", help="Show help and exit")
cli_args, _ = parser.parse_known_args()


# ---------------------- Dataset Selection & Loaders ----------------------
def create_balanced_subset(dataset, target_size, num_classes=10, seed=42):
    """
    Create a balanced subset of the dataset with equal samples per class.

    Args:
        dataset: PyTorch dataset with targets
        target_size: Total number of samples (will be rounded to nearest multiple of num_classes)
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility

    Returns:
        indices: List of indices to use for the balanced subset
    """
    if target_size is None:
        return None  # Use full dataset

    # Ensure target_size is a multiple of num_classes for even split
    samples_per_class = target_size // num_classes
    actual_size = samples_per_class * num_classes

    if actual_size != target_size:
        print(
            f"Warning: target_size {target_size} is not divisible by {num_classes}. Using {actual_size} samples ({samples_per_class} per class)."
        )

    # Get all indices grouped by class
    indices_by_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        if isinstance(target, torch.Tensor):
            target = target.item()
        indices_by_class[target].append(idx)

    # Randomly sample from each class
    np.random.seed(seed)
    selected_indices = []
    for class_indices in indices_by_class:
        if len(class_indices) < samples_per_class:
            print(
                f"Warning: Class has only {len(class_indices)} samples, but {samples_per_class} requested. Using all available samples."
            )
            selected_indices.extend(class_indices)
        else:
            selected = np.random.choice(
                class_indices, size=samples_per_class, replace=False
            )
            selected_indices.extend(selected.tolist())

    return selected_indices


dataset_name = cli_args.dataset.upper()
folder_map = {
    "MNIST": "mnist",
    "KMNIST": "kmnist",
    "FMNIST": "fmnist",
}
data_path = f"./data/{folder_map.get(dataset_name, 'mnist')}"

if dataset_name == "MNIST":
    TrainDS = datasets.MNIST
elif dataset_name == "KMNIST":
    TrainDS = datasets.KMNIST
elif dataset_name == "FMNIST":
    TrainDS = datasets.FashionMNIST
else:
    TrainDS = datasets.MNIST

# Load full datasets first
full_train_dataset = TrainDS(data_path, train=True, download=True, transform=transform)
full_test_dataset = TrainDS(data_path, train=False, download=True, transform=transform)

# Create balanced subsets if specified
num_classes = 10

# Handle validation set - if val_size is specified, we need to split training data
val_dataset = None
val_indices = None
if cli_args.val_size is not None:
    # Create validation set from training data
    # First, determine total samples needed from training set
    train_size_needed = (
        cli_args.train_size or len(full_train_dataset)
    ) + cli_args.val_size
    all_train_indices = create_balanced_subset(
        full_train_dataset, train_size_needed, num_classes=num_classes
    )

    # Split into train and val by class
    samples_per_class_total = train_size_needed // num_classes
    samples_per_class_val = cli_args.val_size // num_classes
    samples_per_class_train = samples_per_class_total - samples_per_class_val

    # Group indices by class
    indices_by_class = [[] for _ in range(num_classes)]
    for idx in all_train_indices:
        _, target = full_train_dataset[idx]
        if isinstance(target, torch.Tensor):
            target = target.item()
        indices_by_class[target].append(idx)

    # Split each class: first samples_per_class_val go to val, rest to train
    train_indices = []
    val_indices = []
    for class_indices in indices_by_class:
        val_indices.extend(class_indices[:samples_per_class_val])
        train_indices.extend(
            class_indices[
                samples_per_class_val : samples_per_class_val + samples_per_class_train
            ]
        )
else:
    # No validation set needed, just create train subset if specified
    train_indices = create_balanced_subset(
        full_train_dataset, cli_args.train_size, num_classes=num_classes
    )

# Create test subset if specified
test_indices = create_balanced_subset(
    full_test_dataset, cli_args.test_size, num_classes=num_classes
)

# Create subsets
if train_indices is not None:
    train_dataset = Subset(full_train_dataset, train_indices)
    print(
        f"Using {len(train_indices)} training samples ({len(train_indices) // num_classes} per class)"
    )
else:
    train_dataset = full_train_dataset
    print(f"Using full training dataset: {len(train_dataset)} samples")

if test_indices is not None:
    test_dataset = Subset(full_test_dataset, test_indices)
    print(
        f"Using {len(test_indices)} test samples ({len(test_indices) // num_classes} per class)"
    )
else:
    test_dataset = full_test_dataset
    print(f"Using full test dataset: {len(test_dataset)} samples")

if val_indices is not None:
    val_dataset = Subset(full_train_dataset, val_indices)
    print(
        f"Using {len(val_indices)} validation samples ({len(val_indices) // num_classes} per class)"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
else:
    val_dataset = None
    val_loader = None
    print("Using test set for validation")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
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
        val_data_loader = val_loader if val_loader is not None else test_loader
        val_loss, val_acc = evaluate_model(
            net, val_data_loader, loss, device, num_steps, batch_size
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
    try:
        results_entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "run_index": run_index,
            "optimizer_mode": cli_args.optimizer_mode,
            "dataset": cli_args.dataset,
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
            except Exception:
                existing = []
        existing.append(results_entry)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"Saved results to {results_path}")
    except Exception as e:
        print(f"Warning: failed to save results JSON: {e}")

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


def get_next_run_number(excel_path, model_name):
    """Get the next run number for the model from existing Excel file."""
    if not os.path.exists(excel_path):
        return 1
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        if "Model" not in df.columns or "Run" not in df.columns:
            return 1
        model_runs = df[df["Model"] == model_name]
        if model_runs.empty:
            return 1
        return int(model_runs["Run"].max()) + 1
    except Exception as e:
        print(f"WARNING: Could not read Excel file to determine run number: {e}")
        return 1


def save_to_excel(
    Sleep_duration, Model, Run, Lambda, Seed, Dataset, Accuracy, excel_path
):
    new_row = {
        "Sleep_duration": Sleep_duration,
        "Model": Model,
        "Run": Run,
        "Lambda": Lambda,
        "Seed": Seed,
        "Dataset": Dataset,
        "Accuracy": Accuracy,
    }

    df = pd.read_excel(excel_path, engine="openpyxl")
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

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Results saved to {excel_path}")


# Orchestrate single or multiple sleep rates and runs
rates = (
    cli_args.sleep_interval_pct
    if isinstance(cli_args.sleep_interval_pct, list)
    else [cli_args.sleep_interval_pct]
)

if len(rates) == 1 and cli_args.runs <= 1:
    _ = train_once(
        run_index=1, cli_args=cli_args, sleep_interval_pct=rates[0], enable_plot=True
    )
else:
    Run = get_next_run_number(excel_path=cli_args.track_excel, model_name="snntorch")
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
            save_to_excel(
                Sleep_duration=rate,
                Model="snntorch",
                Run=Run,
                Lambda=sleep_lambda,
                Seed=r + 1,
                Dataset=cli_args.dataset,
                Accuracy=acc,
                excel_path=cli_args.track_excel,
            )
        acc_pct = [a * 100.0 for a in accuracies]
        print("\n" + "-" * 60)
        print(f"Sleep rate {rate} — final test accuracies (%):")
        print(", ".join([f"{a:.2f}" for a in acc_pct]))
        print(
            f"Mean: {np.mean(acc_pct):.2f}% | Std: {np.std(acc_pct):.2f}% | Best: {np.max(acc_pct):.2f}%"
        )
