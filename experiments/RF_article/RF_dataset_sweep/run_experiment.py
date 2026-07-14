"""
Dataset sweep: the canonical network across the ready static-image datasets.

The network config is fixed to the canonical 1024/225 oriented_rf regime
(generic_testing/run_test.py defaults). Two axes are swept:

    dataset   : mnist  ->  kmnist  ->  fmnist  ->  svhn  ->  cifar10
    condition : trained (TraceSTDP) vs frozen (--freeze-weights, initial weights fixed)

The trained-vs-frozen A/B is the point of the experiment: it isolates what plasticity
contributes, and varying the dataset probes whether training helps more where there is
more headroom. The frozen control is genuinely static — with train_weights=False the
STDP update is gated off, and neuron-wise Normalize only rescales weights back to their
initial sums (a no-op when they never drift), so frozen weights stay at init.

All four datasets are 28x28 grayscale, 10-class, and go through the identical Poisson
rate-encoding (ImageDataStreamer), so the comparison is apples-to-apples — only the
task changes, not the pipeline. (Event-encoded datasets like N-MNIST / DVS-Gesture are
NOT supported here: there is no loader, and the input is event streams over a
non-square / multi-channel grid rather than a static image to rate-encode — that would
be a separate experiment.)

Sample counts
-------------
The streamer merges train+test into one pool and carves train/val/test by COUNT, giving
val/test whatever is left after train. CIFAR-10's pool is exactly 60k (50k+10k), so the
default must not exceed 60k total — we use 49k/1k/10k (=60k) to fill it without clamp.

    mnist/kmnist/fmnist : train 59000 / val 1000 / test 10000   (70k pool)
    cifar10             : train 49000 / val 1000 / test 10000   (60k pool — exact fill)
    svhn                : train 59000 / val 1000 / test 10000   (99.3k pool)

CIFAR-10 and SVHN
-----------------
Both are 32x32 RGB. The ImageDataStreamer transform chain (Grayscale -> Resize -> ToTensor)
converts them to 28x28 single-channel automatically. SVHN uses split='train'/'test' in
torchvision (not train=True/False); this is handled inside _get_torchvision_splits in
get_data.py. SVHN labels are 0-9 (torchvision remaps the original label-10 to 0).

Usage
-----
# Local smoke-test (small/fast)
python experiments/RF_article/RF_dataset_sweep/run_experiment.py \
    --dataset mnist --seed 0 --train-all 800 --val-all 200 --test-all 300

# Full cell (used by the SLURM array)
python experiments/RF_article/RF_dataset_sweep/run_experiment.py \
    --dataset kmnist --seed 1 \
    --output-dir results/kmnist_s1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import neurosnn as snn

# Canonical network — generic_testing/run_test.py engaged-inhibition regime, 1024/225.
N_EXC = 1024
N_INH = 225
PEAK_SE = 4.0
PEAK_EE = 1.0
PEAK_EI = 2.0
PEAK_IE = -2.0
DENSITY_SE = 0.01
DENSITY_EE = 0.01
DENSITY_EI = 0.03
DENSITY_IE = 0.05

# Per-dataset default (train, val, test) counts. notmnist is capped to fit the
# ~18.7k NotMNIST-small pool so the streamer leaves room for val/test.
DEFAULT_COUNTS = {
    "mnist":        (59000, 1000, 10000),
    "kmnist":       (59000, 1000, 10000),
    "fmnist":       (59000, 1000, 10000),
    "fashionmnist": (59000, 1000, 10000),
    "notmnist":     (15000, 1000,  2000),  # kept for backward compat; not in sweep
    # CIFAR-10 pool = 50k+10k = 60k total; 49k+1k+10k = 60k exactly avoids clamp-to-zero
    "cifar10":      (49000, 1000, 10000),
    # SVHN pool = 73257+26032 = 99.3k total; 59k+1k+10k = 70k fits comfortably
    "svhn":         (59000, 1000, 10000),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNN dataset sweep (canonical network across static-image datasets)"
    )

    # --- Sweep parameters ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "cifar10", "svhn"],
        help="Static-image dataset (all 28x28 grayscale, 10-class)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        default=False,
        help="Disable STDP — run with the initial weights fixed. The trained-vs-frozen "
        "A/B that isolates what plasticity contributes (default: False = weights learn)",
    )

    # --- Training control. Counts default to None -> per-dataset DEFAULT_COUNTS. ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--train-all", type=int, default=None)
    parser.add_argument("--train-batch", type=int, default=1000)
    parser.add_argument("--val-all", type=int, default=None)
    parser.add_argument("--val-batch", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=None)
    parser.add_argument("--test-batch", type=int, default=10000)

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Append one summary line here for live tracking "
        "(default: <results-root>/dataset_sweep.jsonl)",
    )

    return parser.parse_args()


def resolve_counts(args):
    """Per-dataset defaults, overridden by any explicitly passed count flag."""
    d_train, d_val, d_test = DEFAULT_COUNTS[args.dataset]
    train_all = args.train_all if args.train_all is not None else d_train
    val_all = args.val_all if args.val_all is not None else d_val
    test_all = args.test_all if args.test_all is not None else d_test
    return train_all, val_all, test_all


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    condition = "frozen" if args.freeze_weights else "trained"
    tag = f"{args.dataset}_{condition}_s{args.seed}"
    return os.path.join(base, tag)


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON line for live tracking. Best-effort; never fatal.

    aggregate_dataset.py rebuilds an authoritative jsonl from the per-cell
    results.json, so a partial / interleaved live jsonl is harmless.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"Warning: could not append to {path} ({e})")


def main():
    args = parse_args()
    train_all, val_all, test_all = resolve_counts(args)
    condition = "frozen" if args.freeze_weights else "trained"

    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Weights — canonical oriented_rf (fixed across datasets)
    # ------------------------------------------------------------------
    weights = snn.weights.oriented_receptive_fields(
        density_se=DENSITY_SE,
        density_ee=DENSITY_EE,
        density_ei=DENSITY_EI,
        density_ie=DENSITY_IE,
        peak_se=PEAK_SE,
        peak_ee=PEAK_EE,
        peak_ei=PEAK_EI,
        peak_ie=PEAK_IE,
        sigma_x=3.0,
        sigma_x_lognormal_std=0.0,
        sigma_x_lognormal_max=0.0,
        n_orientations=4,
        orientation_mode="block",
        sigma_ee_mean=0.0,
        sigma_ee_lognormal_std=0.5,
        ablate_ee=False,
        ablate_ie=False,
    )

    # ------------------------------------------------------------------
    # Layer
    # ------------------------------------------------------------------
    layer = snn.Layer(
        N_exc=N_EXC,
        N_inh=N_INH,
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0,
            tau_m_inh=15.0,
            tau_syn_exc=10.0,
            tau_syn_inh=9.0,
            membrane_resistance_exc=15.0,
            membrane_resistance_inh=15.0,
            resting_potential=-70.0,
            reset_potential=-80.0,
            spike_threshold=-55.0,
            min_mp=-100.0,
            max_mp=40.0,
            mean_noise=0.0,
            var_noise=0.0,
            spike_adaptation=True,
            tau_adaptation=200.0,
            delta_adaptation=0.5,
        ),
        weights=weights,
    )

    # ------------------------------------------------------------------
    # Learner — run_test default (static x_tar, engaged-inhibition regime)
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,
        tau_trace=20,
        w_max=10.0,
        mu_weight=0.5,
        x_tar_mode="static",
        x_tar_pct_se=60.0,
        x_tar_pct_ee=30.0,
        x_tar_static_se=1.0,
        x_tar_static_ee=0.5,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    # ------------------------------------------------------------------
    # Regulariser — fixed: neuron-wise normalisation
    # ------------------------------------------------------------------
    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=args.seed,
        num_steps=350,
        all_images_train=train_all,
        batch_image_train=args.train_batch,
        all_images_val=val_all,
        batch_image_val=args.val_batch,
        all_images_test=test_all,
        batch_image_test=args.test_batch,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    config = dict(
        dataset=args.dataset,
        condition=condition,
        freeze_weights=args.freeze_weights,
        weight_type="oriented_rf",
        seed=args.seed,
        epochs=args.epochs,
        val_every=args.val_every,
        n_exc=N_EXC,
        n_inh=N_INH,
        peak_se=PEAK_SE,
        peak_ee=PEAK_EE,
        peak_ei=PEAK_EI,
        peak_ie=PEAK_IE,
        train_all=train_all,
        val_all=val_all,
        test_all=test_all,
    )
    print(
        f"\nConfig — dataset: {args.dataset} | condition: {condition}"
        f" | weight_type: oriented_rf | N_exc: {N_EXC} | N_inh: {N_INH}"
        f" | train/val/test: {train_all}/{val_all}/{test_all}"
        f" | seed: {args.seed} | epochs: {args.epochs}\n"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []

    t_start = time.time()

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=not args.freeze_weights,
        save_model=True,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=False,
        pca_variance=0.95,
        stat_tracking_frequency=10500,
    ):
        if result.accuracy is None:
            continue

        if result.batch % args.val_every == 0:
            val = model.validate()
            val_acc = val.accuracy if val.accuracy is not None else float("nan")
            val_phi = val.phi if val.phi is not None else float("nan")
            print(
                f"epoch {result.epoch + 1}  batch {result.batch:>3}"
                f"  train_acc {result.accuracy:.4f}  train_phi {result.phi:.3f}"
                f"  val_acc {val_acc:.4f}  val_phi {val_phi:.3f}"
            )
            val_history.append(
                {"batch": result.batch, "val_acc": val_acc, "val_phi": val_phi}
            )
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy
                best_val_phi = val_phi

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test phi      : {test_phi:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Elapsed (s)   : {elapsed:.1f}")
    print(f"{'=' * 55}")

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    results = {
        "config": config,
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "elapsed_s": round(elapsed, 1),
        "val_history": val_history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")

    # Live-tracking jsonl (one summary line per run). aggregate_dataset.py rebuilds an
    # authoritative jsonl from the per-cell results.json, so this is best-effort only.
    jsonl_path = args.jsonl or os.path.join(
        os.path.dirname(__file__), "results", "dataset_sweep.jsonl"
    )
    append_jsonl(
        jsonl_path,
        {
            "dataset": args.dataset,
            "condition": condition,
            "seed": args.seed,
            "test_acc": test_acc,
            "test_phi": test_phi,
            "best_val_acc": best_val_acc,
            "best_val_phi": best_val_phi,
            "elapsed_s": round(elapsed, 1),
        },
    )
    print(f"Tracking row appended -> {jsonl_path}")


if __name__ == "__main__":
    main()
