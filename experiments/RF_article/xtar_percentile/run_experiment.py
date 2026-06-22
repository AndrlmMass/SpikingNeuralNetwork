"""
x_tar percentile sweep: threshold estimator for the trace-STDP rule.

The STDP rule  Δw = lr·(trace[j] − x_tar)·(w_max − w)^mu  uses x_tar as the
threshold separating presynaptic potentiation (trace above x_tar) from depression
(below). This experiment sweeps x_tar as a *percentile over the active
(nonzero-trace) sub-population*, independently per layer:

    SE percentile (input→exc)  — higher  ⇒ depress weak/surround pixels ⇒ sharper RFs
    EE percentile (exc→exc)    — lower   ⇒ spare moderately-active neurons (anti-domination)

It is run against the original "mean" estimator as a baseline. The network config
matches the current engaged-inhibition regime (peak_se=4, peak_ei=2, peak_ie=-2,
mu=0.5, w_max=10, neuron-wise normalisation) so the sweep tunes the network we are
actually using.

Diagnostics (RF entropy/Gini/cosine/participation-ratio, E/I balance, active-E
fraction, trace spread) are recorded every batch via track_stats and persisted to
results.json["stats_history"] so RF specialisation can be tracked across the full
59-batch epoch.

Usage
-----
# Local smoke-test (small, fast)
python experiments/RF_article/xtar_percentile/run_experiment.py \
    --x-tar-mode percentile --x-tar-pct-se 60 --x-tar-pct-ee 30 \
    --seed 0 --train-all 5000 --dataset mnist

# Baseline (mean estimator)
python experiments/RF_article/xtar_percentile/run_experiment.py --x-tar-mode mean --seed 0

# Full MNIST grid cell (used by the SLURM array)
python experiments/RF_article/xtar_percentile/run_experiment.py \
    --x-tar-mode percentile --x-tar-pct-se 70 --x-tar-pct-ee 30 --seed 1 \
    --output-dir results/se70_ee30_s1
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNN x_tar percentile sweep (SE/EE thresholds vs mean baseline)"
    )

    # --- Sweep parameters ---
    parser.add_argument(
        "--x-tar-mode",
        type=str,
        default="percentile",
        choices=["mean", "percentile"],
        help="x_tar estimator: 'mean' (baseline) or 'percentile' over active traces",
    )
    parser.add_argument(
        "--x-tar-pct-se",
        type=float,
        default=60.0,
        help="SE percentile over active input traces (percentile mode only)",
    )
    parser.add_argument(
        "--x-tar-pct-ee",
        type=float,
        default=30.0,
        help="EE percentile over active exc traces (percentile mode only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # --- STDP / learning rule (held fixed at the current regime) ---
    parser.add_argument("--learning-rate", type=float, default=0.0004)
    parser.add_argument("--mu-weight", type=float, default=0.5)
    parser.add_argument("--w-max", type=float, default=10.0)

    # --- Dataset / training control ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "notmnist", "geomfig", "fcx1"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--train-all", type=int, default=59000)
    parser.add_argument("--train-batch", type=int, default=1000)
    parser.add_argument("--val-all", type=int, default=1000)
    parser.add_argument("--val-batch", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=10000)
    parser.add_argument("--test-batch", type=int, default=10000)

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default=None)

    return parser.parse_args()


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    if args.x_tar_mode == "mean":
        tag = f"mean_s{args.seed}"
    else:
        tag = f"se{int(args.x_tar_pct_se)}_ee{int(args.x_tar_pct_ee)}_s{args.seed}"
    return os.path.join(base, tag)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Weights — current engaged-inhibition regime
    # ------------------------------------------------------------------
    weights = snn.weights.receptive_fields(
        density_se=0.01,
        density_ee=0.01,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=4.0,
        peak_ee=1.0,
        peak_ei=2.0,
        peak_ie=-2.0,
    )

    # ------------------------------------------------------------------
    # Layer
    # ------------------------------------------------------------------
    layer = snn.Layer(
        N_exc=1024,
        N_inh=225,
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
    # Learner — x_tar config is the swept dimension
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=args.learning_rate,
        tau_trace=20,
        w_max=args.w_max,
        mu_weight=args.mu_weight,
        x_tar_mode=args.x_tar_mode,
        x_tar_pct_se=args.x_tar_pct_se,
        x_tar_pct_ee=args.x_tar_pct_ee,
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
        all_images_train=args.train_all,
        batch_image_train=args.train_batch,
        all_images_val=args.val_all,
        batch_image_val=args.val_batch,
        all_images_test=args.test_all,
        batch_image_test=args.test_batch,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    config = dict(
        dataset=args.dataset,
        weight_type="rf",
        seed=args.seed,
        epochs=args.epochs,
        val_every=args.val_every,
        x_tar_mode=args.x_tar_mode,
        x_tar_pct_se=args.x_tar_pct_se,
        x_tar_pct_ee=args.x_tar_pct_ee,
        learning_rate=args.learning_rate,
        mu_weight=args.mu_weight,
        w_max=args.w_max,
    )
    print(
        f"\nConfig — dataset: {args.dataset} | x_tar: {args.x_tar_mode}"
        f" | pct_se: {args.x_tar_pct_se} | pct_ee: {args.x_tar_pct_ee}"
        f" | seed: {args.seed} | epochs: {args.epochs}\n"
    )

    # ------------------------------------------------------------------
    # Training — record every batch's diagnostics for the RF trajectory
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []
    stats_history = []

    t_start = time.time()

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=False,
        pca_variance=0.95,
        track_stats=True,
        stat_tracking_frequency=10500,
    ):
        # capture the per-batch diagnostic scalars (RF cosine/Gini/etc.)
        if result.stats:
            stats_history.append(
                {"epoch": result.epoch, "batch": result.batch, **result.stats}
            )

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
        "stats_history": stats_history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()
