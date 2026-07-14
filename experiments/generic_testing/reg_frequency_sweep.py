"""
Local sweep over the REGULARIZATION INTERVAL (Normalize frequency).

Question
--------
Normalize rescales the excitatory weights every ``frequency`` awake timesteps
(num_steps = 350 per image, so frequency=1050 == every 3 images). If we
normalize too often we may over-constrain the weights and stop TraceSTDP from
ever shaping the receptive fields. This sweep varies that interval to see
whether *less frequent* regularization (larger interval) gives better learning,
or whether *more frequent* (smaller interval) helps — and where the sweet spot
is.

Design
------
Everything else is the canonical 1024/225 oriented_rf regime (generic_testing
defaults). Two axes:

    reg interval (frequency) : 100  300  1000  3000  10000   (awake timesteps)
    seed                     : 0 1 2

5 intervals x 3 seeds = 15 short runs. Built to run LOCALLY end to end:

    python experiments/generic_testing/reg_frequency_sweep.py

Each run trains on a small slice (default 6000 train images, 1 epoch) so the
whole sweep finishes in a sitting. Results (jsonl + xlsx + a metric-vs-interval
plot) are written under the top-level results/ tree. The jsonl is appended after
each run so a crash mid-sweep keeps everything finished so far.

Override the grid / sizes:
    python experiments/generic_testing/reg_frequency_sweep.py \
        --reg-freqs 100,1000,10000 --seeds 0,1 --train-all 4000 --test-all 1500
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Canonical network — generic_testing engaged-inhibition regime, 1024/225.
N_EXC, N_INH = 1024, 225


def parse_args():
    p = argparse.ArgumentParser(
        description="Local sweep over the Normalize regularization interval"
    )
    p.add_argument(
        "--reg-freqs",
        type=str,
        default="100,300,1000,3000,10000",
        help="Comma-separated Normalize intervals in awake timesteps "
        "(default: 100,300,1000,3000,10000)",
    )
    p.add_argument(
        "--seeds", type=str, default="0,1,2",
        help="Comma-separated seeds (default: 0,1,2)",
    )
    p.add_argument(
        "--reg-mode", type=str, default="neuron",
        choices=["static", "layer", "neuron"],
        help="Normalize mode (default: neuron)",
    )
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--val-every", type=int, default=2)
    # Small slices so the whole 15-run sweep runs locally in one sitting.
    p.add_argument("--train-all", type=int, default=6000)
    p.add_argument("--train-batch", type=int, default=1000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-batch", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=2000)
    p.add_argument("--test-batch", type=int, default=2000)
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output dir (default: <repo>/results/reg_freq_sweep/<dataset>/<date>)",
    )
    return p.parse_args()


def build_model(seed: int, args):
    """Fresh canonical components for one run."""
    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01, density_ee=0.01, density_ei=0.03, density_ie=0.05,
        peak_se=4.0, peak_ee=1.0, peak_ei=2.0, peak_ie=-2.0,
        sigma_x=3.0, sigma_x_lognormal_std=0.0, sigma_x_lognormal_max=0.0,
        n_orientations=4, orientation_mode="block",
        sigma_ee_mean=0.0, sigma_ee_lognormal_std=0.5,
        ablate_ee=False, ablate_ie=False,
    )
    layer = snn.Layer(
        N_exc=N_EXC, N_inh=N_INH,
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
            membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
            resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
            min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
            spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5,
        ),
        weights=weights,
    )
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004, tau_trace=20, w_max=10.0, mu_weight=0.5,
        x_tar_mode="static", x_tar_pct_se=60.0, x_tar_pct_ee=30.0,
        x_tar_static_se=1.0, x_tar_static_ee=0.5, update_freq=100,
        clip_weights=True, min_weight_exc=0.01, max_weight_exc=25.0,
        min_weight_inh=-25.0, max_weight_inh=-0.01,
    )
    model = snn.Model(
        input_size=784, classes=list(range(10)), random_state=seed, num_steps=350,
        all_images_train=args.train_all, batch_image_train=args.train_batch,
        all_images_val=args.val_all, batch_image_val=args.val_batch,
        all_images_test=args.test_all, batch_image_test=args.test_batch,
        image_dataset=args.dataset, max_rate_hz=90.0, gain=1.0, gabor=False,
    )
    return layer, learner, model


def run_one(reg_freq: int, seed: int, args) -> dict:
    layer, learner, model = build_model(seed, args)
    regularizer = snn.regularizer.Normalize(frequency=reg_freq, mode=args.reg_mode)

    best_val_acc, best_val_phi = 0.0, float("nan")
    t0 = time.time()
    for result in model.train(
        layers=[layer], learner=learner, regularizer=regularizer,
        epochs=args.epochs, train_weights=True, save_model=False,
        accuracy_method="pca_lr", use_LR=True, use_phi=True, use_pca=False,
        pca_variance=0.95, stat_tracking_frequency=10**9,
    ):
        if result.accuracy is None:
            continue
        if result.batch % args.val_every == 0:
            val = model.validate()
            vp = val.phi if val.phi is not None else float("nan")
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc, best_val_phi = val.accuracy, vp
    elapsed = time.time() - t0

    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    # release the per-run objects before the next config
    del layer, learner, model, regularizer
    gc.collect()

    return {
        "reg_freq": reg_freq,
        "reg_mode": args.reg_mode,
        "seed": seed,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "elapsed_s": round(elapsed, 1),
    }


def make_plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pandas as pd

    df = pd.DataFrame(rows)
    panels = [("test_acc", "Test accuracy", "#2a78d6"),
              ("test_phi", "Test phi", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for (col, title, color), ax in zip(panels, axes):
        means = df.groupby("reg_freq")[col].mean()
        ax.scatter(df["reg_freq"], df[col], color=color, alpha=0.5, s=42,
                   zorder=3, label="per seed")
        ax.plot(means.index, means.values, color=color, lw=2, marker="o",
                zorder=4, label="mean")
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["reg_freq"].unique()))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("regularization interval  (awake timesteps, log)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title + " vs regularization interval", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Regularization-interval sweep (canonical 1024/225, TraceSTDP)",
                 fontsize=15)
    out = os.path.join(out_dir, "reg_freq_sweep.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot   -> {out}")
    xlsx = os.path.join(out_dir, "Results_reg_freq.xlsx")
    df.sort_values(["reg_freq", "seed"]).to_excel(xlsx, index=False, engine="openpyxl")
    print(f"  Saved table  -> {xlsx}")


def main():
    args = parse_args()
    freqs = [int(x) for x in args.reg_freqs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "reg_freq_sweep", args.dataset,
        date.today().strftime("%Y.%m.%d"),
    )
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "reg_freq_sweep.jsonl")
    open(jsonl_path, "w").close()  # fresh

    grid = [(f, s) for f in freqs for s in seeds]
    print(f"\nRegularization-interval sweep: {len(freqs)} intervals x {len(seeds)} "
          f"seeds = {len(grid)} runs")
    print(f"intervals : {freqs}")
    print(f"seeds     : {seeds}")
    print(f"counts    : train {args.train_all} / val {args.val_all} / test {args.test_all}"
          f" | epochs {args.epochs} | reg-mode {args.reg_mode}")
    print(f"output    : {out_dir}\n")

    rows = []
    for i, (freq, seed) in enumerate(grid, 1):
        print(f"[{i:>2}/{len(grid)}] reg_freq={freq:<6} seed={seed} ...", flush=True)
        row = run_one(freq, seed, args)
        rows.append(row)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"          -> test_acc {row['test_acc']:.4f}  test_phi {row['test_phi']:.4f}"
              f"  best_val {row['best_val_acc']:.4f}  ({row['elapsed_s']:.0f}s)")

    print(f"\n  Wrote {len(rows)} rows -> {jsonl_path}")
    make_plot(rows, out_dir)

    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby("reg_freq")[["test_acc", "test_phi"]].agg(["mean", "std"])
    print("\n  Mean +/- sd per interval:")
    with pd.option_context("display.width", 120):
        print(summ.to_string())


if __name__ == "__main__":
    main()
