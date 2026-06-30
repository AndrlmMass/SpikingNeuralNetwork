"""
Does the representation erode over the training stream WITHOUT weight changes?

Hypothesis (the confounder)
---------------------------
When training, accuracy drifts down over the stream. That could be the weight
updates (over-regularization / STDP), OR it could be neuronal dynamics alone:
spike adaptation and the adaptive threshold accumulate across the stream (state
is threaded batch->batch, never reset per image), which can erode the population
response regardless of weights.

The current frozen control can't see this: with train_weights=False the runner
processes a single batch then jumps to val/test (runner.py `range(... else 1)`).
So it never lets dynamics accumulate. We added a `run_full_stream` flag that
pushes the *whole* stream through the network with weights frozen.

This script compares the on-stream trajectory of two conditions:

    trained      : train_weights=True,  Normalize regularizer (canonical)
    frozen_full  : train_weights=False, run_full_stream=True, NO regularizer
                   (weights truly fixed; only neuronal dynamics evolve)

The decisive metric is train_acc / train_phi measured every batch (the on-stream
representation). val_acc is also logged — note validation rebuilds fresh state,
so for frozen_full it changes ONLY through the per-batch readout refit on the
(possibly eroding) training representation.

Read it
-------
  * frozen_full train_acc also declines  -> erosion is dynamics-driven
    (adaptation buildup) — the confounder is real.
  * frozen_full train_acc stays flat     -> the trained decline is genuinely
    from weight updates; the reg-interval sweep is the right follow-up.

Run (local, short):
    python experiments/generic_testing/frozen_stream_erosion.py
    python experiments/generic_testing/frozen_stream_erosion.py \
        --seeds 0,1,2 --train-all 12000 --train-batch 250
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
N_EXC, N_INH = 1024, 225
CONDITIONS = ["trained", "frozen_full"]


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-vs-trained on-stream erosion test")
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--reg-freq", type=int, default=1050,
                   help="Normalize interval for the trained condition (timesteps)")
    p.add_argument("--reg-mode", type=str, default="neuron")
    # Many small batches => fine temporal resolution on the trajectory.
    p.add_argument("--train-all", type=int, default=10000)
    p.add_argument("--train-batch", type=int, default=250)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-batch", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=2000)
    p.add_argument("--test-batch", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=5,
                   help="Validate every N batches (fresh-state eval)")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def build_model(seed, args):
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


def run_condition(condition, seed, args):
    layer, learner, model = build_model(seed, args)
    if condition == "trained":
        regularizer = snn.regularizer.Normalize(frequency=args.reg_freq, mode=args.reg_mode)
        train_weights, full_stream = True, False
    else:  # frozen_full: weights fixed, but full stream so dynamics accumulate
        regularizer = None
        train_weights, full_stream = False, True

    records = []
    t0 = time.time()
    for result in model.train(
        layers=[layer], learner=learner, regularizer=regularizer,
        epochs=1, train_weights=train_weights, run_full_stream=full_stream,
        save_model=False, accuracy_method="pca_lr", use_LR=True, use_phi=True,
        use_pca=False, pca_variance=0.95, stat_tracking_frequency=10**9,
    ):
        if result.accuracy is None:
            continue
        val_acc = val_phi = float("nan")
        if result.batch % args.val_every == 0:
            val = model.validate()
            val_acc = val.accuracy if val.accuracy is not None else float("nan")
            val_phi = val.phi if val.phi is not None else float("nan")
        records.append({
            "condition": condition, "seed": seed, "batch": result.batch,
            "train_acc": result.accuracy, "train_phi": result.phi,
            "val_acc": val_acc, "val_phi": val_phi,
        })
    elapsed = time.time() - t0

    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    del layer, learner, model, regularizer
    gc.collect()
    print(f"    {condition} seed {seed}: {len(records)} batches, "
          f"test_acc {test_acc:.4f}, test_phi {test_phi:.4f} ({elapsed:.0f}s)")
    return records, {"condition": condition, "seed": seed,
                     "test_acc": test_acc, "test_phi": test_phi}


def make_plot(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    colors = {"trained": "#2a78d6", "frozen_full": "#e34948"}
    panels = [
        ("train_acc", "On-stream train accuracy", False),
        ("train_phi", "On-stream train phi", False),
        ("val_acc", "Validation accuracy (fresh state)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for (col, title, only_valid), ax in zip(panels, axes):
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            if only_valid:
                sub = sub.dropna(subset=[col])
            if sub.empty:
                continue
            for sd, g in sub.groupby("seed"):
                g = g.sort_values("batch")
                ax.plot(g["batch"], g[col], color=colors[cond], alpha=0.25, lw=1)
            mean = sub.groupby("batch")[col].mean()
            ax.plot(mean.index, mean.values, color=colors[cond], lw=2.4,
                    marker="o", ms=3, label=cond)
        ax.set_xlabel("training batch", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Does the representation erode over the stream without weight changes?",
                 fontsize=15)
    out = os.path.join(out_dir, "frozen_stream_erosion.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved plot -> {out}")


def main():
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = args.output_dir or os.path.join(
        REPO_ROOT, "results", "frozen_erosion", args.dataset,
        date.today().strftime("%Y.%m.%d"),
    )
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "frozen_stream_erosion.jsonl")
    open(jsonl_path, "w").close()

    n_batches = max(1, args.train_all // args.train_batch)
    print(f"\nFrozen-vs-trained erosion test")
    print(f"conditions : {CONDITIONS}")
    print(f"seeds      : {seeds}")
    print(f"stream     : {args.train_all} train / batch {args.train_batch} "
          f"= ~{n_batches} batches | val every {args.val_every}")
    print(f"output     : {out_dir}\n")

    all_rows, finals = [], []
    for cond in CONDITIONS:
        for seed in seeds:
            print(f"[{cond} seed {seed}] ...", flush=True)
            records, final = run_condition(cond, seed, args)
            all_rows.extend(records)
            finals.append(final)
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")

    print(f"\n  Wrote {len(all_rows)} per-batch rows -> {jsonl_path}")
    make_plot(all_rows, out_dir)

    import pandas as pd
    fdf = pd.DataFrame(finals)
    print("\n  Final test scores (mean over seeds):")
    print(fdf.groupby("condition")[["test_acc", "test_phi"]].mean().to_string())


if __name__ == "__main__":
    main()
