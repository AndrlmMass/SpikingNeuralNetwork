"""How the weight distribution changes across each sleep episode.

Runs one long batch of MNIST through the STDP-SNN and records, for every sleep
episode, the synaptic weight distribution immediately before and immediately
after that episode. No test set is touched: this is a mechanism measurement, not
a performance measurement.

Three metrics per episode and per weight group (excitatory, inhibitory):

  variance   the scale of the distribution -- what the homeostatic pull shrinks.

  entropy    the shape. Shannon entropy of p_i = |w_i| / sum_j |w_j|, normalised
             by log(n) to [0, 1]. Says how evenly total synaptic mass is spread
             over synapses, independently of how large that total is, so it is
             not a restatement of the variance.

  spearman   rank preservation before vs after, over a fixed synapse set.
             Eq. 5's power law is monotone, so decay ALONE gives rho = 1 exactly
             (verified in the unit tests). Departures from 1 therefore isolate
             reordering caused by STDP during sleep, clipping at the bounds, and
             noise-driven spontaneous spiking -- that is, whether sleep
             reorganises which synapses are strong or merely rescales them.

Configuration follows main.py's MNIST-family settings so the run is comparable
to the sweep, with two changes: one batch instead of 15, and weight tracking on.

Usage (from the repo root):
    python src/sleep_distribution.py                      # 9000 samples, 10% sleep
    python src/sleep_distribution.py --samples 3000       # quicker check
    python src/sleep_distribution.py --sleep-rate 0.5
    python src/sleep_distribution.py --no-sleep-stdp      # isolate pure decay

Output: results/sleep_distribution/episodes_<tag>.csv (one row per episode)
        results/sleep_distribution/meta_<tag>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# big_comb prints check marks and warning glyphs. On Windows the default console
# encoding is cp1252, which raises UnicodeEncodeError mid-run and aborts after
# training but before results are written. Force UTF-8 on our streams instead of
# editing those print statements.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from big_comb import snn_sleepy  # noqa: E402

EPI_FIELDS = [
    "episode",
    "batch",
    "t",
    "sleep_iters",
    "exc_spearman",
    "inh_spearman",
]
for _g in ("exc", "inh"):
    for _k in ("n", "mean", "var", "sum_abs", "entropy", "eff_n"):
        EPI_FIELDS += [f"{_g}_{_k}_before", f"{_g}_{_k}_after"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=9000,
                   help="training samples, all in ONE batch (default 9000)")
    p.add_argument("--num-steps", type=int, default=100,
                   help="stimulus duration per sample in ms (default 100, as MNIST runs)")
    p.add_argument("--sleep-rate", type=float, default=0.1,
                   help="sleep ratio per interval (default 0.1, the published optimum)")
    p.add_argument("--check-sleep-interval", type=int, default=35000,
                   help="timesteps between sleep episodes (default 35000)")
    p.add_argument("--dataset", type=str, default="mnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sleep-noise-var", type=float, default=3.0)
    p.add_argument("--sleep-decay-rate", type=float, default=0.9997)
    p.add_argument("--no-sleep-stdp", action="store_true",
                   help="disable STDP during sleep: isolates the pure decay, "
                        "which should give Spearman rho = 1")
    p.add_argument("--no-downscale", action="store_true",
                   help="disable the homeostatic decay, keeping the rest")
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    tag = args.tag or (
        f"{args.dataset}_n{args.samples}_sr{args.sleep_rate:g}"
        f"{'_nostdp' if args.no_sleep_stdp else ''}"
        f"{'_nodown' if args.no_downscale else ''}"
        f"_seed{args.seed}"
    )
    outdir = os.path.join("results", "sleep_distribution")
    os.makedirs(outdir, exist_ok=True)

    print(f"[sleep_distribution] {tag}")
    print(f"  {args.samples} samples x {args.num_steps} ms = "
          f"{args.samples * args.num_steps:,} timesteps, one batch")
    print(f"  sleep every {args.check_sleep_interval:,} steps -> "
          f"~{args.samples * args.num_steps // args.check_sleep_interval} episodes")

    snn = snn_sleepy()

    # One batch: batch size == total samples. Validation/test are set to the
    # minimum the loader accepts and never evaluated.
    snn.prepare_data(
        all_images_train=args.samples,
        batch_image_train=args.samples,
        all_images_test=100,
        batch_image_test=100,
        all_images_val=100,
        batch_image_val=100,
        num_steps=args.num_steps,
        add_breaks=False,
        force_recreate=False,
        noisy_data=False,
        gain=1.0,
        noise_level=0.0,
        audioMNIST=False,
        imageMNIST=True,
        create_data=False,
        plot_spectrograms=False,
        image_dataset=args.dataset,
    )

    snn.prepare_network(
        plot_weights=False,
        w_dense_ee=0.15,
        w_dense_se=0.1,
        w_dense_ei=0.2,
        w_dense_ie=0.25,
        se_weights=0.15,
        ee_weights=0.3,
        ei_weights=0.3,
        ie_weights=-0.3,
        create_network=False,
    )

    t0 = time.time()
    snn.train_network(
        train_weights=True,
        noisy_potential=True,
        compare_decay_rates=False,
        check_sleep_interval=args.check_sleep_interval,
        weight_decay_rate_exc=[args.sleep_decay_rate],
        weight_decay_rate_inh=[args.sleep_decay_rate],
        samples=10,
        force_train=True,
        plot_spikes_train=False,
        plot_weights=False,
        plot_epoch_performance=False,
        # This is the flag that turns on track_weights inside train.train_network,
        # which is what populates the per-episode metrics.
        plot_weights_per_epoch=True,
        plot_spikes_per_epoch=False,
        weight_track_samples=32,
        weight_track_interval=0,
        weight_track_sleep_interval=0,
        sleep_synchronized=False,
        plot_top_response_test=False,
        plot_top_response_train=False,
        plot_tsne_during_training=False,
        plot_spectrograms=False,
        use_validation_data=False,
        var_noise=args.sleep_noise_var,
        max_weight_exc=25,
        min_weight_inh=-25,
        sleep=True,
        tau_syn=30,
        narrow_top=0.2,
        A_minus=0.5,
        A_plus=0.5,
        tau_LTD=10,
        tau_LTP=10,
        learning_rate_exc=0.0005,
        learning_rate_inh=0.0005,
        accuracy_method="pca_lr",
        test_only=False,
        use_QDA=False,
        early_stopping=False,
        sleep_ratio=args.sleep_rate,
        sleep_max_iters=int(1e6),
        on_timeout="none",
        normalize_weights=False,
        sleep_termination="band",
        sleep_downscale=(not args.no_downscale),
        sleep_noise=True,
        sleep_stdp=(not args.no_sleep_stdp),
        sleep_suppress_input=True,
    )
    elapsed = time.time() - t0

    tracking = getattr(snn, "weight_tracking_sleep", None)
    episodes = (tracking or {}).get("episodes", [])
    if not episodes:
        print("ERROR: no sleep episodes were recorded. Either sleep never "
              "triggered (check --check-sleep-interval against total timesteps) "
              "or weight tracking did not initialise.")
        return 1

    csv_path = os.path.join(outdir, f"episodes_{tag}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(EPI_FIELDS) + "\n")
        for r in episodes:
            fh.write(",".join(
                "" if r.get(k) is None else
                (f"{r[k]:.10g}" if isinstance(r.get(k), float) else str(r.get(k, "")))
                for k in EPI_FIELDS
            ) + "\n")

    meta = {
        "tag": tag,
        "samples": args.samples,
        "num_steps": args.num_steps,
        "total_timesteps": args.samples * args.num_steps,
        "sleep_rate": args.sleep_rate,
        "check_sleep_interval": args.check_sleep_interval,
        "dataset": args.dataset,
        "seed": args.seed,
        "sleep_stdp": not args.no_sleep_stdp,
        "sleep_downscale": not args.no_downscale,
        "sleep_noise_var": args.sleep_noise_var,
        "sleep_decay_rate": args.sleep_decay_rate,
        "n_episodes": len(episodes),
        "elapsed_s": elapsed,
    }
    with open(os.path.join(outdir, f"meta_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # --- console summary: the paired before/after comparison ---
    def col(k):
        return np.array([r[k] for r in episodes if r.get(k) is not None], dtype=float)

    print(f"\n{len(episodes)} sleep episodes in {elapsed/60:.1f} min")
    bar = "=" * 72
    print(f"\n{bar}\nPaired before -> after across sleep episodes (medians)\n{bar}")
    print(f"{'group':5s} {'metric':10s} {'before':>12s} {'after':>12s} {'ratio':>9s}")
    print("-" * 72)
    for g in ("exc", "inh"):
        for m in ("var", "entropy", "eff_n", "sum_abs"):
            b, a = col(f"{g}_{m}_before"), col(f"{g}_{m}_after")
            if b.size == 0:
                continue
            mb, ma = float(np.median(b)), float(np.median(a))
            ratio = ma / mb if mb != 0 else float("nan")
            print(f"{g:5s} {m:10s} {mb:12.6g} {ma:12.6g} {ratio:9.4f}")
    print("-" * 72)
    for g in ("exc", "inh"):
        rho = col(f"{g}_spearman")
        if rho.size:
            print(f"{g:5s} rank preservation (Spearman rho): "
                  f"median {np.median(rho):.6f}, min {rho.min():.6f}, max {rho.max():.6f}")
    print(bar)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
