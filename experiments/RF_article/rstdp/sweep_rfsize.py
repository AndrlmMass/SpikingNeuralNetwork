"""
RF-size sweep for the tiled reward-STDP run.

Question: the tiled init wires each SE neuron to ~1/3 of the image (sigma_se=3.0),
so reward-STDP paints whole-digit "quintessential" templates and the spatial tiling
is cosmetic. Does forcing LOCAL receptive fields (smaller structural footprint) — or
HETEROGENEOUS sizes — change accuracy and make the RFs interpretable parts?

Runs a handful of named configs at the tuned hyperparameters, holding everything
except the RF size fixed, then reports the readout/pool accuracy + coverage and
stitches every config's per-class RF grid into one comparison image.

  python experiments/RF_article/rstdp/sweep_rfsize.py --train-all 6000

Structural footprint by sigma_se (measured, 28px image):
  3.0 -> ~257 syn/neuron, ~19px radius (near-global; current default)
  2.0 -> ~113 syn/neuron, ~17px radius
  1.5 -> ~64  syn/neuron, ~6px  radius (local patch)
  1.0 -> ~30  syn/neuron, ~3.6px radius (tight patch)
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

# name -> (sigma_se, sigma_se_lognormal). sigma_se=0 keeps the grouped default (3.0).
CONFIGS = [
    ("baseline_3.0", 3.0, 0.0),   # current: near-global, whole-digit templates
    ("local_1.5",    1.5, 0.0),   # local patch detectors
    ("tight_1.0",    1.0, 0.0),   # tight patch detectors
    ("hetero_2.0ln", 2.0, 1.0),   # heterogeneous sizes (lognormal spread)
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=5e-6)
    p.add_argument("--readout-lr", type=float, default=0.1)
    p.add_argument("--peak-ei", type=float, default=50.0)
    p.add_argument("--peak-ie", type=float, default=-2.0)
    p.add_argument("--train-all", type=int, default=6000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(name, sigma, sigma_ln, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, HARNESS, "--tag", name, "--prior", "isotropic",
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(a.reward_lr), "--readout-lr", str(a.readout_lr),
           "--peak-ei", str(a.peak_ei), "--peak-ie", str(a.peak_ie),
           "--sigma-se", str(sigma), "--sigma-se-lognormal", str(sigma_ln),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1",
               KMP_DUPLICATE_LIB_OK="TRUE")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    ok = pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json"))
    return "OK" if ok else f"FAIL(rc={pr.returncode})"


def final(out, key):
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    vals = [t.get(key, float("nan")) for t in tr[-2:]]  # mean of last 2 checkpoints
    return float(np.nanmean(vals)) if vals else float("nan")


def main():
    a = parse_args()
    run_id = a.run_id or f"rfsize_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_rfsize", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"RF-size sweep -> {run_dir}")
    print(f"  tuned config: reward_lr={a.reward_lr:g} readout_lr={a.readout_lr:g} "
          f"peak_ei={a.peak_ei:g}  train_all={a.train_all}\n")
    t0 = time.time()
    for name, sigma, sigma_ln in CONFIGS:
        t1 = time.time()
        status = run_one(name, sigma, sigma_ln, os.path.join(run_dir, name), a)
        print(f"  {name:<14} sigma={sigma:<4g} ln={sigma_ln:<4g} {status:<12} "
              f"{(time.time()-t1)/60:5.1f} min", flush=True)

    # results table
    rows = []
    for name, sigma, sigma_ln in CONFIGS:
        out = os.path.join(run_dir, name)
        rows.append((name, sigma, sigma_ln,
                     final(out, "readout_learned_acc"), final(out, "softmax_acc"),
                     final(out, "dead_frac"), final(out, "rf_diversity"),
                     final(out, "fixed_acc")))
    print(f"\n{'config':<14} {'sigma':>6} {'learned':>9} {'uniform':>9} {'dead':>6} "
          f"{'rf_div':>7} {'LR_ceil':>8}")
    for name, sigma, sln, le, un, de, rd, fx in rows:
        tag = f"{sigma:g}" + (f"+ln{sln:g}" if sln else "")
        print(f"{name:<14} {tag:>6} {le:>9.3f} {un:>9.3f} {de:>6.2f} {rd:>7.3f} {fx:>8.3f}")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump([dict(config=n, sigma=s, sigma_ln=sl, learned=le, uniform=un,
                        dead=de, rf_div=rd, lr_ceil=fx)
                   for n, s, sl, le, un, de, rd, fx in rows], f, indent=2)

    # side-by-side per-class RF grids (one column per config)
    imgs = [(n, os.path.join(run_dir, n, "weights", "group_rfs.png")) for n, _, _ in
            [(c[0], c[1], c[2]) for c in CONFIGS]]
    imgs = [(n, p) for n, p in imgs if os.path.isfile(p)]
    if imgs:
        fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 7))
        if len(imgs) == 1:
            axes = [axes]
        for ax, (n, p) in zip(axes, imgs):
            ax.imshow(mpimg.imread(p)); ax.axis("off"); ax.set_title(n, fontsize=11)
        fig.suptitle("Per-class RFs vs structural RF size (sigma_se)", fontsize=13)
        cmp = os.path.join(run_dir, "rf_compare.png")
        fig.savefig(cmp, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"\nRF comparison -> {cmp}")

    # accuracy-vs-size bar
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows)); w = 0.35
    ax.bar(x - w / 2, [r[3] for r in rows], w, color="#9467bd", label="readout (learned)")
    ax.bar(x + w / 2, [r[4] for r in rows], w, color="#d62728", label="readout (uniform pool)")
    ax.plot(x, [r[7] for r in rows], "k--o", label="linear-classifier ceiling")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, axis="y")
    ax.legend(); ax.set_title("Accuracy vs RF size")
    bar = os.path.join(run_dir, "acc_vs_rfsize.png")
    fig.savefig(bar, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Accuracy chart -> {bar}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
