"""
RF geometry sweep: does Domantas' tuned prior actually build a better representation?

His run_rstdp.py config reports strong per-class pooled accuracy, but it changes THREE
things at once relative to our harness default:

  1. oriented, THIN RFs      sigma_x=3.0 along the bar, 1.2 across (gamma=0.4),
                             vs our near-isotropic default
  2. concentrated centers    tiled_center_margin=4, so RF centers tile the central
                             region instead of the full 28x28. MNIST digits are
                             centered, so full tiling puts the outer tiles on
                             permanently blank border pixels and those neurons die —
                             a direct attack on our dead_frac of 0.48-0.60.
  3. no WTA competition      ablate_ie=True, zeroing the I->E block entirely.

Changed together we cannot tell which one pays. This sweep factorizes them, and scores
every cell with the REPLACEMENT metrics rather than class_selectivity, which we showed
is confounded by firing rate (it correlated +0.91 with dead fraction across the
inhibition sweep, and its correlation with linear decodability vanished once dead
fraction was partialled out).

The metrics that decide it:
  eta2         per-neuron class information, noise-aware (replaces selectivity)
  corr_within  mean |pairwise correlation| inside a class group = REDUNDANCY. The
               real target: our current network sits at 0.55, i.e. neurons meant to
               cover a class from different angles carry half the same signal.
  pr           effective dimensionality. Needs >= 9 to separate 10 classes at all;
               we currently measure ~4.2, i.e. under half the required rank.

  python experiments/RF_article/rstdp/sweep_rfgeom.py --train-all 6000 --jobs 6
"""
import argparse, datetime as dt, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

# name -> (prior, rf_length, rf_thickness, center_margin, ablate_ie)
# Each row changes ONE thing from `base`, except `dom` which is his full config.
CELLS = {
    "base":         ("isotropic", 3.0, 1.2, 0.0, False),   # our harness default
    "margin":       ("isotropic", 3.0, 1.2, 4.0, False),   # + concentrated centers
    "thin":         ("oriented",  3.0, 1.2, 0.0, False),   # + thin oriented bars
    "thin_margin":  ("oriented",  3.0, 1.2, 4.0, False),   # + both
    "dom":          ("oriented",  3.0, 1.2, 4.0, True),    # + inhibition off = his config
    "no_wta":       ("isotropic", 3.0, 1.2, 0.0, True),    # inhibition off alone
}

METRICS = [
    ("readout_learned_acc", "learned"),
    ("softmax_acc",         "pool"),
    ("refit_acc",           "LRceil"),
    ("perplexity",          "perplex"),
    ("eta2",                "eta2"),
    ("corr_within",         "corr_in"),
    ("corr_all",            "corr_all"),
    ("pr",                  "PR"),
    ("n_active",            "n_act"),
    ("dead_frac",           "dead"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=2e-5)
    p.add_argument("--readout-lr", type=float, default=0.1)
    p.add_argument("--dense-readout", action="store_true", default=True)
    p.add_argument("--peak-ei", type=float, default=20.0)
    p.add_argument("--peak-ie", type=float, default=-2.0)
    p.add_argument("--cells", nargs="+", default=list(CELLS),
                   help="subset of: " + " ".join(CELLS))
    p.add_argument("--train-all", type=int, default=6000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def is_complete(out):
    """True only if the cell ran to completion — the harness rewrites results.json at
    every checkpoint, so mere existence means "started"."""
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return False
    try:
        return "test_acc" in json.load(open(rp))
    except Exception:
        return False


def run_one(name, out, a):
    if is_complete(out):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    prior, length, thick, margin, ablate = CELLS[name]
    cmd = [sys.executable, HARNESS, "--tag", name, "--prior", prior,
           "--rule", "reward", "--tiled", "--seed", str(a.seed),
           "--reward-lr", str(a.reward_lr), "--readout-lr", str(a.readout_lr),
           "--peak-ei", str(a.peak_ei), "--peak-ie", str(a.peak_ie),
           "--rf-length", str(length), "--rf-thickness", str(thick),
           "--center-margin", str(margin), "--no-plots",
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out]
    if a.dense_readout:
        cmd.append("--dense-readout")
    if ablate:
        cmd.append("--ablate-ie")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1",
               KMP_DUPLICATE_LIB_OK="TRUE")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return "OK" if pr.returncode == 0 and is_complete(out) else f"FAIL(rc={pr.returncode})"


def final(out, key):
    rp = os.path.join(out, "results.json")
    if not os.path.isfile(rp):
        return float("nan")
    tr = json.load(open(rp)).get("trajectory", [])
    v = [t[key] for t in tr[-2:] if t.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def main():
    a = parse_args()
    run_id = a.run_id or f"rfgeom_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_rfgeom", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"RF geometry sweep -> {run_dir}")
    print(f"  cells: {a.cells}")
    print(f"  reward_lr={a.reward_lr:g} readout_lr={a.readout_lr:g} dense={a.dense_readout} "
          f"peak_ei={a.peak_ei:g} peak_ie={a.peak_ie:g} train={a.train_all} jobs={a.jobs}\n")
    t0 = time.time()

    def work(name):
        t1 = time.time()
        st = run_one(name, os.path.join(run_dir, name), a)
        print(f"  {name:<14} {st:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    if a.jobs > 1:
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(work, a.cells))
    else:
        for c in a.cells:
            work(c)

    print(f"\n{'cell':<14} {'prior':<10} {'margin':>7} {'wta':>5}"
          + "".join(f" {n:>9}" for _, n in METRICS))
    rows = []
    for name in a.cells:
        out = os.path.join(run_dir, name)
        prior, length, thick, margin, ablate = CELLS[name]
        rec = dict(cell=name, prior=prior, rf_length=length, rf_thickness=thick,
                   center_margin=margin, wta=not ablate)
        line = f"{name:<14} {prior:<10} {margin:>7g} {('off' if ablate else 'on'):>5}"
        for key, label in METRICS:
            v = final(out, key); rec[label] = v
            line += f" {v:>9.3f}" if np.isfinite(v) else f" {'nan':>9}"
        print(line)
        rows.append(rec)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # Two panels that decide it: is the representation less redundant, and is the
    # pooled (network-native) readout better?
    names = [r["cell"] for r in rows]
    xs = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, lab, good) in zip(axes, [
            ("corr_in", "within-group correlation (lower = less redundant)", "down"),
            ("PR", "participation ratio (needs >= 9 for 10 classes)", "up"),
            ("learned", "readout accuracy", "up")]):
        vals = [r[key] for r in rows]
        ax.bar(xs, vals, color=["#7F77DD" if n != "dom" else "#1D9E75" for n in names])
        ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3, axis="y")
        if key == "PR":
            ax.axhline(9, ls="--", lw=1.2, color="k")
            ax.text(0.02, 9.2, "rank needed", fontsize=8, transform=ax.get_yaxis_transform())
    fig.suptitle("RF geometry: factorizing Domantas' config (thin oriented / centered tiles / no WTA)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(run_dir, "rfgeom.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nPlots -> {out}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
