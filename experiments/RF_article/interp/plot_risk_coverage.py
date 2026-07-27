"""
Risk-coverage figures for one interp run: selective accuracy across the FULL
0->100% coverage sweep, overall and per class.

Reads the confidence-vs-correctness ranking directly, so the curve answers
"if the network only answers its most-confident X% of images, how often is it
right?" for every X, rather than the handful of operating points results.json
records. That is the trade-off curve the article needs, and it is where the
abstention threshold gets chosen.

Two figures:
  risk_coverage.png            overall, all three readouts on one axis
  risk_coverage_per_class.png  small multiples, one panel per class

Plus risk_coverage.csv -- the table view of every operating point, so no number
is reachable only by reading a color off the plot.

Statistics note: selective accuracy at low coverage is estimated from few items,
so every curve carries a 95% Wilson band and the quoted safe limits are the
CONSERVATIVE ones (lower bound clears the target). The 100% line is the single
exception -- no finite sample bounds a proportion at 1.0 -- so it is reported as
"zero errors up to X% coverage" alongside the confidence floor actually achieved
there. Quoting a bare "100% accurate" without that floor is not defensible.

  python experiments/RF_article/interp/plot_risk_coverage.py --run <run_dir>
  python experiments/RF_article/interp/plot_risk_coverage.py --run <run_dir> --readout pool --theme dark
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import (  # noqa: E402
    MIN_KEPT, group_rates, share_probs, softmax_probs, selective_curve,
    safe_coverage, zero_error_point,
)

# Validated categorical slots 1-3 (blue / orange / aqua). This set passes the
# all-pairs CVD and normal-vision gates in both modes; aqua sits below 3:1 on the
# light surface, which is why every series is also direct-labelled and the CSV
# table view ships alongside. Do not add a 4th series without re-validating.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=("#2a78d6", "#eb6834", "#1baf7a"), ref="#c3c2b7"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835",
                  series=("#3987e5", "#d95926", "#199e70"), ref="#383835"),
}
READOUTS = ("learned_readout", "linear_probe", "pool")
LABELS = {"learned_readout": "learned readout", "linear_probe": "linear probe",
          "pool": "uniform pool"}
ACC_TARGETS = (0.99, 0.98, 0.95)
Z = 1.96


# ------------------------------------------------------------------ data access

def load_features(run):
    path = run if run.endswith(".npz") else os.path.join(run, "uncertainty_features.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no uncertainty_features.npz at {path} -- rerun the harness "
                         "or point --run at a run dir that has one")
    return np.load(path), os.path.dirname(os.path.abspath(path))


def readout_probs(f, name):
    """(probabilities, predictions) for one readout, or None if the run lacks it."""
    X, asg = f["X_test"].astype(np.float64), f.get("assignment")
    if name == "pool":
        if asg is None:
            return None
        R = group_rates(X, asg, 10)
        return share_probs(R), R.argmax(1)
    if name == "learned_readout":
        if "score_test" not in f.files:
            return None
        s = f["score_test"].astype(np.float64)
        return softmax_probs(s), s.argmax(1)
    if name == "linear_probe":
        if "probe_test" not in f.files:
            return None
        p = f["probe_test"].astype(np.float64)
        return p, p.argmax(1)
    raise ValueError(name)


def curves_for(f, name, condition="predicted"):
    """Overall curve + one curve per class.

    condition='predicted' slices by what the network SAID, which is the only
    grouping available at inference and therefore the one a deployed threshold
    acts on. condition='true' slices by ground truth, which answers the different
    question "which digits does the network handle well" -- useful for diagnosis,
    not for setting a limit.
    """
    got = readout_probs(f, name)
    if got is None:
        return None
    p, pred = got
    y = f["y_test"].astype(int)
    conf = p.max(1)
    correct = (pred == y).astype(float)
    out = {"overall": selective_curve(conf, correct, Z), "per_class": {}}
    key = pred if condition == "predicted" else y
    for c in range(p.shape[1]):
        m = key == c
        if m.sum() >= 2:
            out["per_class"][c] = selective_curve(conf[m], correct[m], Z)
    return out


# ---------------------------------------------------------------------- drawing

def style_axes(ax, T, xlabel, ylabel):
    ax.set_facecolor(T["surface"])
    ax.grid(True, color=T["grid"], lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=T["muted"], labelsize=8, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=T["ink2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=T["ink2"], fontsize=9)


def pct(x, _=None):
    return f"{100 * x:.0f}%"


def trim(cur, min_frac=0.0):
    """Drop the head of the sweep where the estimate is meaningless.

    Below MIN_KEPT answered items the selective accuracy is 100% or 95% purely by
    luck and swings wildly; plotting it dominates the y-range with noise and
    invites reading a safe limit off a sample of three. The per-class panels pass
    a min_frac as well: with ~900 items per class, MIN_KEPT alone still leaves a
    confidence band wide enough to swamp the panel.
    """
    return cur["n_kept"] >= max(MIN_KEPT, int(min_frac * cur["n"]))


def draw_curve(ax, cur, color, T, label=None, band=True, lw=1.8, z=3,
               min_frac=0.0, alpha=0.13):
    m = trim(cur, min_frac)
    if band:
        ax.fill_between(cur["coverage"][m], cur["lo"][m], cur["hi"][m], color=color,
                        alpha=alpha, lw=0, zorder=z - 1)
    ax.plot(cur["coverage"][m], cur["sel_acc"][m], color=color, lw=lw, label=label,
            solid_capstyle="round", zorder=z)


def acc_ticks(lo, hi, coarse=False):
    """Ticks doubling as the accuracy reference lines, so no floating rule labels."""
    cand = ([0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0] if coarse else
            [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0])
    return [t for t in cand if lo - 1e-9 <= t <= hi + 1e-9]


def main_figure(cur_by_readout, T, tag, out, condition):
    fig = plt.figure(figsize=(11, 6.4), facecolor=T["surface"])
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.085, right=0.80,
                           top=0.855, bottom=0.115)
    ax = fig.add_subplot(gs[0])
    style_axes(ax, T, "coverage — fraction of test images the network answers",
               "selective accuracy on answered images")

    lo_all = []
    for i, name in enumerate(READOUTS):
        cur = cur_by_readout.get(name)
        if cur is None:
            continue
        c = T["series"][i]
        draw_curve(ax, cur["overall"], c, T, label=LABELS[name], z=3 + i)
        o = cur["overall"]
        lo_all.append(o["sel_acc"][trim(o)].min())
        # direct end label -- required relief for the sub-3:1 slot, and it puts
        # the full-coverage accuracy (the headline number) on the figure itself
        ax.annotate(f"{LABELS[name]}  {o['sel_acc'][-1]:.3f}",
                    xy=(1.0, o["sel_acc"][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8.5, color=T["ink2"], annotation_clip=False)

    # the demarcation the article is after: how far we get with zero errors
    prim = cur_by_readout.get("learned_readout") or next(iter(cur_by_readout.values()))
    zc, zn, zlo = zero_error_point(prim["overall"])
    if zc > 0:
        ax.axvline(zc, color=T["ink2"], lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"zero errors up to {zc:.1%} coverage\n"
                    f"(n={zn:,}; 95% lower bound {zlo:.1%})",
                    xy=(zc, 0.06), xycoords=("data", "axes fraction"),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=8.5, color=T["ink2"], va="bottom", ha="left")

    ymin = max(0.0, min(lo_all) - 0.035) if lo_all else 0.0
    ax.set_xlim(0, 1.0)
    ax.set_ylim(ymin, 1.005)
    # accuracy targets ride on the y ticks, so the grid IS the reference rule set
    ax.set_yticks(acc_ticks(ymin, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))

    fig.text(0.085, 0.965, "Accuracy against coverage", fontsize=14,
             color=T["ink"], ha="left", va="top")
    fig.text(0.085, 0.922,
             f"{tag} — 9,000 held-out test images, ranked by each readout's own confidence. "
             "Bands are 95% Wilson intervals; the sweep starts once 20 images are answered.",
             fontsize=8.5, color=T["muted"], ha="left", va="top")
    # legend above the plot: horizontal, out of the curves' way entirely
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
                    fontsize=8.5, frameon=False, handlelength=1.6,
                    columnspacing=1.8, borderpad=0.0)
    for t_ in leg.get_texts():
        t_.set_color(T["ink2"])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return f"{out}.png"


def per_class_figure(cur, T, tag, out, readout, condition):
    classes = sorted(cur["per_class"])
    ncol = 5
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow + 1.0),
                            facecolor=T["surface"], sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.34, wspace=0.12))
    axes = np.atleast_1d(axes).ravel()
    colour = T["series"][0]
    # Per-class sweeps start at 10% coverage (~90 items). With only ~900 items per
    # class the Wilson band below that is wider than the whole panel, so it stops
    # reading as an interval and just fills the axes. Every zero-error limit here
    # lands well above 10%, so nothing the figure is for is lost.
    MF = 0.10
    ymin = min(c["sel_acc"][trim(c, MF)].min() for c in cur["per_class"].values())
    ymin = max(0.0, ymin - 0.02)

    for ax, c in zip(axes, classes):
        cc = cur["per_class"][c]
        style_axes(ax, T, "", "")
        # overall curve as the shared reference so each panel is comparable
        draw_curve(ax, cur["overall"], T["ref"], T, band=False, lw=1.2, z=2,
                   min_frac=MF)
        draw_curve(ax, cc, colour, T, z=4, min_frac=MF, alpha=0.10)
        zc, zn, zlo = zero_error_point(cc)
        if zc > 0:
            ax.axvline(zc, color=T["ink2"], lw=0.9, ls=(0, (4, 3)), zorder=3)
            right = zc > 0.62      # keep the label off the right-hand panel edge
            ax.annotate(f"{zc:.0%}", xy=(zc, 0.06), xycoords=("data", "axes fraction"),
                        xytext=(-4 if right else 4, 0), textcoords="offset points",
                        fontsize=8, color=T["ink2"],
                        ha="right" if right else "left", va="bottom")
        ax.set_title(f"class {c}   n={cc['n']}   base {cc['base_acc']:.1%}",
                     fontsize=9, color=T["ink"], pad=6)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))
    for ax in axes[len(classes):]:
        ax.set_visible(False)

    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(ymin, 1.005)
    axes[0].set_yticks(acc_ticks(ymin, 1.0, coarse=True))
    for i, ax in enumerate(axes[:len(classes)]):
        if i % ncol == 0:
            ax.set_ylabel("selective accuracy", color=T["ink2"], fontsize=8.5)
        if i >= len(classes) - ncol:
            ax.set_xlabel("coverage", color=T["ink2"], fontsize=8.5)

    fig.text(0.02, 0.975, f"Accuracy against coverage, per class — {LABELS[readout]}",
             fontsize=14, color=T["ink"], ha="left", va="top")
    fig.text(0.02, 0.943,
             f"{tag} — each panel sweeps its own threshold within that class "
             f"({condition} class), from 10% coverage up. Grey = overall curve. "
             "Band = 95% Wilson. Dashed = last coverage with zero errors.",
             fontsize=8.5, color=T["muted"], ha="left", va="top")
    fig.subplots_adjust(top=0.90 if nrow > 1 else 0.82, left=0.055,
                        right=0.985, bottom=0.10)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor=T["surface"])
    plt.close(fig)
    return f"{out}.png"


# ------------------------------------------------------------------ table view

def write_table(cur_by_readout, path, condition):
    rows = []
    for name, cur in cur_by_readout.items():
        for scope, c in [("overall", cur["overall"])] + \
                        [(f"class {k}", v) for k, v in sorted(cur["per_class"].items())]:
            zc, zn, zlo = zero_error_point(c)
            row = dict(readout=name, scope=scope, condition=condition, n=c["n"],
                       base_acc=round(c["base_acc"], 4), aurc=round(c["aurc"], 4),
                       zero_error_coverage=round(zc, 4), zero_error_n=zn,
                       zero_error_lower95=round(zlo, 4))
            for t in ACC_TARGETS:
                row[f"cov_at_{t:.2f}_conservative"] = round(safe_coverage(c, t), 4)
                row[f"cov_at_{t:.2f}_empirical"] = round(
                    safe_coverage(c, t, conservative=False), 4)
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def print_summary(rows, readout):
    sel = [r for r in rows if r["readout"] == readout]
    if not sel:
        return
    print(f"\n  {readout} — coverage you can buy at each accuracy target")
    print(f"  {'scope':<10} {'n':>5} {'base':>7} {'>=99%':>7} {'>=98%':>7} "
          f"{'>=95%':>7} {'0-err':>7} {'(n)':>6} {'lo95':>7}")
    for r in sel:
        print(f"  {r['scope']:<10} {r['n']:>5} {r['base_acc']:>7.1%} "
              f"{r['cov_at_0.99_conservative']:>7.1%} {r['cov_at_0.98_conservative']:>7.1%} "
              f"{r['cov_at_0.95_conservative']:>7.1%} {r['zero_error_coverage']:>7.1%} "
              f"{r['zero_error_n']:>6} {r['zero_error_lower95']:>7.1%}")


def make_risk_coverage_plots(run, readout="learned_readout", theme="light",
                             condition="predicted", tag=None):
    f, outdir = load_features(run)
    T = THEMES[theme]
    tag = tag or os.path.basename(outdir)
    cur_by_readout = {}
    for name in READOUTS:
        c = curves_for(f, name, condition)
        if c is not None:
            cur_by_readout[name] = c
    if not cur_by_readout:
        raise SystemExit("no readout scores in the features file")
    if readout not in cur_by_readout:
        readout = next(iter(cur_by_readout))

    p1 = main_figure(cur_by_readout, T, tag, os.path.join(outdir, "risk_coverage"),
                     condition)
    p2 = per_class_figure(cur_by_readout[readout], T, tag,
                          os.path.join(outdir, "risk_coverage_per_class"),
                          readout, condition)
    csv_path = os.path.join(outdir, "risk_coverage.csv")
    rows = write_table(cur_by_readout, csv_path, condition)
    print_summary(rows, readout)
    return p1, p2, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="run dir containing uncertainty_features.npz, or the .npz itself")
    ap.add_argument("--readout", default="learned_readout", choices=READOUTS,
                    help="which readout the per-class panels use")
    ap.add_argument("--theme", default="light", choices=tuple(THEMES))
    ap.add_argument("--condition", default="predicted", choices=("predicted", "true"),
                    help="slice per-class panels by predicted class (what a deployed "
                         "threshold sees) or by true class (diagnostic)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    for p in make_risk_coverage_plots(a.run, a.readout, a.theme, a.condition, a.tag):
        print("saved ->", p)


if __name__ == "__main__":
    main()
