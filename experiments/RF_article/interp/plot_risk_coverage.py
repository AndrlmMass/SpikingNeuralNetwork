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

Selective accuracy at low coverage is estimated from few items, so every curve
carries a 95% Wilson band and the CSV's quoted limits are the conservative ones
(lower bound clears the target). These are descriptive: they characterise this
test set rather than promising anything about future data, and the manuscript
should word them that way.

The figures carry no title, subtitle or legend -- each curve is named where it
ends, and the caption belongs in the manuscript text, not burned into the image.

  python experiments/RF_article/interp/plot_risk_coverage.py --run <run_dir>
  python experiments/RF_article/interp/plot_risk_coverage.py --run <run_dir> --palette ink
  python experiments/RF_article/interp/plot_risk_coverage.py --run <run_dir> --readout pool --theme dark
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import (  # noqa: E402
    MIN_KEPT, group_rates, share_probs, softmax_probs, selective_curve,
    safe_coverage, zero_error_point,
)

# Series palettes, in READOUTS order (learned / probe / pool). Measured on a WHITE
# print surface, over all pairs, with a colour-vision-deficiency simulation:
#
#   project  THE PROJECT DEFAULT (Andreas' pick, 2026-07-25). Muted rose / slate /
#            sage. NOTE: rose and sage separate by only dE 3.5 under deuteranopia,
#            so red-green colourblind readers (~8% of men) see the learned-readout
#            and uniform-pool curves as nearly the same colour. Acceptable here
#            ONLY because every curve carries a direct end label, so identity never
#            rests on hue. Do not reuse it on a chart without direct labels, and do
#            not extend it to a 4th series.
#   muted    Paul Tol muted. Same rose/blue/green family, nearly the same feel, but
#            actually separable: CVD dE 8.7, normal-vision dE 20.0, all >= 3:1
#            contrast. The drop-in swap if the CVD issue ever matters.
#   okabe    Okabe-Ito, the colourblind-safe standard in biology. Passes outright
#            (CVD dE 11.0). Louder than the project default.
#   tol      Paul Tol high-contrast. Separation to spare (dE 16.2); its yellow is
#            2.13:1 on white, so it leans on the direct labels.
#   ink      Emphasis, not identity: the network's readout near-black, the controls
#            subordinate. Fails the chroma floor by design -- that check assumes hue
#            carries identity, and here lightness does (worst pair dE 20.9). The
#            only option that survives greyscale printing.
PALETTES = {
    "project": ("#DB8383", "#6D8BA9", "#6FA885"),
    "muted":   ("#CC6677", "#4477AA", "#228833"),
    "okabe":   ("#0072B2", "#D55E00", "#009E73"),
    "tol":     ("#004488", "#BB5566", "#DDAA33"),
    "ink":     ("#1B1B1B", "#A63603", "#9A9A9A"),
}
DEFAULT_PALETTE = "project"
THEMES = {
    "light": dict(surface="#ffffff", ink="#1a1a1a", ink2="#333333", muted="#666666",
                  grid="#e6e6e6", axis="#444444", ref="#bbbbbb"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#e0e0e0", muted="#a0a0a0",
                  grid="#2c2c2a", axis="#888888", ref="#555555"),
}
# Type scale. The figure is drawn large and scaled down into a column, so these
# are sized to stay legible at ~half size in print.
FS_LABEL, FS_TICK, FS_SERIES, FS_PANEL = 24, 18, 18, 17
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

def style_axes(ax, T, xlabel, ylabel, fs_label=FS_LABEL, fs_tick=FS_TICK, grid=True):
    ax.set_facecolor(T["surface"])
    if grid:
        ax.grid(True, axis="y", color=T["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"]); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=T["ink2"], labelsize=fs_tick, length=5, width=1.0,
                   direction="out")
    if xlabel:
        ax.set_xlabel(xlabel, color=T["ink"], fontsize=fs_label, labelpad=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=T["ink"], fontsize=fs_label, labelpad=0)


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
               min_frac=0.0, alpha=0.13, ls="-"):
    m = trim(cur, min_frac)
    if band:
        ax.fill_between(cur["coverage"][m], cur["lo"][m], cur["hi"][m], color=color,
                        alpha=alpha, lw=0, zorder=z - 1)
    ax.plot(cur["coverage"][m], cur["sel_acc"][m], color=color, lw=lw, label=label,
            ls=ls, solid_capstyle="round", zorder=z)


def acc_ticks(lo, hi, step=0.05):
    """An even ladder of accuracy ticks; the gridlines are the reference rules.

    Uneven ticks (…95, 98, 99, 100) crowd illegibly wherever the curves live, which
    is exactly the top of the range, so keep the spacing regular and let the caption
    carry any specific operating point.
    """
    n0 = int(np.floor(lo / step + 1e-9))
    return [round(k * step, 4) for k in range(n0, int(round(hi / step)) + 1)
            if lo - 1e-9 <= k * step <= hi + 1e-9]


def main_figure(cur_by_readout, T, series, out):
    """Bare figure: no title, no subtitle, no legend box.

    Each curve is named where it ends, which is both the legend and the headline
    number. Caption, run tag and method notes belong in the manuscript text, not
    burned into the image.
    """
    fig = plt.figure(figsize=(10.5, 6.2), facecolor=T["surface"])
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.115, right=0.735,
                           top=0.965, bottom=0.165)
    ax = fig.add_subplot(gs[0])
    style_axes(ax, T, "Coverage", "Selective accuracy")

    lo_all, ends = [], []
    for i, name in enumerate(READOUTS):
        cur = cur_by_readout.get(name)
        if cur is None:
            continue
        draw_curve(ax, cur["overall"], series[i], T, z=3 + i, lw=2.2)
        o = cur["overall"]
        lo_all.append(o["sel_acc"][trim(o)].min())
        ends.append((o["sel_acc"][-1], LABELS[name], series[i]))

    ymin = max(0.0, min(lo_all) - 0.035) if lo_all else 0.0
    ax.set_xlim(0, 1.0)
    ax.set_ylim(ymin, 1.005)
    ax.set_yticks(acc_ticks(ymin, 1.0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(pct))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))

    # Name each curve where it ends. Nudge apart only if two finish close enough
    # that two lines of text would overlap.
    ends.sort(key=lambda e: -e[0])
    gap = 0.115 * (1.005 - ymin)
    ys = []
    for val, _, _ in ends:
        y = val if not ys else min(val, ys[-1] - gap)
        ys.append(y)
    for (val, lbl, col), y in zip(ends, ys):
        ax.annotate(f"{lbl}\n{val:.1%}", xy=(1.02, y), xycoords="data",
                    va="center", ha="left", fontsize=FS_SERIES, color=col,
                    linespacing=1.35, annotation_clip=False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=T["surface"])
    plt.close(fig)
    return f"{out}.png"


def per_class_figure(cur, T, series, out):
    classes = sorted(cur["per_class"])
    ncol = 5
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow),
                            facecolor=T["surface"], sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.28, wspace=0.22))
    axes = np.atleast_1d(axes).ravel()
    colour = series[0]
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
        # The SAME readout's overall curve, repeated in every panel so a class can
        # be read against the aggregate. Dashed and pale so it cannot be mistaken
        # for a second data series -- and so the solid curve's confidence band
        # visibly belongs to the solid curve.
        draw_curve(ax, cur["overall"], T["ref"], T, band=False, lw=1.4, z=2,
                   min_frac=MF, ls=(0, (5, 3)))
        draw_curve(ax, cc, colour, T, z=4, lw=2.4, min_frac=MF, alpha=0.13)
        ax.set_title(f"class {c}", fontsize=FS_PANEL + 2, color=T["ink"], pad=8)
        ax.annotate(f"n={cc['n']}   {cc['base_acc']:.1%}", xy=(0.04, 0.06),
                    xycoords="axes fraction", fontsize=FS_PANEL - 3,
                    color=T["muted"], ha="left", va="bottom")
        # Bare numbers on x, with the unit moved into the axis label. At the main
        # figure's tick size a "100%" on one panel and a "0%" on the next collide
        # in the gutter; dropping the sign is the cheapest character to lose.
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(pct))
    for ax in axes[len(classes):]:
        ax.set_visible(False)

    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(ymin, 1.005)
    axes[0].set_yticks(acc_ticks(ymin, 1.0, step=0.05))
    axes[0].set_xticks([0, 0.5, 1.0])
    for ax in axes[:len(classes)]:
        ax.tick_params(labelsize=FS_TICK)     # match the main figure
    # one axis label for the whole grid rather than ten repetitions
    fig.supxlabel("Coverage (%)", color=T["ink"], fontsize=FS_LABEL, y=0.105)
    fig.supylabel("Selective accuracy", color=T["ink"], fontsize=FS_LABEL, x=0.013)

    # One legend for the grid. The swatch is drawn a little more opaque than the
    # band itself -- at legend-handle size the true alpha is invisible.
    handles = [
        Line2D([], [], color=colour, lw=2.4, label="this class"),
        Patch(facecolor=colour, alpha=0.22, lw=0, label="95% interval"),
        Line2D([], [], color=T["ref"], lw=1.4, ls=(0, (5, 3)),
               label="all classes"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=FS_TICK, handlelength=2.4, columnspacing=2.6,
                     bbox_to_anchor=(0.5, 0.004))
    for t_ in leg.get_texts():
        t_.set_color(T["ink2"])

    fig.subplots_adjust(top=0.94, left=0.098, right=0.976, bottom=0.20)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=220, facecolor=T["surface"])
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
                             condition="predicted", palette=DEFAULT_PALETTE,
                             suffix=""):
    f, outdir = load_features(run)
    T = THEMES[theme]
    series = PALETTES[palette]
    cur_by_readout = {}
    for name in READOUTS:
        c = curves_for(f, name, condition)
        if c is not None:
            cur_by_readout[name] = c
    if not cur_by_readout:
        raise SystemExit("no readout scores in the features file")
    if readout not in cur_by_readout:
        readout = next(iter(cur_by_readout))

    p1 = main_figure(cur_by_readout, T, series,
                     os.path.join(outdir, f"risk_coverage{suffix}"))
    p2 = per_class_figure(cur_by_readout[readout], T, series,
                          os.path.join(outdir, f"risk_coverage_per_class{suffix}"))
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
    ap.add_argument("--palette", default=DEFAULT_PALETTE, choices=tuple(PALETTES))
    ap.add_argument("--suffix", default="", help="appended to the output filenames")
    ap.add_argument("--condition", default="predicted", choices=("predicted", "true"),
                    help="slice per-class panels by predicted class (what a deployed "
                         "threshold sees) or by true class (diagnostic)")
    a = ap.parse_args()
    for p in make_risk_coverage_plots(a.run, a.readout, a.theme, a.condition,
                                      a.palette, a.suffix):
        print("saved ->", p)


if __name__ == "__main__":
    main()
