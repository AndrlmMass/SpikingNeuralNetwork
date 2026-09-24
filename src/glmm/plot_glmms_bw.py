"""Black-and-white figures for the three revision studies.

Matches the RF-article figure conventions (plot_readout_compare.py,
plot_risk_coverage.py): white surface, recessive y-grid, top/right spines
removed, dpi 220, Agg backend.

Monochrome encoding
-------------------
Print figures cannot rely on hue, so series identity is carried twice: by a
grey step with real lightness separation (>= 0.17 apart, so adjacent steps stay
distinguishable in print and under any CVD) AND by a hatch pattern. Every
figure with more than one series therefore has a legend, and values are labelled
directly where there are few enough to read. Text stays in the ink tokens; the
grey of a bar never carries meaning that the label does not repeat.

    python src/glmm/plot_glmms_bw.py

Inputs:  results/glmm/*.csv (written by fit_glmms.R), results/*/*_summary.csv
Outputs: figures/sweep_ratio_BW.pdf
         figures/baselines_methods_BW.pdf
         figures/ablation_components_BW.pdf
"""
import csv
import math
import os
import collections
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GL = os.path.join(REPO, "results", "glmm")
FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

# --- house style (RF article) -------------------------------------------------
T = dict(surface="#ffffff", ink="#1a1a1a", ink2="#333333", muted="#666666",
         grid="#e6e6e6", axis="#444444")
FS_LABEL, FS_TICK, FS_SERIES = 25, 20, 19

# Grey ramp, light -> dark, steps >= 0.17 apart in lightness so adjacent pairs
# remain separable in greyscale print. Paired with hatches for redundancy.
GREY = ["#e8e8e8", "#c0c0c0", "#949494", "#666666", "#3a3a3a"]
HATCH = ["", "///", "...", "xxx", "\\\\\\"]

# --- which individual runs to draw ------------------------------------------
# Drawing all 5 seeds at every condition is cluttered; drawing none hides the
# bimodality that is the main qualitative finding. Rules:
#
#   "collapse" (default) -- draw runs below COLLAPSE_AT. Accuracy in these
#       studies is bimodal with an empty gap (84 runs in 0.067-0.199, 136 in
#       0.257-0.802, none between), so this threshold separates runs whose
#       network collapsed from runs that learned. It is the distinction the
#       figures exist to show.
#   "tukey" -- the boxplot flier rule, outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
#       Included for completeness, but it is a poor fit at n = 5: Q1 and Q3 are
#       just the 2nd and 4th values, so it flags 15% of runs, including both
#       extremes of tight clusters (MNIST at 10%: 0.734-0.788, spread 0.054,
#       both ends flagged), while returning NO fliers for genuinely bimodal
#       cells (MNIST at 80%: 0.082/0.124/0.130/0.723/0.737) because the IQR
#       spans the gap.
#   "all" -- every run, the previous behaviour.
#   "none" (current) -- no individual runs; the box and the model interval
#       carry the distribution on their own.
OUTLIER_RULE = "none"
COLLAPSE_AT = 0.20
RUN_LABEL = {"none": None,
             "collapse": f"collapsed run (< {COLLAPSE_AT:g})",
             "tukey": "outlying run (1.5 x IQR)",
             "all": "individual run"}[OUTLIER_RULE]


def to_draw(vals, rule=None):
    """The subset of a condition's runs to plot individually. `rule` overrides
    OUTLIER_RULE for one figure -- the sweep panels show every run, the
    baselines boxes show none."""
    rule = OUTLIER_RULE if rule is None else rule
    if rule == "none":
        return []
    if rule == "all":
        return list(vals)
    if rule == "collapse":
        return [v for v in vals if v < COLLAPSE_AT]
    v = sorted(vals)
    n = len(v)
    if n < 4:
        return []
    def pct(q):                      # linear interpolation, as numpy does
        h = (n - 1) * q
        f = int(h)
        return v[f] + (h - f) * (v[min(f + 1, n - 1)] - v[f])
    q1, q3 = pct(0.25), pct(0.75)
    iqr = q3 - q1
    return [x for x in vals if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]



def style(ax, ygrid=True, xgrid=False):
    ax.set_facecolor(T["surface"])
    if ygrid:
        ax.grid(True, axis="y", color=T["grid"], lw=0.7, zorder=0)
    if xgrid:
        ax.grid(True, axis="x", color=T["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=T["ink2"], labelsize=FS_TICK, length=5, width=1.0)


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p + ".pdf", dpi=220, facecolor=T["surface"], bbox_inches="tight")
    fig.savefig(p + ".png", dpi=220, facecolor=T["surface"], bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.relpath(p, REPO) + ".{pdf,png}")


# =============================================================================
# Figure 1 — sleep-ratio sweep
# =============================================================================
def fig_sweep():
    """Four-panel small multiple, one panel per dataset, mirroring the layout of
    the submitted manuscript's main figure: observed per-ratio means as open
    markers, model-predicted means as filled markers, both with 95% intervals
    and dotted connectors.

    Predictions come from the ratio x dataset interaction fit, not the
    main-effect fit. The main-effect model's dataset random intercept is
    ~0, so it would draw the identical curve in all four panels and visibly
    miss the observed points; the interaction is preferred at
    chi2(32) = 159.1, p < 2.2e-16. The pooled main-effect fit remains the one
    reported in the table.

    The published version of this figure carried a second series for the
    surrogate-gradient model. That model is dropped from the revision (its
    results are not reproducible), so only the STDP network is shown.
    """
    pred = collections.defaultdict(dict)
    for r in rd(os.path.join(GL, "sweep_predictions_by_dataset.csv")):
        pred[r["dataset"]][float(r["sleep_rate"])] = (
            float(r["fit"]), float(r["lo"]), float(r["hi"]))

    obs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(os.path.join(REPO, "results", "sweep", "sweep_summary.csv")):
        if r["test_accuracy"]:
            obs[r["dataset"]][float(r["sleep_rate"])].append(float(r["test_accuracy"]))

    panels = [("mnist", "MNIST"), ("kmnist", "KMNIST"),
              ("fmnist", "Fashion-MNIST"), ("notmnist", "NotMNIST")]
    rates = sorted(pred[panels[0][0]])

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.6), sharey=True,
                             facecolor=T["surface"])
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        x = [r * 100 for r in rates]

        # Observed runs: the five seeds at each sleep duration, plotted on
        # the tick. No box and no horizontal jitter -- a box over five points
        # that are all drawn anyway restated them as an empty rectangle, and
        # spreading them sideways read as a trend within one condition.
        for r, xv in zip(rates, x):
            v = to_draw(obs[key][r], "all")
            if not v:
                continue
            ax.plot([xv] * len(v), v, "o", mfc="none", mec=T["muted"], ms=2.6,
                    mew=0.7, ls="none", zorder=2)

        # Model-predicted mean with its 95% interval -- the only interval here.
        pm = [pred[key][r][0] for r in rates]
        plo = [pred[key][r][1] for r in rates]
        phi = [pred[key][r][2] for r in rates]
        ax.errorbar(x, pm, yerr=[[m - l for m, l in zip(pm, plo)],
                                 [h - m for m, h in zip(pm, phi)]],
                    fmt="v", mfc=T["ink"], mec=T["ink"], ms=7,
                    ecolor=T["ink"], elinewidth=1.1, capsize=3, capthick=1.1,
                    ls=":", lw=1.1, color=T["ink"], zorder=7,
                    label="predicted mean, 95% CI")

        ax.set_title(title, fontsize=FS_LABEL - 3, color=T["ink"], pad=8)
        ax.set_xlabel("sleep duration (%)", fontsize=FS_LABEL - 5,
                      color=T["ink2"])
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-6, 108)
        ax.set_ylim(0, 1.0)

    # "predicted", not "test": every marker in this figure is a model-fitted
    # mean from sweep_predictions_by_dataset.csv, not an observed value. The
    # baselines and ablation A/C figures plot raw runs and stay "test accuracy".
    axes[0].set_ylabel("predicted test accuracy", fontsize=FS_LABEL - 3,
                       color=T["ink2"])
    # No legend: only one series is drawn, and the y-axis already names it.
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.19, wspace=0.08)
    save(fig, "sweep_ratio_BW")


def fig_baselines():
    """Bars of the five stabilization methods, GROUPED BY DATASET, each the
    observed mean of five seeds with a 95% interval.

    Grouping by dataset rather than pooling is what makes the continuous-decay
    result legible: its rate was calibrated on MNIST, and it only works there.
    Pooled, that shows up as unexplained bimodal spread; grouped, it is a clean
    per-dataset pattern.

    Bars show observed means, so the axis is "test accuracy" and not
    "predicted". Note what a bar cannot show: several cells here are bimodal
    across seeds (Fashion-MNIST/none runs 0.126 to 0.571), and a mean with a
    symmetric interval represents that badly. The box version of this figure is
    recoverable from git if that spread needs to be visible.

    Method identity is carried twice -- grey step and hatch -- because within a
    group it is position, not an axis label, that separates the boxes. The GLMM
    estimates are not drawn here: the fitted model has no method x dataset
    interaction, so a per-group prediction would differ only by the dataset
    intercept and would imply structure the model does not contain. Those
    estimates belong in the table.
    """
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(os.path.join(REPO, "results", "baselines", "baselines_summary.csv")):
        if r["test_accuracy"]:
            runs[r["dataset"]][r["method"]].append(float(r["test_accuracy"]))

    order = ["none", "decay", "norm_neuron", "sleep", "norm_layer"]
    # Abbreviated so the five entries fit on a single legend row; the caption
    # carries the full names ("none" = unregularized STDP, "layer norm." =
    # layer-wise weight normalization).
    nice = {"none": "none", "decay": "continuous decay",
            "norm_neuron": "synaptic scaling", "sleep": "sleep",
            "norm_layer": "layer norm."}
    dsets = [("mnist", "MNIST"), ("kmnist", "KMNIST"),
             ("fmnist", "Fashion-MNIST"), ("notmnist", "NotMNIST")]

    fig = plt.figure(figsize=(15.0, 6.9), facecolor=T["surface"])
    ax = fig.add_axes([0.072, 0.115, 0.915, 0.785])
    style(ax)

    W = 0.155                      # bar width
    STEP = 0.171                   # spacing between methods within a group
    top = 0.0                      # tallest bar + interval, for the y limit
    for gi, (d, _) in enumerate(dsets):
        for mi, m in enumerate(order):
            v = runs[d].get(m, [])
            if not v:
                continue
            pos = gi + (mi - (len(order) - 1) / 2) * STEP
            mean = st.mean(v)
            se = st.stdev(v) / (len(v) ** 0.5) if len(v) > 1 else 0.0
            top = max(top, mean + 1.96 * se, *(to_draw(v) or [0.0]))
            ax.bar(pos, mean, width=W, color=GREY[mi], hatch=HATCH[mi],
                   edgecolor=T["ink"], lw=1.0, zorder=3)
            ax.errorbar(pos, mean, yerr=1.96 * se, fmt="none",
                        ecolor=T["ink"], elinewidth=1.1, capsize=3.5,
                        capthick=1.1, zorder=5)
            dv = to_draw(v)
            if dv:
                ax.plot([pos] * len(dv), dv, "o", mfc="none", mec=T["ink2"],
                        ms=3.8, mew=0.9, ls="none", zorder=6)

    for gi in range(1, len(dsets)):
        ax.axvline(gi - 0.5, color=T["grid"], lw=1.0, zorder=1)

    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels([lab for _, lab in dsets], fontsize=FS_TICK + 1)
    ax.set_xlim(-0.52, len(dsets) - 0.48)
    ax.set_ylabel("test accuracy", fontsize=FS_LABEL, color=T["ink2"])
    # Full 0-1 accuracy range. The legend sits outside the axes to the right,
    # so it costs the data no headroom. `top` (the tallest bar plus interval)
    # is still computed above and is used only to check nothing is clipped.
    assert top <= 1.0, f"an interval exceeds 1.0 ({top:.3f}); the axis would clip it"
    ax.set_ylim(0, 1.0)

    handles = [Patch(facecolor=GREY[i], hatch=HATCH[i], edgecolor=T["ink"],
                     label=nice[m]) for i, m in enumerate(order)]
    if RUN_LABEL is not None:
        handles.append(plt.Line2D([], [], marker="o", mfc="none",
                                  mec=T["ink2"], ls="none", ms=4.5,
                                  label=RUN_LABEL))
    # One horizontal row above the panel, left-to-right in the same order the
    # bars appear within each group. Labels are abbreviated (see `nice`) so all
    # five fit on one line at this width.
    ax.legend(handles=handles, loc="upper center", frameon=True,
              edgecolor=T["axis"], facecolor=T["surface"], framealpha=1.0,
              borderpad=0.5, fontsize=FS_SERIES + 1, labelcolor=T["ink2"],
              ncol=len(order), handletextpad=0.5, handlelength=1.6,
              columnspacing=1.1, bbox_to_anchor=(0.5, 1.075))
    save(fig, "baselines_methods_BW")


# =============================================================================
# Figure 3 — component ablation (R3.2)
# =============================================================================
def fig_ablation():
    pred = rd(os.path.join(GL, "ablation_predictions.csv"))
    ref = float(rd(os.path.join(GL, "ablation_nosleep_reference.csv"))[0]["mean_acc"])

    def on(v):
        return str(v).strip().lower() in ("on", "true", "1")

    rows = []
    for r in pred:
        rows.append(dict(
            ds=on(r["downscale"]), no=on(r["noise"]),
            sd=on(r["stdp"]), su=on(r["suppress"]),
            fit=float(r["fit"]), lo=float(r["lo"]), hi=float(r["hi"])))

    fig = plt.figure(figsize=(11.0, 7.0), facecolor=T["surface"])
    axes = [fig.add_axes([0.30, 0.565, 0.56, 0.355]),
            fig.add_axes([0.30, 0.105, 0.56, 0.355])]

    for ax, want, title in zip(
            axes, (True, False),
            ("downscaling ON", "downscaling OFF")):
        sub = sorted([r for r in rows if r["ds"] is want], key=lambda r: r["fit"])
        style(ax, ygrid=False, xgrid=True)
        ypos = range(len(sub))
        for y, r in zip(ypos, sub):
            # Hatch marks sleep-phase STDP, the component the factorial shows to
            # be harmful; grey depth is fixed within a panel so the hatch is the
            # only encoding that varies and cannot be misread as magnitude.
            ax.barh(y, r["fit"], height=0.66,
                    color=GREY[1] if r["sd"] else GREY[3],
                    hatch="///" if r["sd"] else "",
                    edgecolor=T["ink"], lw=1.1, zorder=3)
            ax.errorbar(r["fit"], y, xerr=[[r["fit"] - r["lo"]], [r["hi"] - r["fit"]]],
                        fmt="none", ecolor=T["ink"], elinewidth=1.2,
                        capsize=4, capthick=1.2, zorder=5)
            # past the upper CI, not the point estimate, or the label lands
            # on the error-bar cap
            ax.text(r["hi"] + 0.015, y, f"{r['fit']:.3f}", va="center",
                    fontsize=FS_SERIES - 1, color=T["ink"])
        labs = []
        for r in sub:
            act = [n for n, f in (("noise", r["no"]), ("STDP", r["sd"]),
                                  ("suppress", r["su"])) if f]
            labs.append(", ".join(act) if act else "(none)")
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(labs, fontsize=FS_TICK)
        ax.axvline(ref, color=T["muted"], lw=1.2, ls="--", zorder=1)
        ax.set_xlim(0, 0.95)
        ax.set_ylim(-0.7, len(sub) - 0.3)
        ax.set_title(title, fontsize=FS_LABEL, color=T["ink"], loc="left", pad=8)

    axes[1].set_xlabel("predicted test accuracy", fontsize=FS_LABEL, color=T["ink2"])
    axes[0].set_xticklabels([])   # shared with the lower panel
    handles = [
        Patch(facecolor=GREY[3], edgecolor=T["ink"], label="sleep-phase STDP off"),
        Patch(facecolor=GREY[1], hatch="///", edgecolor=T["ink"],
              label="sleep-phase STDP on"),
        plt.Line2D([], [], color=T["muted"], ls="--", lw=1.2,
                   label="no-sleep reference"),
    ]
    axes[0].legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.16),
                   frameon=True, edgecolor=T["axis"], facecolor=T["surface"],
              framealpha=1.0, borderpad=0.55, fontsize=FS_SERIES, ncol=3,
                   labelcolor=T["ink2"])
    save(fig, "ablation_components_BW")


if __name__ == "__main__":
    print("writing BW figures to", os.path.relpath(FIG, REPO))
    fig_sweep()
    fig_baselines()
    fig_ablation()
