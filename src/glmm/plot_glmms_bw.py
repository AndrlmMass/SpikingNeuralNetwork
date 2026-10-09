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

Inputs:  results/glmm/*.csv (written by the src/glmm/fit_*.R scripts), results/*/*_summary.csv
Outputs: figures/sweep_ratio_BW.pdf
         figures/baselines_methods_BW.pdf
         figures/ablation_slope_BW.pdf
         figures/sweep_aggregated_{runs,means}_BW.pdf      (all datasets, one panel)
         figures/baselines_aggregated_{runs,means}_BW.pdf
         (each also as .png)
"""
import csv
import math
import os
import collections
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    Predictions come from the same model as the regression table (fit_sweep.R,
    ratio + (1|dataset) + (1|seed) + (1|dataset:seed)): each panel is that dataset's fitted mean
    at every sleep duration with its 95% CI -- fixed effects plus the dataset's
    random intercept, seed at zero.

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
    """Bars of the five stabilization methods, GROUPED BY DATASET: the
    model-predicted accuracy of each method on each dataset with its 95% CI.

    Predictions come from the same model as the regression table
    (fit_baselines.R, method + (1|dataset) + (1|seed) + (1|dataset:seed)):
    fixed effects plus the dataset's random intercept, seed at zero. Dataset
    enters only as an intercept, so the methods differ by the same amount on
    the logit scale in every dataset.

    Method identity is carried twice -- grey step and hatch -- because within a
    group it is position, not an axis label, that separates the bars.
    """
    pred = collections.defaultdict(dict)
    for r in rd(os.path.join(GL, "baselines_predictions_by_dataset.csv")):
        pred[r["dataset"]][r["method"]] = (
            float(r["fit"]), float(r["lo"]), float(r["hi"]))

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
    top = 0.0                      # tallest interval, for the y limit
    for gi, (d, _) in enumerate(dsets):
        for mi, m in enumerate(order):
            if m not in pred[d]:
                continue
            pos = gi + (mi - (len(order) - 1) / 2) * STEP
            fit, lo, hi = pred[d][m]
            top = max(top, hi)
            ax.bar(pos, fit, width=W, color=GREY[mi], hatch=HATCH[mi],
                   edgecolor=T["ink"], lw=1.0, zorder=3)
            ax.errorbar(pos, fit, yerr=[[fit - lo], [hi - fit]], fmt="none",
                        ecolor=T["ink"], elinewidth=1.1, capsize=3.5,
                        capthick=1.1, zorder=5)

    for gi in range(1, len(dsets)):
        ax.axvline(gi - 0.5, color=T["grid"], lw=1.0, zorder=1)

    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels([lab for _, lab in dsets], fontsize=FS_TICK + 1)
    ax.set_xlim(-0.52, len(dsets) - 0.48)
    ax.set_ylabel("predicted test accuracy", fontsize=FS_LABEL, color=T["ink2"])
    # Full 0-1 accuracy range. The legend sits outside the axes to the right,
    # so it costs the data no headroom. `top` (the highest interval end)
    # is still computed above and is used only to check nothing is clipped.
    assert top <= 1.0, f"an interval exceeds 1.0 ({top:.3f}); the axis would clip it"
    ax.set_ylim(0, 1.0)

    handles = [Patch(facecolor=GREY[i], hatch=HATCH[i], edgecolor=T["ink"],
                     label=nice[m]) for i, m in enumerate(order)]
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
# Figure 3 -- component ablation (R3.2): each component off -> on, per dataset
# =============================================================================
DS_ORDER = ["mnist", "kmnist", "fmnist", "notmnist"]
DS_NICE = {"mnist": "MNIST", "kmnist": "KMNIST", "fmnist": "Fashion-MNIST",
           "notmnist": "NotMNIST"}
DS_MARK = {"mnist": "o", "kmnist": "s", "fmnist": "^", "notmnist": "D"}
# Panel titles, short enough to fit on one line at this size; the caption gives
# the full names (membrane noise, input suppression).
SHORT = {"downscale": "downscaling", "stdp": "STDP", "noise": "noise",
         "suppress": "no input"}


def fig_ablation():
    """One panel per sleep component: predicted accuracy with the component off
    and on, one line per dataset. A slope chart rather than bars, to set it
    apart from the baselines figure.

    From the coefficients of the ablation interaction GLMM
    (fit_ablation_datasets.R; results/glmm/ablation_ds_marginal.csv): "off" is
    the intercept (all components off) and "on" is the intercept plus that
    component's coefficient, each with the dataset's random intercept and seed
    at zero. The panels share the y-axis, so slopes are directly comparable.
    The no-sleep reference is the mean over datasets of the runs without sleep.
    """
    rows = rd(os.path.join(GL, "ablation_ds_marginal.csv"))
    m = {(r["dataset"], r["component"], r["state"]): float(r["fit"]) for r in rows}
    ref = st.mean(float(r["mean_acc"])
                  for r in rd(os.path.join(GL, "ablation_ds_nosleep_reference.csv")))
    comps = ["downscale", "stdp", "noise", "suppress"]
    dodge = dict(zip(DS_ORDER, (-0.09, -0.03, 0.03, 0.09)))

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 6.0), sharey=True,
                             facecolor=T["surface"])
    for ax, comp in zip(axes, comps):
        style(ax)
        col = T["ink"] if comp == "downscale" else GREY[3]
        for ds in DS_ORDER:
            xs = [0 + dodge[ds], 1 + dodge[ds]]
            ys = [m[(ds, comp, s)] for s in ("off", "on")]
            ax.plot(xs, ys, color=col, lw=3.4, zorder=3)
            ax.plot(xs, ys, ls="", marker=DS_MARK[ds], ms=13, mfc=col, mec=T["surface"],
                    mew=1.0, zorder=5)
        ax.axhline(ref, color=T["muted"], ls=":", lw=2.2, zorder=2)
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["off", "on"])
        ax.tick_params(labelsize=FS_TICK + 4)
        ax.set_title(SHORT[comp], fontsize=FS_LABEL + 1, color=T["ink"], pad=12)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("predicted test accuracy", fontsize=FS_LABEL + 3, color=T["ink2"])

    # Stacked, centred on the last panel's right edge: half inside the empty
    # area above the flat lines, half outside.
    handles = [Line2D([], [], ls="", marker=DS_MARK[d], ms=14, mfc=T["muted"],
                      mec=T["surface"], label=DS_NICE[d]) for d in DS_ORDER]
    handles.append(Line2D([], [], color=T["muted"], ls=":", lw=2.4, label="no sleep"))
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(1.0, 1.0),
                    ncol=1, frameon=True, edgecolor=T["axis"], facecolor=T["surface"],
                    framealpha=1.0, fontsize=FS_SERIES, labelcolor=T["ink2"],
                    handletextpad=0.4, borderpad=0.4, labelspacing=0.2, handlelength=1.6)
    fig.tight_layout(w_pad=2.5)
    save(fig, "ablation_slope_BW")


def fig_ablation_aggregated():
    """Population-level version of the ablation figure: per component, the
    accuracy predicted from the interaction GLMM's coefficients for a typical
    dataset and seed (random effects at zero) with its 95% CI
    (results/glmm/ablation_pop_marginal.csv). "off" is the intercept (all
    components off); "on" adds that component's coefficient. The open markers
    are the observed per-dataset means of the same two conditions: the all-off
    runs, and the runs with only that component on."""
    pop = {(r["component"], r["state"]): (float(r["fit"]), float(r["lo"]), float(r["hi"]))
           for r in rd(os.path.join(GL, "ablation_pop_marginal.csv"))}
    comps = ["downscale", "stdp", "noise", "suppress"]
    obs = collections.defaultdict(list)
    ref = []
    for path in (os.path.join(REPO, "results", "ablation", "ablation_summary.csv"),
                 os.path.join(REPO, "results", "ablation_datasets", "ablation_summary.csv")):
        for r in rd(path):
            if not r["test_accuracy"]:
                continue
            if r["code"] == "none":
                ref.append(float(r["test_accuracy"]))
                continue
            on = [c for c in comps if r[c].strip().lower() in ("true", "1")]
            acc = float(r["test_accuracy"])
            if not on:                     # all off: the "off" point of every panel
                for c in comps:
                    obs[(c, "off", r["dataset"])].append(acc)
            elif len(on) == 1:             # only this component on
                obs[(on[0], "on", r["dataset"])].append(acc)
    dodge = dict(zip(DS_ORDER, (-0.21, -0.07, 0.07, 0.21)))
    dot = "#555555"

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.6), sharey=True, facecolor=T["surface"])
    for ax, comp in zip(axes, comps):
        style(ax)
        for j, s in enumerate(("off", "on")):
            for d in DS_ORDER:
                ax.plot(j + dodge[d], st.mean(obs[(comp, s, d)]), DS_MARK[d], mfc="white",
                        mec=dot, ms=10, mew=1.5, zorder=4)
        f = [pop[(comp, s)] for s in ("off", "on")]
        ax.plot([0, 1], [v[0] for v in f], color=T["ink"], lw=2.6, marker="v", ms=11, zorder=5)
        ax.errorbar([0, 1], [v[0] for v in f], yerr=[[v[0] - v[1] for v in f],
                                                     [v[2] - v[0] for v in f]],
                    fmt="none", ecolor=T["ink"], elinewidth=1.8, capsize=7, capthick=1.8,
                    zorder=6)
        ax.axhline(st.mean(ref), color=T["muted"], ls=":", lw=2.0, zorder=1)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["off", "on"])
        ax.tick_params(labelsize=FS_TICK + 1)
        ax.set_title(SHORT[comp], fontsize=FS_LABEL, color=T["ink"], pad=12)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("test accuracy", fontsize=FS_LABEL + 2, color=T["ink2"])
    pop_h = Line2D([], [], color=T["ink"], lw=2.6, marker="v", ms=11, label="population (95% CI)")
    ds_h = [Line2D([], [], ls="", marker=DS_MARK[d], mfc="white", mec=dot, ms=10, mew=1.5,
                   label=DS_NICE[d]) for d in DS_ORDER]
    # "none": the 20 no-sleep runs, identical to the baselines' "none" arm.
    none_h = Line2D([], [], color=T["muted"], ls=":", lw=2.0, label="none")
    _legend(axes[-1], pop_h, ds_h, "means", extra=[none_h], loc="upper left",
            bbox_to_anchor=(1.04, 1.0), fontsize=FS_SERIES)
    fig.tight_layout(w_pad=1.5)
    save(fig, "ablation_aggregated_BW")


# =============================================================================
# Aggregated versions of figures 1 and 2 — one panel, all datasets
# =============================================================================
# The population-level prediction (random effects at zero: a typical dataset and
# seed, 95% CI for the mean) drawn over the observed data, with one marker shape
# per dataset. Two variants each: "runs" shows every run, "means" each dataset's
# mean over seeds.
def _legend(ax, pop, ds, variant, extra=(), **kw):
    """Population line first (then any `extra` entries), then one subheading
    ("sample average" / "runs") over the dataset markers, so the qualifier is
    not repeated per dataset."""
    head = Line2D([], [], ls="", marker="", label="sample average" if variant == "means" else "runs")
    leg = ax.legend(handles=[pop, *extra, head] + ds, frameon=True, edgecolor=T["axis"],
                    facecolor=T["surface"], framealpha=1.0, labelcolor=T["ink2"], **kw)
    leg.get_texts()[1 + len(extra)].set_fontstyle("italic")
    return leg


def _obs(path, key):
    obs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(path):
        if r["test_accuracy"]:
            obs[r["dataset"]][r[key]].append(float(r["test_accuracy"]))
    return obs


def fig_sweep_aggregated(variant):
    pred = sorted((float(r["sleep_rate"]), float(r["fit"]), float(r["lo"]), float(r["hi"]))
                  for r in rd(os.path.join(GL, "sweep_predictions.csv")))
    x = [100 * p[0] for p in pred]
    obs = _obs(os.path.join(REPO, "results", "sweep", "sweep_summary.csv"), "sleep_rate")
    # Each dataset at its own small offset around the sleep duration, as in the
    # aggregated baselines figure, so the points group by dataset. No lines
    # between sleep durations: each duration is a separate condition.
    dodge = dict(zip(DS_ORDER, (-3.0, -1.0, 1.0, 3.0)))
    dot = "#555555"
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor=T["surface"])
    style(ax)
    handles = []
    for d in DS_ORDER:
        mk = DS_MARK[d]
        if variant == "runs":
            for r, v in obs[d].items():
                ax.plot(100 * float(r) + dodge[d] + rng.uniform(-0.35, 0.35, len(v)), v, mk,
                        mfc="white", mec=dot, ms=5.5, mew=1.0, zorder=4)
            handles.append(Line2D([], [], ls="", marker=mk, mfc="white", mec=dot,
                                  ms=7, mew=1.0, label=DS_NICE[d]))
        else:
            rs = sorted(obs[d], key=float)
            ax.plot([100 * float(r) + dodge[d] for r in rs], [st.mean(obs[d][r]) for r in rs],
                    mk, mfc="white", mec=dot, ms=8, mew=1.4, zorder=4)
            handles.append(Line2D([], [], ls="", marker=mk, mfc="white", mec=dot, ms=8,
                                  mew=1.4, label=DS_NICE[d]))
    ax.fill_between(x, [p[2] for p in pred], [p[3] for p in pred], color=T["ink"],
                    alpha=0.12, lw=0, zorder=3)
    ax.plot(x, [p[1] for p in pred], color=T["ink"], lw=2.6, marker="v", ms=9, zorder=5)
    pop = Line2D([], [], color=T["ink"], lw=2.6, marker="v", ms=9, label="population (95% CI)")
    ax.axhline(0.1, color=T["muted"], ls=":", lw=1.4, zorder=1)
    ax.text(99, 0.115, "chance", ha="right", fontsize=FS_SERIES - 3, color=T["muted"])
    ax.set_xlabel("sleep duration (%)", fontsize=FS_LABEL + 2, color=T["ink2"])
    ax.set_ylabel("test accuracy", fontsize=FS_LABEL + 2, color=T["ink2"])
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlim(-5.5, 105.5)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=FS_TICK + 1)
    _legend(ax, pop, handles, variant, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            fontsize=FS_SERIES, markerscale=1.3)
    save(fig, f"sweep_aggregated_{variant}_BW")


def fig_baselines_aggregated(variant):
    pred = {r["method"]: (float(r["fit"]), float(r["lo"]), float(r["hi"]))
            for r in rd(os.path.join(GL, "baselines_predictions.csv"))}
    obs = _obs(os.path.join(REPO, "results", "baselines", "baselines_summary.csv"), "method")
    # Methods sorted by population-level predicted accuracy, lowest first; the
    # bars darken from left to right with the same ordering.
    order = sorted(pred, key=lambda m: pred[m][0])
    nice = {"none": "none", "decay": "continuous\ndecay", "norm_neuron": "synaptic\nscaling",
            "sleep": "sleep", "norm_layer": "layer\nnorm."}
    dodge = dict(zip(DS_ORDER, (-0.21, -0.07, 0.07, 0.21)))
    dot = "#555555"
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(12.5, 6.2), facecolor=T["surface"])
    style(ax)
    for i, m in enumerate(order):
        fit, lo, hi = pred[m]
        ax.bar(i, fit, width=0.66, color=GREY[i], hatch=HATCH[i], edgecolor=T["ink"], lw=1.0,
               zorder=3, alpha=0.8)
        ax.errorbar(i, fit, yerr=[[fit - lo], [hi - fit]], fmt="none", ecolor=T["ink"],
                    elinewidth=1.6, capsize=6, capthick=1.6, zorder=6)
        for d in DS_ORDER:
            v = obs[d][m]
            if variant == "runs":
                ax.plot(i + dodge[d] + rng.uniform(-0.04, 0.04, len(v)), v, DS_MARK[d],
                        mfc="white", mec=dot, ms=6.5, mew=1.2, zorder=5)
            else:
                ax.plot(i + dodge[d], st.mean(v), DS_MARK[d], mfc="white", mec=dot, ms=10,
                        mew=1.6, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([nice[m] for m in order], fontsize=FS_TICK + 3)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=FS_TICK + 1)
    ax.set_ylim(0, 0.82)  # highest run is .775
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("test accuracy", fontsize=FS_LABEL + 2, color=T["ink2"])
    pop = Line2D([], [], color=T["ink"], lw=1.6, marker="_", ms=14, label="population (95% CI)")
    handles = [Line2D([], [], ls="", marker=DS_MARK[d], mfc="white", mec=dot, ms=10, mew=1.4,
                      label=DS_NICE[d]) for d in DS_ORDER]
    # Outside the axes, top-aligned: inside, it would cover the highest runs.
    _legend(ax, pop, handles, variant, loc="upper left", bbox_to_anchor=(1.01, 1.0),
            fontsize=FS_SERIES, handletextpad=0.5)
    save(fig, f"baselines_aggregated_{variant}_BW")


if __name__ == "__main__":
    print("writing BW figures to", os.path.relpath(FIG, REPO))
    fig_sweep()
    fig_baselines()
    fig_ablation()
    fig_ablation_aggregated()
    for v in ("runs", "means"):
        fig_sweep_aggregated(v)
        fig_baselines_aggregated(v)
