"""Three alternative layouts for the component-ablation figure, plus a
distribution-based replacement for the baselines figure.

The submitted ablation figure sorted all 16 factorial cells by accuracy and
labelled each with a comma-separated component list. That hides the thing the
figure exists to show: the design is a 2^4 factorial in which only
`downscale` and `stdp` matter, and they interact. Three alternatives:

  A  interaction    the 2x2 that carries the result, plus the evidence that
                    the two collapsed factors are null
  B  forest         every coefficient with its CI, ordered by magnitude
  C  matrix         all 16 cells, condition read off a dot matrix instead of
                    a text label (UpSet-style)

Style is inherited from plot_glmms_bw.py so these drop into the same paper.

    python src/glmm/plot_ablation_alts.py

Outputs: figures/ablation_A_interaction_BW.{pdf,png}
         figures/ablation_B_forest_BW.{pdf,png}
         figures/ablation_C_matrix_BW.{pdf,png}
         figures/baselines_methods_BW.{pdf,png}   (boxplots, replaces bars)
"""
import collections
import csv
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_glmms_bw import T, FS_LABEL, FS_TICK, FS_SERIES, GREY, HATCH, style, rd, save, REPO, GL

FACTORS = ("downscale", "noise", "stdp", "suppress")
NICE = {"downscale": "downscaling", "noise": "membrane noise",
        "stdp": "sleep-phase STDP", "suppress": "input suppression"}


def load_cells():
    """Per-condition lists of raw run accuracies, keyed by the 4-bool tuple."""
    cells = collections.defaultdict(list)
    ref = []
    for r in rd(os.path.join(REPO, "results", "ablation", "ablation_summary.csv")):
        if not r["test_accuracy"]:
            continue
        acc = float(r["test_accuracy"])
        if r["code"] == "none":
            ref.append(acc)
            continue
        key = tuple(r[f].strip().lower() in ("true", "1") for f in FACTORS)
        cells[key].append(acc)
    return cells, ref


def mean_ci(vals):
    """Mean and normal-approximation 95% CI. Used only for display; the
    inferential numbers in the paper come from the GLMM."""
    m = st.mean(vals)
    if len(vals) < 2:
        return m, m, m
    se = st.stdev(vals) / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


# =============================================================================
# A — interaction plot: the 2x2 that carries the result
# =============================================================================
def alt_A():
    """Left: accuracy against sleep-phase STDP, one line per downscaling level,
    collapsed over noise and input suppression. Right: the evidence that
    collapsing those two is legitimate -- their effect within each tier, with
    intervals straddling zero.

    This is the message-first option. It shows the interaction as two
    non-parallel lines: STDP costs ~3 points when downscaling is present and
    ~40 when it is not.
    """
    cells, ref = load_cells()
    fig = plt.figure(figsize=(15.5, 6.6), facecolor=T["surface"])
    axL = fig.add_axes([0.070, 0.315, 0.40, 0.615])
    axR = fig.add_axes([0.665, 0.315, 0.322, 0.615])
    style(axL); style(axR)

    for di, down in enumerate((True, False)):
        xs, ys, los, his = [], [], [], []
        for si, stdp in enumerate((False, True)):
            v = [a for k, vv in cells.items() if k[0] == down and k[2] == stdp
                 for a in vv]
            m, lo, hi = mean_ci(v)
            xs.append(si); ys.append(m); los.append(lo); his.append(hi)
        mk = "o" if down else "s"
        axL.errorbar(xs, ys, yerr=[[m - l for m, l in zip(ys, los)],
                                   [h - m for m, h in zip(ys, his)]],
                     marker=mk, ms=11, mfc=T["ink"] if down else "none",
                     mec=T["ink"], mew=1.6, color=T["ink"],
                     ls="-" if down else "--", lw=2.0,
                     ecolor=T["ink"], elinewidth=1.2, capsize=5, capthick=1.2,
                     zorder=4,
                     label=f"downscaling {'on' if down else 'off'}")
        for x, y in zip(xs, ys):
            axL.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(-14 if x else 0, 18 if down else -28),
                         ha="right" if x else "center",
                         fontsize=FS_SERIES, color=T["ink"])

    rm = st.mean(ref)
    axL.axhline(rm, ls=":", lw=1.4, color=T["muted"], zorder=2)
    axL.annotate("no sleep at all", (1.44, rm), xytext=(0, 9),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=FS_SERIES - 2, color=T["muted"])
    axL.set_xticks([0, 1]); axL.set_xticklabels(["off", "on"], fontsize=FS_TICK)
    axL.set_xlim(-0.30, 1.45)
    axL.set_ylim(0, 0.95)
    axL.set_xlabel("sleep-phase STDP", fontsize=FS_LABEL - 2, color=T["ink2"])
    axL.set_ylabel("test accuracy", fontsize=FS_LABEL - 2, color=T["ink2"])
    # Legend below the axis: inside the panel it landed on the 0.539 label and
    # on the no-sleep rule.
    axL.legend(loc="upper center", frameon=True, edgecolor=T["axis"], facecolor=T["surface"],
              framealpha=1.0, borderpad=0.55, fontsize=FS_SERIES,
               labelcolor=T["ink2"], ncol=2, bbox_to_anchor=(0.5, -0.235),
               columnspacing=2.0, handletextpad=0.7)

    # Right panel: effect of each collapsed factor, within each downscaling tier
    # Single-line row labels. Two-line labels reading
    # "membrane noise / (downscaling on)" were wide enough at this font size
    # to cross into the left panel, so the downscaling level is annotated once
    # per group inside the panel instead of repeated on every row.
    labels, ests, elo, ehi, ypos, groups = [], [], [], [], [], []
    y = 0
    for down in (True, False):
        groups.append((y, f"downscaling {'on' if down else 'off'}"))
        for f, fi in (("noise", 1), ("suppress", 3)):
            on = [a for k, vv in cells.items() if k[0] == down and k[fi]
                  for a in vv]
            off = [a for k, vv in cells.items() if k[0] == down and not k[fi]
                   for a in vv]
            d = st.mean(on) - st.mean(off)
            se = math.sqrt(st.stdev(on) ** 2 / len(on) + st.stdev(off) ** 2 / len(off))
            labels.append(NICE[f])
            ests.append(d); elo.append(d - 1.96 * se); ehi.append(d + 1.96 * se)
            ypos.append(y); y += 1
        y += 0.6

    axR.axvline(0, ls="-", lw=1.2, color=T["axis"], zorder=2)
    axR.errorbar(ests, ypos,
                 xerr=[[e - l for e, l in zip(ests, elo)],
                       [h - e for e, h in zip(ests, ehi)]],
                 fmt="o", ms=9, mfc=T["surface"], mec=T["ink"], mew=1.6,
                 ecolor=T["ink"], elinewidth=1.3, capsize=5, capthick=1.3,
                 ls="none", zorder=4)
    axR.set_yticks(ypos)
    axR.set_yticklabels(labels, fontsize=FS_TICK - 3)
    axR.invert_yaxis()
    for gy, gl in groups:
        axR.annotate(gl, (0.02, gy - 0.34), xycoords=("axes fraction", "data"),
                     ha="left", va="center", fontsize=FS_SERIES - 2,
                     color=T["muted"])
    axR.set_xlabel("change in test accuracy", fontsize=FS_LABEL - 4,
                   color=T["ink2"])
    axR.grid(False, axis="y")
    axR.grid(True, axis="x", color=T["grid"], lw=0.7)
    save(fig, "ablation_A_interaction_BW")


# =============================================================================
# B — forest plot of the GLMM coefficients
# =============================================================================
def alt_B():
    """Every term of the 2^4 GLMM with its 95% interval, ordered by magnitude,
    on the logit scale the model is fitted on.

    This is the statistics-first option: it is the conventional way to report a
    factorial, it shows all 15 terms without the reader decoding conditions,
    and the zero line makes the null factors unmistakable.
    """
    rows = rd(os.path.join(GL, "ablation_coefs.csv"))
    pretty = {"(Intercept)": "intercept (all components off)"}
    def name(term):
        if term in pretty:
            return pretty[term]
        parts = [p for p in term.split(":")]
        short = {"downscaleon": "downscale", "noiseon": "noise",
                 "stdpon": "STDP", "suppresson": "suppress"}
        return " x ".join(short.get(p, p) for p in parts)

    items = []
    for r in rows:
        if r["term"] == "(Intercept)":
            continue
        e = float(r["estimate"]); se = float(r["std_error"])
        items.append((name(r["term"]), e, e - 1.96 * se, e + 1.96 * se,
                      float(r["p"])))
    items.sort(key=lambda t: abs(t[1]))

    fig = plt.figure(figsize=(11.5, 8.4), facecolor=T["surface"])
    ax = fig.add_axes([0.32, 0.10, 0.65, 0.86])
    style(ax, ygrid=False, xgrid=True)
    ax.axvline(0, ls="-", lw=1.3, color=T["axis"], zorder=2)

    for i, (lab, e, lo, hi, p) in enumerate(items):
        sig = p < 0.05
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", ms=10,
                    mfc=T["ink"] if sig else T["surface"], mec=T["ink"],
                    mew=1.6, ecolor=T["ink"], elinewidth=1.3, capsize=5,
                    capthick=1.3, zorder=4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([t[0] for t in items], fontsize=FS_TICK - 1)
    ax.set_ylim(-0.8, len(items) - 0.2)
    ax.set_xlabel("coefficient (logit scale), 95% CI", fontsize=FS_LABEL - 3,
                  color=T["ink2"])
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", mfc=T["ink"], mec=T["ink"], ls="none",
                   ms=9, label="p < 0.05"),
        plt.Line2D([], [], marker="o", mfc=T["surface"], mec=T["ink"],
                   ls="none", ms=9, mew=1.6, label="n.s.")],
        loc="lower right", frameon=True, edgecolor=T["axis"], facecolor=T["surface"],
              framealpha=1.0, borderpad=0.55, fontsize=FS_SERIES,
        labelcolor=T["ink2"])
    save(fig, "ablation_B_forest_BW")


# =============================================================================
# C — all 16 cells, condition read off a dot matrix (UpSet-style)
# =============================================================================
def alt_C():
    """All 16 factorial cells as bars, sorted by accuracy, with the condition
    shown as a filled/open dot matrix beneath rather than a comma-separated
    text label.

    This is the complete option: nothing is collapsed or hidden, but the
    reader decodes each condition by position instead of by reading prose, and
    the tier structure (downscaling sets the level, STDP subtracts within it)
    becomes visible as a pattern in the dot rows.
    """
    cells, ref = load_cells()
    order = sorted(cells, key=lambda k: -st.mean(cells[k]))

    fig = plt.figure(figsize=(13.5, 8.0), facecolor=T["surface"])
    axB = fig.add_axes([0.085, 0.36, 0.895, 0.60])   # bars
    axM = fig.add_axes([0.085, 0.115, 0.895, 0.235]) # dot matrix
    style(axB, ygrid=True)

    xs = range(len(order))
    for i, k in enumerate(order):
        m, lo, hi = mean_ci(cells[k])
        # Grey step by downscaling level: the factor that sets the tier.
        ax_face = GREY[1] if k[0] else GREY[3]
        axB.bar(i, m, width=0.68, color=ax_face, edgecolor=T["ink"], lw=1.1,
                zorder=3)
        axB.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none",
                     ecolor=T["ink"], elinewidth=1.2, capsize=4, capthick=1.2,
                     zorder=5)
        axB.text(i, hi + 0.02, f"{m:.3f}", ha="center", va="bottom",
                 fontsize=FS_SERIES - 3, color=T["ink"], rotation=90)

    rm = st.mean(ref)
    axB.axhline(rm, ls=":", lw=1.5, color=T["ink2"], zorder=6)
    axB.annotate(f"no sleep at all ({rm:.3f})", (len(order) - 0.45, rm),
                 xytext=(0, 6), textcoords="offset points", ha="right",
                 va="bottom", fontsize=FS_SERIES - 1, color=T["ink2"])

    axB.set_xlim(-0.6, len(order) - 0.4)
    axB.set_ylim(0, 1.0)
    axB.set_xticks([])
    axB.set_ylabel("test accuracy", fontsize=FS_LABEL - 2, color=T["ink2"])
    axB.legend(handles=[
        Patch(facecolor=GREY[1], edgecolor=T["ink"], label="downscaling on"),
        Patch(facecolor=GREY[3], edgecolor=T["ink"], label="downscaling off")],
        loc="upper right", frameon=True, edgecolor=T["axis"], facecolor=T["surface"],
              framealpha=1.0, borderpad=0.55, fontsize=FS_SERIES,
        labelcolor=T["ink2"])

    # --- dot matrix ---------------------------------------------------------
    axM.set_facecolor(T["surface"])
    for s in ("top", "right", "left", "bottom"):
        axM.spines[s].set_visible(False)
    for fi, f in enumerate(FACTORS):
        y = len(FACTORS) - 1 - fi
        axM.axhline(y, color=T["grid"], lw=6, zorder=1)   # guide rail
        for i, k in enumerate(order):
            on = k[fi]
            axM.plot(i, y, "o", ms=11, mfc=T["ink"] if on else T["surface"],
                     mec=T["ink"], mew=1.4, zorder=3)
    axM.set_xlim(-0.6, len(order) - 0.4)
    axM.set_ylim(-0.6, len(FACTORS) - 0.4)
    axM.set_xticks([])
    axM.set_yticks(range(len(FACTORS)))
    axM.set_yticklabels([NICE[f] for f in reversed(FACTORS)],
                        fontsize=FS_TICK - 1, color=T["ink2"])
    axM.tick_params(length=0)
    # Key goes under the leftmost columns: on the right it sat on the last
    # row's dots.
    axM.annotate("filled = component active", (-0.5, -0.56),
                 ha="left", va="top", fontsize=FS_SERIES - 1,
                 color=T["muted"], annotation_clip=False)
    save(fig, "ablation_C_matrix_BW")




if __name__ == "__main__":
    print("writing ablation alternatives to figures")
    alt_A(); alt_B(); alt_C()
