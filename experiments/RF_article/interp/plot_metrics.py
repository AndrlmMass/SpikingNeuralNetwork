"""
Live metrics dashboard for one interp run — regenerated at every checkpoint.

Complements confusion.png (which covers per-class recall and the confusion matrices)
and replaces what the library's stats.png was being used for. The distinction matters:
stats.png is driven by `--track-stats` and plots WEIGHT-space diagnostics
(`rf_participation_ratio`, `rf_mean_cosine`, `rf_gini` are all computed on the W_se
columns). Those are confounded by receptive-field size — they drop mechanically as RFs
shrink, regardless of feature content — so they cannot answer "did the representation
get better". Everything here is computed on the RESPONSES instead.

Top anchor: the three readouts that mean different things.
  pool     uniform pooling over each class group. No fitted parameters, so it is the
           network's decision in its rawest form.
  learned  the reward rule's plastic readout. Still the network's decision, and the
           only one that can express evidence AGAINST a class.
  probe    a linear classifier refitted each checkpoint = a CONTROL. It bounds what is
           extractable from the representation, not what the network achieves.
Grouped eta-squared (`phi`) rides the twin axis as the clustering anchor.

Panels, and what each is for:
  representation   eta2 up + within-group correlation down = neurons carry class
                   information without duplicating each other. eta2 replaces
                   class_selectivity, which was confounded by firing rate.
  dimensionality   participation ratio on the RESPONSE covariance, plus how many
                   neurons clear the activity threshold at all.
  uncertainty      perplexity (effective classes still in play) and margin (top-1 minus
                   the best rival). Both come from the pooled share distribution.
  health           dead fraction and winner entropy — the monopolization diagnostics.
  prior and drift  orientation coherence tracks whether the RF prior survives training;
                   refit minus fixed is the DRIFT gap, i.e. how far the representation
                   has rotated off the directions its readout was fitted on. A growing
                   gap with flat refit is drift without learning.

  python experiments/RF_article/interp/plot_metrics.py --results <run_dir>
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _ser(traj, key):
    return np.asarray([t.get(key, np.nan) for t in traj], dtype=float)


def _has(traj, key):
    return np.any(np.isfinite(_ser(traj, key)))


def _twin(fig, pos, title, xs, traj, left, right, ylab_l="", ylab_r=""):
    ax = fig.add_subplot(pos)
    ax2 = ax.twinx()
    for key, lbl, c in left:
        if _has(traj, key):
            ax.plot(xs, _ser(traj, key), "-o", ms=3, color=c, label=lbl)
    for key, lbl, c in right:
        if _has(traj, key):
            ax2.plot(xs, _ser(traj, key), "s", ms=3, ls="--", color=c, label=lbl)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("checkpoint"); ax.grid(alpha=0.3)
    ax.set_ylabel(ylab_l, fontsize=8, color=left[0][2] if left else "k")
    ax2.set_ylabel(ylab_r, fontsize=8, color=right[0][2] if right else "k")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
    return ax


def make_metrics_plot(d, outdir):
    """Render metrics.png from a results dict (works mid-run on a partial dict)."""
    traj = d.get("trajectory", [])
    if not traj:
        return None
    tag = d.get("config", {}).get("tag", "run")
    xs = [t["batch"] + 1 for t in traj]

    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    # ---- anchor: the three readouts, plus grouped eta-squared on the twin ----
    ax = fig.add_subplot(gs[0, :])
    for key, lbl, c in [("readout_learned_acc", "readout (learned)", "#7F77DD"),
                        ("softmax_acc", "pool (uniform)", "#D85A30"),
                        ("refit_acc", "linear probe, refit (control)", "#888780")]:
        if _has(traj, key):
            ax.plot(xs, _ser(traj, key), "-o", ms=4, color=c, label=lbl)
    ax.axhline(0.1, ls=":", lw=0.8, color="k")
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_xlabel("checkpoint")
    ax.set_title("Readout accuracy — the probe is a CONTROL (external decoder), "
                 "not something the network does", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    if _has(traj, "val_phi"):
        axp = ax.twinx()
        axp.plot(xs, _ser(traj, "val_phi"), "-", color="#1D9E75", lw=1.6,
                 label="grouped $\\eta^2$ (phi)")
        axp.set_ylabel("grouped $\\eta^2$", color="#1D9E75")
        axp.tick_params(axis="y", labelcolor="#1D9E75")
        axp.legend(loc="lower right", fontsize=8)

    _twin(fig, gs[1, 0], "representation", xs, traj,
          [("eta2", "per-neuron $\\eta^2$ (up)", "#1D9E75")],
          [("corr_within", "within-group |corr| (down)", "#D85A30"),
           ("corr_all", "overall |corr|", "#F0997B")],
          "class-explained variance", "redundancy")

    ax = _twin(fig, gs[1, 1], "dimensionality", xs, traj,
               [("pr", "participation ratio", "#378ADD"),
                ("pr_cov", "PR (covariance)", "#85B7EB")],
               [("n_active", "active neurons", "#5F5E5A")],
               "effective dimensions", "count")

    # Learned-readout perplexity is the primary series: the pooled one sits near
    # chance regardless of representation quality, because thin local features are
    # shared across digits so every class group ends up with a similar MEAN rate.
    # Pooling averages away the pattern of WHICH neurons fired. The pooled curve is
    # kept faint for reference, on the same axis, so the gap is visible.
    _twin(fig, gs[1, 2], "uncertainty (learned readout)", xs, traj,
          [("perplexity_readout", "perplexity, readout (down)", "#534AB7"),
           ("perplexity", "perplexity, pooled (reference)", "#CECBF6")],
          [("margin_readout", "margin, readout (up)", "#0F6E56"),
           ("share_ce", "share CE (pooled)", "#B4B2A9")],
          "effective classes in play", "margin / CE")

    _twin(fig, gs[2, 0], "population health", xs, traj,
          [("dead_frac", "dead fraction (down)", "#A32D2D")],
          [("winner_entropy", "winner entropy (up)", "#639922"),
           ("frac_ever_winner", "ever a winner", "#97C459")],
          "fraction dead", "spread")

    _twin(fig, gs[2, 1], "prior survival and drift", xs, traj,
          [("orient_coh", "orientation coherence", "#BA7517")],
          [("_drift", "refit $-$ fixed (drift)", "#993C1D")],
          "coherence", "accuracy gap")

    _twin(fig, gs[2, 2], "drive decomposition", xs, traj,
          [("cur_se", "feedforward |SE|", "#185FA5"),
           ("cur_ie", "inhibitory |IE|", "#A32D2D"),
           ("cur_ee", "recurrent |EE|", "#5F5E5A")],
          [("w_floor_frac", "W_se at floor", "#888780")],
          "mean drive", "fraction")

    fig.suptitle(f"Representation metrics — {tag}", fontsize=14)
    out = os.path.join(outdir, "metrics.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run dir or results.json path")
    args = ap.parse_args()
    path = args.results
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    d = json.load(open(path))
    # derived series the harness does not store directly
    for t in d.get("trajectory", []):
        if t.get("refit_acc") is not None and t.get("fixed_acc") is not None:
            t["_drift"] = t["refit_acc"] - t["fixed_acc"]
    print("saved ->", make_metrics_plot(d, os.path.dirname(path)))


if __name__ == "__main__":
    main()
