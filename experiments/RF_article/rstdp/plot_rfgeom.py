"""
Plot an RF-geometry sweep: redundancy, dimensionality, uncertainty, accuracy.

Reads each cell's `uncertainty_features.npz` (dumped by interp_harness at end of run)
and recomputes everything post-hoc, so this needs no re-training and can be re-run with
different groupings or thresholds.

Why post-hoc rather than the scalars in results.json: `corr_within` is logged only as a
MEAN over the 10 class groups, which hides whether a config helps every class or just
rescues one. Recomputing from the features gives all 10 per-group values, so the
redundancy panel can be a real boxplot rather than a bar.

Four panels, in the order they should be read:
  1. within-group correlation   the redundancy we are trying to break. LOWER is better.
                                Boxplot over the 10 class groups.
  2. participation ratio        effective dimensionality; a linear readout needs >= K-1
                                dims to separate K classes, so the 9 line is a hard floor.
  3. perplexity                 effective number of classes still in play, per readout.
                                LOWER = sharper predictive distribution.
  4. accuracy                   pool (uniform, network-native) vs learned readout vs
                                linear probe. The probe is a CONTROL -- an external
                                decoder, not part of the network -- so it bounds what is
                                extractable rather than what the network achieves.

  python experiments/RF_article/rstdp/plot_rfgeom.py --run-dir results/rstdp_rfgeom/rfgeom_main
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "RF_article", "interp"))

from neurosnn._evaluation.analysis import active_mask, participation_ratio
from uncertainty import group_rates, share_probs, softmax_probs

ORDER = ["base", "margin", "thin", "thin_margin", "dom", "no_wta"]
C_POOL, C_LEARNED, C_PROBE = "#D85A30", "#7F77DD", "#888780"


def per_group_correlation(X, assignment, n_groups=10, min_rel=0.02):
    """|pairwise correlation| within each class group -> one value per group."""
    m = active_mask(X, min_rel) & (X.std(0) > 1e-12)
    if m.sum() < 2:
        return []
    Xa, ga = X[:, m], np.asarray(assignment)[m]
    R = np.corrcoef(Xa.T)
    out = []
    for g in range(n_groups):
        idx = np.flatnonzero(ga == g)
        if idx.size >= 2:
            sub = R[np.ix_(idx, idx)]
            out.append(float(np.nanmean(np.abs(sub[np.triu_indices(idx.size, 1)]))))
    return out


def perplexity_of(p):
    H = -(p * np.log(p + 1e-12)).sum(1)
    return float(np.exp(H).mean())


def load_cell(cell_dir):
    """Everything the panels need for one cell, or None if it did not finish."""
    npz = os.path.join(cell_dir, "uncertainty_features.npz")
    rj = os.path.join(cell_dir, "results.json")
    if not (os.path.isfile(npz) and os.path.isfile(rj)):
        return None
    d = np.load(npz)
    res = json.load(open(rj))
    if "test_acc" not in res:
        return None
    X, y, asg = d["X_test"], d["y_test"], d["assignment"]

    R = group_rates(X, asg, 10)
    p_pool = share_probs(R)
    rec = dict(
        corr_groups=per_group_correlation(X, asg),
        pr=participation_ratio(X, scale_free=True),
        n_active=int(active_mask(X).sum()),
        acc_pool=float((R.argmax(1) == y).mean()),
        perp_pool=perplexity_of(p_pool),
    )
    if "score_test" in d:
        p_learned = softmax_probs(d["score_test"])
        rec["acc_learned"] = float((d["score_test"].argmax(1) == y).mean())
        rec["perp_learned"] = perplexity_of(p_learned)
    if "probe_test" in d:
        rec["acc_probe"] = float((d["probe_test"].argmax(1) == y).mean())
        rec["perp_probe"] = perplexity_of(d["probe_test"])
    # entropy AUROC on the readout that can carry the claim
    unc = res.get("uncertainty", [])
    for r in unc:
        if r["readout"] == "learned_readout":
            for s in r["statistics"]:
                if s["statistic"] == "entropy":
                    rec["auroc_learned"] = s["auroc"]
                    rec["cov95_learned"] = s["cov_at_acc"]["0.95"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cells, data = [], {}
    for name in ORDER:
        r = load_cell(os.path.join(a.run_dir, name))
        if r is not None:
            cells.append(name); data[name] = r
    if not cells:
        print("no completed cells yet"); return
    print(f"loaded {len(cells)}/{len(ORDER)} cells: {', '.join(cells)}\n")

    hdr = (f"{'cell':<13} {'corr_in':>8} {'PR':>6} {'n_act':>6} "
           f"{'acc_pool':>9} {'acc_learn':>10} {'acc_probe':>10} "
           f"{'perp_pool':>10} {'perp_lrn':>9} {'AUROC':>7} {'cov@95':>7}")
    print(hdr)
    for n in cells:
        r = data[n]
        g = lambda k: r.get(k, float("nan"))
        print(f"{n:<13} {np.mean(r['corr_groups']):>8.3f} {r['pr']:>6.1f} {r['n_active']:>6d} "
              f"{r['acc_pool']:>9.3f} {g('acc_learned'):>10.3f} {g('acc_probe'):>10.3f} "
              f"{r['perp_pool']:>10.2f} {g('perp_learned'):>9.2f} "
              f"{g('auroc_learned'):>7.3f} {g('cov95_learned'):>7.3f}")

    xs = np.arange(len(cells))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. redundancy — boxplot over the 10 class groups
    ax = axes[0, 0]
    ax.boxplot([data[n]["corr_groups"] for n in cells], positions=xs, widths=0.55,
               patch_artist=True,
               boxprops=dict(facecolor="#CECBF6", edgecolor="#534AB7"),
               medianprops=dict(color="#26215C", lw=2))
    for i, n in enumerate(cells):
        v = data[n]["corr_groups"]
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.12, .12, len(v)), v,
                   s=14, color="#3C3489", alpha=0.55, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("|correlation| within class group")
    ax.set_title("Redundancy per class group — LOWER is better\n(box = 10 class groups)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # 2. dimensionality
    ax = axes[0, 1]
    ax.bar(xs, [data[n]["pr"] for n in cells], color="#1D9E75", width=0.6)
    ax.axhline(9, ls="--", lw=1.4, color="k")
    ax.text(0.02, 9.3, "rank needed for 10 classes", fontsize=9,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(xs); ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("participation ratio")
    ax.set_title("Effective dimensionality — HIGHER is better", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # 3. perplexity per readout
    ax = axes[1, 0]
    w = 0.27
    for off, key, lab, col in ((-w, "perp_pool", "pool (uniform)", C_POOL),
                               (0.0, "perp_learned", "learned readout", C_LEARNED),
                               (w, "perp_probe", "linear probe (control)", C_PROBE)):
        ax.bar(xs + off, [data[n].get(key, np.nan) for n in cells], width=w,
               color=col, label=lab)
    ax.axhline(10, ls=":", lw=1, color="k")
    ax.text(0.02, 10.2, "chance (10 classes)", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(xs); ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("perplexity"); ax.legend(fontsize=8)
    ax.set_title("Predictive perplexity — LOWER = sharper\n(effective classes still in play)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # 4. accuracy per readout
    ax = axes[1, 1]
    for off, key, lab, col in ((-w, "acc_pool", "pool (uniform)", C_POOL),
                               (0.0, "acc_learned", "learned readout", C_LEARNED),
                               (w, "acc_probe", "linear probe (control)", C_PROBE)):
        ax.bar(xs + off, [data[n].get(key, np.nan) for n in cells], width=w,
               color=col, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8, loc="lower right")
    ax.axhline(0.1, ls=":", lw=0.8, color="k")
    ax.set_title("Accuracy by readout — the probe is a CONTROL,\nnot something the network does",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("RF geometry: factorizing thin-oriented / centered-tiles / no-WTA", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = a.out or os.path.join(a.run_dir, "rfgeom_panels.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved -> {out}")

    # Per-CLASS redundancy. The boxplot above shows the spread over class groups but
    # anonymously, so it cannot answer "which digit is the problem?" — and that matters:
    # class 0 was reported as the success case (several neurons covering it at different
    # angles) while others collapse onto one preference. Rows = config, cols = digit.
    M = np.full((len(cells), 10), np.nan)
    for i, n in enumerate(cells):
        g = data[n]["corr_groups"]
        M[i, :len(g)] = g
    fig2, ax = plt.subplots(figsize=(9, 0.62 * len(cells) + 2.4))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=np.nanmin(M), vmax=np.nanmax(M), aspect="auto")
    ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
    ax.set_yticks(range(len(cells))); ax.set_yticklabels(cells)
    ax.set_xlabel("digit class"); ax.set_title(
        "Within-class redundancy: |correlation| among the 100 neurons of each class\n"
        "green = diverse (neurons cover the class differently), red = collapsed onto one preference",
        fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="black")
    fig2.colorbar(im, ax=ax, label="|correlation| within class")
    fig2.tight_layout()
    out2 = os.path.join(a.run_dir, "rfgeom_per_class.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print(f"saved -> {out2}")

    print(f"\n{'cell':<13} " + " ".join(f"{c:>5}" for c in range(10)) + "   mean")
    for i, n in enumerate(cells):
        print(f"{n:<13} " + " ".join(f"{M[i, j]:>5.2f}" for j in range(10))
              + f"   {np.nanmean(M[i]):.3f}")


if __name__ == "__main__":
    main()
