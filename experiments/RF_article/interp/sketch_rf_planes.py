"""
SKETCH: two flat planes, one postsynaptic neuron, RF highlighted in red by weight.

Drawn as 2D polygons under a hand-rolled projection rather than with matplotlib's 3D
renderer, which is what made the earlier sketches look cluttered: flat planes in
perspective read as an architecture diagram, a 3D surface reads as a landscape.

Four drafts, differing only in how the projection from presynaptic sheet to neuron is
drawn. Weights come from the model's own W_se column, so the red patch is the real RF.

    python experiments/RF_article/interp/sketch_rf_planes.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from neurosnn._network.init_weights import oriented_gaussian_se_weights  # noqa: E402

G, PAD = 28, 7
CMAP = plt.get_cmap("OrRd")
INK, EDGE, MUTED = "#1a1a1a", "#333333", "#5a5a5a"
SKEW = np.array([[1.0, 0.42], [0.0, 0.62]])     # plane -> page (a sheared square)


def rf():
    W = oriented_gaussian_se_weights(
        N_x=G * G, N_exc=1, input_size=G, sigma_x=3.0, gamma=1.2 / 3.0,
        n_orientations=4, orientation_mode="block", peak=1.0, fraction=1.0,
        rng=np.random.default_rng(0), centers=np.array([[G / 2., G / 2.]]),
        orientation_idx=np.array([0]))
    c = G // 2
    Z = W[:, 0].reshape(G, G)[c - PAD:c + PAD, c - PAD:c + PAD]
    return Z / Z.max()


def proj(u, v, y):
    """Plane coords (u, v) on a sheet at height y -> page coords."""
    p = SKEW @ np.array([u, v], dtype=float)
    return p[0], p[1] + y


def sheet(ax, y, n, label, face="white"):
    corners = [proj(0, 0, y), proj(n, 0, y), proj(n, n, y), proj(0, n, y)]
    ax.add_patch(Polygon(corners, closed=True, facecolor=face,
                         edgecolor=EDGE, lw=1.4, zorder=2))
    ax.text(*proj(-0.8, n * 0.5, y), label, ha="right", va="center",
            fontsize=11, color=MUTED)
    return corners


def cells(ax, Z, y, thresh=0.04):
    """The RF, flat on the sheet, shaded by weight."""
    n = Z.shape[0]
    for i in range(n):
        for j in range(n):
            w = Z[i, j]
            if w < thresh:
                continue
            q = [proj(j, i, y), proj(j + 1, i, y), proj(j + 1, i + 1, y), proj(j, i + 1, y)]
            ax.add_patch(Polygon(q, closed=True, facecolor=CMAP(0.20 + 0.80 * w),
                                 edgecolor="none", zorder=3))


def main():
    Z = rf()
    n = Z.shape[0]
    y0, y1 = 0.0, 9.5
    npos = proj(n * 0.5, n * 0.5, y1)
    # the RF's bounding quad, used as the mouth of the projection cone
    # mouth of the cone = the cells that actually READ as the RF, not every non-zero
    # weight; a loose bounding box made the cone land wide of the red patch
    ii, jj = np.nonzero(Z > 0.25)
    box = [proj(jj.min(), ii.min(), y0), proj(jj.max() + 1, ii.min(), y0),
           proj(jj.max() + 1, ii.max() + 1, y0), proj(jj.min(), ii.max() + 1, y0)]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), facecolor="white")
    titles = ["A  filled cone", "B  outline cone", "C  spokes", "D  cone + spokes"]
    # presynaptic sheet is drawn first (lowest), postsynaptic last (highest)
    for ax, t in zip(axes, titles):
        sheet(ax, y0, n, "presynaptic")
        cells(ax, Z, y0)
        if t.startswith("A") or t.startswith("D"):
            for k in range(4):
                ax.add_patch(Polygon([box[k], box[(k + 1) % 4], npos], closed=True,
                                     facecolor=CMAP(0.55), alpha=0.16,
                                     edgecolor="none", zorder=4))
        if t.startswith("B"):
            for c in box:
                ax.plot([c[0], npos[0]], [c[1], npos[1]], color=CMAP(0.75),
                        lw=1.1, alpha=0.9, zorder=4)
        if t.startswith("C") or t.startswith("D"):
            for i in range(0, n, 2):
                for j in range(0, n, 2):
                    w = Z[i, j]
                    if w < 0.20:
                        continue
                    p = proj(j + .5, i + .5, y0)
                    ax.plot([p[0], npos[0]], [p[1], npos[1]], color=CMAP(0.35 + 0.6 * w),
                            lw=0.4 + 1.1 * w, alpha=0.25 + 0.55 * w, zorder=4)
        # postsynaptic sheet last and UNFILLED, so the projection stays visible through it
        sheet(ax, y1, n, "postsynaptic", face="none")
        ax.scatter([npos[0]], [npos[1]], s=90, c=INK, zorder=6,
                   edgecolors="white", linewidths=1.4)
        ax.set_title(t, fontsize=13, loc="left", color=MUTED)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-2, n * 1.55); ax.set_ylim(-3, y1 + n * 0.75)

    fig.tight_layout()
    out = os.path.join(REPO, "results", "figures", "sketch_rf_planes")
    fig.savefig(out + ".png", dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[sketch]", out + ".png")


if __name__ == "__main__":
    main()
