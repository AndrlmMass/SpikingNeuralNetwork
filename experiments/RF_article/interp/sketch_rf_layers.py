"""
SKETCH: four ways to draw the RF as a CONNECTIVITY picture -- presynaptic plane to
postsynaptic neuron -- rather than as a standalone surface.

The surface sketches answer "what shape is the field". These answer the different question
"what does one postsynaptic neuron see in presynaptic space", which needs both layers in
the frame and the projection between them drawn explicitly.

  A  fan            input plane carrying the RF as a heatmap, one line per synapse rising
                    to the neuron above. Line weight tracks synaptic weight, so the cone
                    thins toward the edge of the field the way the weights do.
  B  plane + relief the same fan with the weight profile lifted OFF the plane as a surface,
                    so the oval footprint and the weight profile are both visible at once
                    instead of the reader inferring one from the other.
  C  population     three neurons at different preferred orientations, each with its own
                    cone onto the same input plane. This is the only panel that shows the
                    thing the architecture actually relies on: overlapping fields tiling
                    the input at several orientations.
  D  exploded       input image, then the weight mask, then their product, as parallel
                    planes -- the literal layer-by-layer reading of what the neuron
                    integrates. Closest to a textbook wiring diagram.

Greyscale geometry with the project's OrRd ramp for weight, so it matches the RF heatmaps
elsewhere in the paper.

    python experiments/RF_article/interp/sketch_rf_layers.py
"""
import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from neurosnn._network.init_weights import oriented_gaussian_se_weights  # noqa: E402

RF_LENGTH, RF_THICKNESS, GRID = 3.0, 1.2, 28
CMAP = cm.get_cmap("OrRd") if hasattr(cm, "get_cmap") else plt.get_cmap("OrRd")
INK, MUTED = "#1a1a1a", "#6b6b6b"
FS = 14
Z_NEURON = 1.0          # height of the postsynaptic neuron above the input plane


def rf_bank(centers, orientations):
    """Real W_se columns for the given centres and orientation indices."""
    n = len(orientations)
    W = oriented_gaussian_se_weights(
        N_x=GRID * GRID, N_exc=n, input_size=GRID,
        sigma_x=RF_LENGTH, gamma=RF_THICKNESS / RF_LENGTH,
        n_orientations=4, orientation_mode="block", peak=1.0, fraction=1.0,
        rng=np.random.default_rng(0), centers=np.asarray(centers, dtype=float),
        orientation_idx=np.asarray(orientations))
    return [W[:, k].reshape(GRID, GRID) for k in range(n)]


def crop_box(pad=7):
    c = GRID // 2
    return c - pad, c + pad


def draw_plane(ax, Z, lo, hi, z0=0.0, alpha=0.95):
    """The presynaptic sheet, shaded by synaptic weight."""
    sub = Z[lo:hi, lo:hi]
    n = sub.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    norm = sub / (sub.max() or 1.0)
    ax.plot_surface(X, Y, np.full_like(sub, z0), rstride=1, cstride=1,
                    facecolors=CMAP(0.15 + 0.85 * norm), shade=False,
                    linewidth=0, antialiased=False, alpha=alpha, zorder=1)
    return X, Y, sub


def draw_fan(ax, X, Y, sub, every=2, z1=Z_NEURON):
    """One line per synapse, opacity and width tracking the weight."""
    m = sub > sub.max() * 0.06
    xs, ys, ws = X[m], Y[m], sub[m]
    cx, cy = X.mean(), Y.mean()
    order = np.argsort(ws)
    for i in order[::every]:
        w = ws[i] / ws.max()
        ax.plot([xs[i], cx], [ys[i], cy], [0.02, z1],
                color=CMAP(0.35 + 0.65 * w), lw=0.4 + 1.6 * w,
                alpha=0.25 + 0.6 * w, zorder=2)
    ax.scatter([cx], [cy], [z1], s=150, c=[INK], marker="o",
               edgecolors="white", linewidths=1.6, zorder=5)


def style(ax, title, zmax=1.25):
    ax.set_title(title, fontsize=FS, loc="left", color=MUTED, pad=2)
    ax.set_zlim(0, zmax)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.view_init(elev=26, azim=-68)


def main():
    lo, hi = crop_box()
    c = GRID / 2
    fig = plt.figure(figsize=(15.5, 12.0), facecolor="white")

    # ---- A: fan ----------------------------------------------------------
    Z0 = rf_bank([[c, c]], [0])[0]
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    X, Y, sub = draw_plane(ax, Z0, lo, hi)
    draw_fan(ax, X, Y, sub)
    style(ax, "A  fan: one line per synapse")

    # ---- B: plane + relief ----------------------------------------------
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    X, Y, sub = draw_plane(ax, Z0, lo, hi, alpha=0.75)
    rel = 0.55 * sub / sub.max()
    ax.plot_surface(X, Y, rel, rstride=1, cstride=1, cmap="OrRd",
                    vmin=0, vmax=rel.max() * 1.1, linewidth=0.2,
                    edgecolors="0.55", shade=False, alpha=0.95, zorder=3)
    ax.scatter([X.mean()], [Y.mean()], [Z_NEURON], s=150, c=[INK], marker="o",
               edgecolors="white", linewidths=1.6, zorder=5)
    ax.plot([X.mean(), X.mean()], [Y.mean(), Y.mean()], [rel.max(), Z_NEURON],
            color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=4)
    style(ax, "B  plane + weight relief")

    # ---- C: population, three orientations ------------------------------
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    offs = [(-3.5, -2.0, 0), (0.0, 0.0, 1), (3.5, 2.0, 2)]
    bank = rf_bank([[c + dy, c + dx] for dx, dy, _ in offs], [o for _, _, o in offs])
    # one plane carrying all three fields, so the overlap is visible
    tot = np.maximum.reduce(bank)
    X, Y, sub = draw_plane(ax, tot, lo, hi, alpha=0.95)
    for (dx, dy, _), Zk in zip(offs, bank):
        s = Zk[lo:hi, lo:hi]
        m = s > s.max() * 0.25
        xs, ys, ws = X[m], Y[m], s[m]
        nx, ny = X.mean() + dx, Y.mean() + dy
        for i in np.argsort(ws)[::4]:
            w = ws[i] / ws.max()
            ax.plot([xs[i], nx], [ys[i], ny], [0.02, Z_NEURON],
                    color=CMAP(0.4 + 0.6 * w), lw=0.5 + 1.0 * w,
                    alpha=0.2 + 0.5 * w, zorder=2)
        ax.scatter([nx], [ny], [Z_NEURON], s=130, c=[INK], marker="o",
                   edgecolors="white", linewidths=1.5, zorder=5)
    style(ax, "C  population: three orientations onto one input")

    # ---- D: exploded layers ---------------------------------------------
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    n = hi - lo
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    # a stand-in "image": a bar the neuron is tuned to, plus the weights, plus the product
    img = np.zeros((n, n))
    img[n // 2 - 1:n // 2 + 2, 2:n - 2] = 1.0
    w = Z0[lo:hi, lo:hi] / Z0[lo:hi, lo:hi].max()
    for z0, layer, lab in ((0.0, img, "input"), (0.45, w, "weights"),
                           (0.90, img * w, "product")):
        ax.plot_surface(X, Y, np.full_like(layer, z0), rstride=1, cstride=1,
                        facecolors=CMAP(0.12 + 0.88 * layer / (layer.max() or 1)),
                        shade=False, linewidth=0, antialiased=False,
                        alpha=0.95, zorder=int(z0 * 10))
        ax.text(-2.0, n * 0.5, z0 + 0.06, lab, fontsize=FS - 2, color=MUTED,
                ha="right", va="center", zorder=6)
    ax.scatter([X.mean()], [Y.mean()], [1.30], s=150, c=[INK], marker="o",
               edgecolors="white", linewidths=1.6, zorder=7)
    style(ax, "D  exploded: input x weights -> neuron", zmax=1.45)

    fig.tight_layout()
    out = os.path.join(REPO, "results", "figures", "sketch_rf_layers")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[sketch] wrote {out}.png")


if __name__ == "__main__":
    main()
