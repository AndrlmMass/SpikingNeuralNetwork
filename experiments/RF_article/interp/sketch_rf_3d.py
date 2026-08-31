"""
SKETCH: four ways to draw one elongated, oriented receptive field in 3D, black and white.

Not a finished figure -- four candidate treatments side by side so we can pick one and
then build it properly. Every panel shows the SAME neuron's real W_se column, pulled from
the model's own weight builder rather than an idealized Gabor drawn by hand, so what is on
screen is what the network actually has: an elliptical Gaussian with sigma along the bar
set by `rf_length` and sigma across it by `rf_thickness`, hard-cut at r_cut_factor * sigma,
and sparsified to `fraction` of the input pixels.

  A  surface        the energy-landscape reading. Height is synaptic weight; the ridge
                    IS the preferred orientation. Most immediate, least quantitative.
  B  layered slices the "layer by layer" idea: iso-weight contours lifted to their own
                    heights. Reads as a contour map with depth, and the spacing between
                    rings shows how sharply the field falls off.
  C  wireframe      the same surface as a mesh. Anisotropy is legible directly from the
                    grid lines, and it prints cleanly at small size with no gray fills.
  D  stem / lollipop one stem per non-zero synapse. The only option that does not imply a
                    continuous field -- it shows that the RF is a SPARSE set of weighted
                    connections, which is what the model actually stores.

Grayscale throughout, as asked. Where a colormap is needed it is "Greys" with the light
end reserved for zero, so a printed figure keeps the background white.

    python experiments/RF_article/interp/sketch_rf_3d.py
"""
import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from neurosnn._network.init_weights import oriented_gaussian_se_weights  # noqa: E402

# phase-2 RF geometry: sigma along the bar 3.0 px, across it 1.2 px
RF_LENGTH, RF_THICKNESS, GRID = 3.0, 1.2, 28
FS = 15


def real_rf(neuron=0, orientation=0):
    """One column of the model's own W_se, reshaped to the 28x28 input grid."""
    n_exc = 4
    W = oriented_gaussian_se_weights(
        N_x=GRID * GRID, N_exc=n_exc, input_size=GRID,
        sigma_x=RF_LENGTH, gamma=RF_THICKNESS / RF_LENGTH,
        n_orientations=4, orientation_mode="block", peak=1.0, fraction=1.0,
        rng=np.random.default_rng(0),
        centers=np.tile(np.array([[GRID / 2, GRID / 2]]), (n_exc, 1)),
        orientation_idx=np.arange(n_exc) % 4)
    return W[:, orientation].reshape(GRID, GRID)


def crop(Z, pad=7):
    c = GRID // 2
    return Z[c - pad:c + pad, c - pad:c + pad]


def main():
    # orientation 0 (bar along the x axis), viewed from across the ridge. Orientation 1
    # is the 45-degree field, and at the default azimuth the camera looks straight ALONG
    # its ridge, which hides the very elongation the figure exists to show.
    Z = crop(real_rf(orientation=0))
    n = Z.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    fig = plt.figure(figsize=(17, 4.6), facecolor="white")

    # ---- A: surface ------------------------------------------------------
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="Greys", vmin=0, vmax=Z.max() * 1.15,
                    rstride=1, cstride=1, linewidth=0.25, edgecolors="0.4",
                    antialiased=True, shade=False)
    ax.set_title("A  surface", fontsize=FS, loc="left", color="0.25")

    # ---- B: layered slices ----------------------------------------------
    # each iso-weight contour drawn at its own height, so the stack reads as depth
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    levels = np.linspace(Z.max() * 0.08, Z.max() * 0.95, 7)
    for lv in levels:
        ax.contour(X, Y, Z, levels=[lv], zdir="z", offset=lv,
                   colors="0.15", linewidths=1.4)
    ax.contourf(X, Y, Z, levels=12, zdir="z", offset=0, cmap="Greys", alpha=0.30)
    ax.set_zlim(0, Z.max() * 1.05)
    ax.set_title("B  layered slices", fontsize=FS, loc="left", color="0.25")

    # ---- C: wireframe ----------------------------------------------------
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="0.2", linewidth=0.6)
    ax.set_title("C  wireframe", fontsize=FS, loc="left", color="0.25")

    # ---- D: stems, one per synapse --------------------------------------
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    m = Z > Z.max() * 0.02
    xs, ys, zs = X[m], Y[m], Z[m]
    for xi, yi, zi in zip(xs, ys, zs):
        ax.plot([xi, xi], [yi, yi], [0, zi], color="0.55", lw=0.7, zorder=1)
    ax.scatter(xs, ys, zs, s=16, c="0.1", depthshade=False, zorder=2)
    ax.set_title("D  stems (one per synapse)", fontsize=FS, loc="left", color="0.25")

    for ax in fig.axes:
        ax.view_init(elev=34, azim=-72)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        ax.grid(False)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_visible(False)

    fig.tight_layout()
    out = os.path.join(REPO, "results", "figures", "sketch_rf_3d")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"RF grid {Z.shape}, peak {Z.max():.3f}, non-zero {(Z > 0).sum()} px")
    print(f"[sketch] wrote {out}.png")


if __name__ == "__main__":
    main()
