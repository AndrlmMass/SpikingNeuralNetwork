"""
SKETCH: input sheet facing the reader, excitatory sheet behind it, one filled cone.

Geometry follows the Diehl-and-Cook style layout: the input plane stands in front, tilted
about its vertical axis, and the excitatory plane sits behind and to the right. The input
plane is drawn semi-transparent so the sheet behind stays visible through it.

A real MNIST digit fills the input plane; the receptive field of ONE excitatory neuron is
the small red patch on it, and the filled cone carries that patch back to the single
neuron that reads it. The point of the figure is the size relation -- one neuron sees a
small, elongated, oriented fragment of the image, not the whole thing.

Three drafts varying tilt and separation.

    python experiments/RF_article/interp/sketch_rf_cone.py
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from neurosnn._network.init_weights import oriented_gaussian_se_weights  # noqa: E402

G = 28
CMAP = plt.get_cmap("OrRd")
INK, EDGE, MUTED = "#141414", "#8f8f8f", "#555555"   # sheet frame kept light


def digit():
    from neurosnn._data.get_data import ImageDataStreamer
    s = ImageDataStreamer(data_dir=os.path.join(REPO, "data"), pixel_size=G,
                          dataset="mnist")
    imgs = np.asarray(s.train_images)
    lab = np.asarray(getattr(s, "train_labels", np.zeros(len(imgs))))
    k = int(np.flatnonzero(lab == 3)[0]) if (lab == 3).any() else 0
    im = imgs[k].reshape(G, G)
    return im / (im.max() or 1.0)


def rf_patch(cx, cy, orient=1):
    W = oriented_gaussian_se_weights(
        N_x=G * G, N_exc=1, input_size=G, sigma_x=3.0, gamma=1.2 / 3.0,
        n_orientations=4, orientation_mode="block", peak=1.0, fraction=1.0,
        rng=np.random.default_rng(0), centers=np.array([[cy, cx]], dtype=float),
        orientation_idx=np.array([orient]))
    Z = W[:, 0].reshape(G, G)
    return Z / Z.max()


def make_proj(dx, dy):
    """(u, v, plane) -> page. Sheets stay perfectly square; depth is the offset alone.

    Shearing the sheets to fake perspective distorted the digit and the RF along with
    them. Two identical squares, the second displaced diagonally, reads as depth without
    deforming anything drawn on either one.
    """
    def proj(u, v, plane):
        return (u + plane * dx, v + plane * dy)
    return proj


GRIDC = "#dcdcdc"


def rule(ax, p, n, plane, step=2, z=1.5):
    """Light grid on a sheet, so the plane reads as a surface rather than blank paper."""
    for t in range(0, n + 1, step):
        ax.plot(*zip(p(t, 0, plane), p(t, n, plane)), color=GRIDC, lw=0.6, zorder=z)
        ax.plot(*zip(p(0, t, plane), p(n, t, plane)), color=GRIDC, lw=0.6, zorder=z)


def quad(ax, p, u0, u1, v0, v1, plane, **kw):
    ax.add_patch(Polygon([p(u0, v0, plane), p(u1, v0, plane),
                          p(u1, v1, plane), p(u0, v1, plane)], closed=True, **kw))


def draw(ax, dx, dy, title):
    p = make_proj(dx, dy)
    img = digit()
    cx, cy = 15.5, 15.0                     # RF centre, on a stroke of the digit
    Z = rf_patch(cx, cy)

    # --- back sheet: excitatory layer, drawn first so the front overlays it ---
    quad(ax, p, 0, G, 0, G, 1, facecolor="white", edgecolor=EDGE, lw=1.3,
         alpha=0.92, zorder=1)
    rule(ax, p, G, 1, z=1.5)
    ax.text(*p(G * 0.5, G + 1.2, 1), "postsynaptic", ha="center", va="bottom",
            fontsize=12, color=MUTED)
    npos = p(cx, cy, 1)

    # --- cone: from the RF's mouth on the input sheet back to the one neuron -------
    ii, jj = np.nonzero(Z > 0.28)
    box = [p(jj.min(), ii.min(), 0), p(jj.max() + 1, ii.min(), 0),
           p(jj.max() + 1, ii.max() + 1, 0), p(jj.min(), ii.max() + 1, 0)]
    for k in range(4):
        ax.add_patch(Polygon([box[k], box[(k + 1) % 4], npos], closed=True,
                             facecolor=CMAP(0.62), alpha=0.17, edgecolor="none",
                             zorder=2))

    # --- front sheet: the input image, semi-transparent so the back stays visible --
    for i in range(G):
        for j in range(G):
            g = img[i, j]
            if g < 0.04:
                continue
            quad(ax, p, j, j + 1, G - i - 1, G - i, 0,
                 facecolor=str(1.0 - 0.85 * g), edgecolor="none", alpha=0.55, zorder=3)
    for i in range(G):                       # the RF, red, over the digit
        for j in range(G):
            w = Z[i, j]
            if w < 0.18:
                continue
            quad(ax, p, j, j + 1, G - i - 1, G - i, 0,
                 facecolor=CMAP(0.30 + 0.70 * w), edgecolor="none",
                 alpha=0.60 + 0.22 * w, zorder=4)
    rule(ax, p, G, 0, z=2.6)
    quad(ax, p, 0, G, 0, G, 0, facecolor="none", edgecolor=EDGE, lw=1.4, zorder=5)
    ax.text(*p(G * 0.5, -1.4, 0), "presynaptic", ha="center", va="top",
            fontsize=12, color=MUTED)

    # fill the grid square that receives the projection, rather than marking it with a
    # dot: the neuron occupies a cell of the postsynaptic sheet, and a filled cell says so
    gu, gv = int(cx // 2) * 2, int(cy // 2) * 2
    ctr = p(gu + 1, gv + 1, 1)
    # Semi-transparent: drawn fully opaque it read as sitting in front of everything,
    # which fights the whole point of putting the sheet behind.
    ax.add_patch(Circle(ctr, 1.15, facecolor=CMAP(0.52), edgecolor=CMAP(0.72),
                        lw=1.1, alpha=0.55, zorder=6))
    if title:
        ax.set_title(title, fontsize=13, loc="left", color=MUTED)
    ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", action="store_true",
                    help="render the side-by-side drafts instead of the final figure")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.drafts:
        cfgs = [(9.0, 7.0, "A  close"), (13.0, 10.0, "B  medium")]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor="white")
        for ax, (dx, dy, name) in zip(axes, cfgs):
            draw(ax, dx, dy, name)
        out = a.out or os.path.join(REPO, "results", "figures", "sketch_rf_cone")
    else:
        # the chosen layout: sheets offset by (13, 10). The neuron falls beyond the
        # presynaptic sheet's edge here, so nothing is ambiguous about which sheet it
        # sits on -- the reason this spacing was picked over the closer one.
        fig, ax = plt.subplots(figsize=(7.4, 7.0), facecolor="white")
        draw(ax, 13.0, 10.0, None)
        out = a.out or os.path.join(REPO, "results", "figures", "rf_cone")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("[figure]", out + ".png (+ .pdf)")


if __name__ == "__main__":
    main()
