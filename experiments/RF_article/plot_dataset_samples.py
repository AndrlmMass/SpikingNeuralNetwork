"""
Appendix figure: one training image per class for each of the four datasets.

Images are pulled through the SAME transform the model sees (grayscale, resized to
28x28, scaled to [0,1]) -- so what the figure shows is what the network is encoded from,
not the datasets' native formats.

Usage:
    python experiments/RF_article/plot_dataset_samples.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)   # the notMNIST deeplake copy is looked up by relative path

from neurosnn._data.get_data import ImageDataStreamer  # noqa: E402

DATASETS = [("mnist", "MNIST"), ("kmnist", "KMNIST"),
            ("fmnist", "Fashion-MNIST"), ("notmnist", "notMNIST")]
# Column headers: the class index is shared, but its meaning is not.
CLASS_NAMES = {
    "mnist":    [str(i) for i in range(10)],
    "kmnist":   ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
    "fmnist":   ["T-shirt", "trouser", "pullover", "dress", "coat",
                 "sandal", "shirt", "sneaker", "bag", "boot"],
    "notmnist": list("ABCDEFGHIJ"),
}


def first_per_class(ds):
    """(10, 28, 28) array: the first training image of each class label 0..9."""
    s = ImageDataStreamer(data_dir=os.path.join(REPO, "data"), pixel_size=28, dataset=ds)
    imgs = s.train_images
    labels = np.asarray(s.train_labels)
    out = []
    for k in range(10):
        idx = int(np.flatnonzero(labels == k)[0])
        im = np.asarray(imgs[idx]).squeeze()
        out.append(im)
    return np.stack(out)


def figure(rows, out):
    nds, ncls = len(rows), 10
    fig, axes = plt.subplots(nds, ncls, figsize=(ncls * 0.78, nds * 0.95))
    for r, (ds, pretty, imgs) in enumerate(rows):
        for c in range(ncls):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#999999")
            ax.set_xlabel(CLASS_NAMES[ds][c], fontsize=6.5, labelpad=1.5)
            if c == 0:
                ax.set_ylabel(pretty, fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=6)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.98, bottom=0.02,
                        wspace=0.08, hspace=0.32)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = []
    for ds, pretty in DATASETS:
        try:
            rows.append((ds, pretty, first_per_class(ds)))
            print(f"[ok] {ds}")
        except Exception as exc:
            print(f"[skip] {ds}: {exc}")
    out = os.path.join(REPO, "results", "figures", "dataset_samples")
    figure(rows, out)
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    main()
