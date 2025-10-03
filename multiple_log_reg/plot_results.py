import os
import numpy as np
import matplotlib.pyplot as plt


def plot_grouped_accuracy(results, title="Accuracy (train/val/test)", out_path=None):
    modalities = list(results.keys())
    splits = ["train", "val", "test"]

    x = np.arange(len(modalities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "train": "#ffbfb3",  # soft peach
        "val": "#6afae9",  # soft mint
        "test": "#cdb4db",  # pastel lavender (better contrast)
    }

    for i, split in enumerate(splits):
        vals = [results[m][split] for m in modalities]
        rects = ax.bar(
            x + (i - 1) * width, vals, width, label=split, color=colors[split]
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "grouped_accuracy.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Replace with your latest printed results if they change
    results = {
        "image": {"train": 0.9187, "val": 0.9082, "test": 0.9128},
        "audio": {"train": 0.7646, "val": 0.7577, "test": 0.7508},
        "combined": {"train": 0.9765, "val": 0.9672, "test": 0.9672},
    }

    plot_grouped_accuracy(
        results,
        title="Image vs Audio vs Combined — Logistic Regression",
        out_path=os.path.join(os.path.dirname(__file__), "grouped_accuracy.png"),
    )
