import os
import numpy as np
import matplotlib.pyplot as plt


def preview_loaded_data(
    self, dataset, num_image_samples: int = 9, save_path: str | None = None
):
    """
    Plot a small grid of images from the loaded dataset once so the user can
    verify that the expected dataset is being used.
    """
    # Special handling for geomfig: show N examples per class (0..3)
    if dataset.lower() == "geomfig":
        from get_data import _geomfig_generate_one  # type: ignore

        try:
            classes = [0, 1, 2, 3]
            per_class = max(1, int(num_image_samples))
            # Special case: if 1 per class and 4 classes, arrange as 2x2 instead of 4x1
            if per_class == 1 and len(classes) == 4:
                rows, cols = 2, 2
                fig, axes = plt.subplots(rows, cols, figsize=(4.0, 4.0))
                axes = axes.flatten()
                titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                for idx, cls in enumerate(classes):
                    img = _geomfig_generate_one(
                        cls_id=cls,
                        pixel_size=self.pixel_size,
                        noise_var=getattr(self, "geom_noise_var", 0.02),
                        jitter=getattr(self, "geom_jitter", False),
                        jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                    )
                    ax = axes[idx]
                    ax.imshow(img, cmap="gray")
                    ax.set_title(titles[idx], fontsize=10)
                    ax.axis("off")
            else:
                fig, axes = plt.subplots(
                    len(classes),
                    per_class,
                    figsize=(2.0 * per_class, 2.0 * len(classes)),
                )
                if per_class == 1:
                    axes = np.atleast_2d(axes).reshape(len(classes), 1)
                titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                for r, cls in enumerate(classes):
                    for c in range(per_class):
                        img = _geomfig_generate_one(
                            cls_id=cls,
                            pixel_size=self.pixel_size,
                            noise_var=getattr(self, "geom_noise_var", 0.02),
                            jitter=getattr(self, "geom_jitter", False),
                            jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                        )
                        ax = axes[r, c]
                        ax.imshow(img, cmap="gray")
                        if c == 0:
                            ax.set_title(titles[r], fontsize=10)
                        ax.axis("off")
            plt.tight_layout()
            try:
                if save_path is None:
                    os.makedirs("plots", exist_ok=True)
                    save_path = os.path.join("plots", "geomfig_preview.png")
                fig.savefig(save_path)
                print(f"Dataset preview saved to {save_path}")
            except Exception as exc:
                print(f"Failed to save dataset preview ({exc})")
            plt.show()
            plt.close(fig)
        except Exception as exc:
            print(f"Dataset preview skipped ({exc})")
        finally:
            self._image_preview_done = True
        return
    # Default: use image streamer preview if available
    if not hasattr(self, "image_streamer") or self.image_streamer is None:
        return
    try:
        self.image_streamer.show_preview(
            num_samples=num_image_samples, save_path=save_path
        )
    except Exception as exc:  # pragma: no cover - visualization fallback
        print(f"Dataset preview skipped ({exc})")
    finally:
        self._image_preview_done = True


def plot_accuracy(self, wta, mcc, phi, pca, acc_log_file, read_jsonl):
    import pandas as pd

    records = read_jsonl(acc_log_file)

    # Collect series keyed by epoch and method (JSONL may have multiple lines per epoch)
    if pca:
        train_acc_pca = {}
        val_acc_pca = {}
    if wta:
        train_acc_top = {}
        val_acc_top = {}
    if phi:
        val_phi = {}
        train_phi = {}
    if mcc:
        train_mcc = {}
        val_mcc = {}

    for r in records:
        epoch = r.get("epoch")
        if epoch is None:
            continue

        split = r.get("split")
        method = r.get("method")

        if "accuracy" in r and r["accuracy"] is not None:
            acc_val = float(r["accuracy"])
            if split == "train":
                if method == "pca_lr" and pca:
                    train_acc_pca[int(epoch)] = acc_val
                elif method == "top" and wta:
                    train_acc_top[int(epoch)] = acc_val
            elif split == "val":
                if method == "pca_lr" and pca:
                    val_acc_pca[int(epoch)] = acc_val
                elif method == "top" and wta:
                    val_acc_top[int(epoch)] = acc_val

        if "phi" in r and r["phi"] is not None:
            # Based on your log format: phi is recorded under split="val"
            if split == "val" and phi:
                val_phi[int(epoch)] = float(r["phi"])
            if split == "train" and phi:
                train_phi[int(epoch)] = float(r["phi"])

        if "mcc" in r and r["mcc"] is not None:
            # Based on your log format: mcc is recorded under split="val"
            if split == "val" and mcc:
                val_mcc[int(epoch)] = float(r["mcc"])
            if split == "train" and mcc:
                train_mcc[int(epoch)] = float(r["mcc"])

    fig, (ax, ax2) = plt.subplots(2, 1)

    handles = []
    labels = []
    labels2 = []
    handles2 = []

    # if train_acc_pca:
    #     xs = sorted(train_acc_pca)
    #     train_acc = np.asarray([train_acc_pca[e] for e in xs])
    #     line = ax.plot(
    #         xs,
    #         train_acc,
    #         label="train acc (pca_lr)",
    #         linestyle="-",
    #         marker="o",
    #         markersize=3,
    #         color="gray",
    #     )
    #     handles.append(line[0])
    #     labels.append("Train accuracy")
    if pca:
        xs = sorted(val_acc_pca)
        val_acc = np.asarray([val_acc_pca[e] for e in xs])
        line = ax.plot(
            xs,
            val_acc,
            linestyle="none",
            marker="o",
            markersize=1.0,
            color="indianred",
        )
        window = max(1, val_acc.shape[0] // 5)
        val_acc_roll_mean = (
            pd.Series(val_acc).rolling(window=window, min_periods=1).mean()
        )
        line2 = ax.plot(
            xs,
            val_acc_roll_mean,
            color="indianred",
            linewidth=1.5,
        )
        handles.append(line2[0])
        labels.append("Val accuracy")
        train_acc = np.asarray([train_acc_pca[e] for e in xs])
        line = ax.plot(
            xs,
            train_acc,
            linestyle="none",
            marker="o",
            markersize=1.0,
            color="lightcoral",
        )
        window = max(1, train_acc.shape[0] // 5)
        train_acc_roll_mean = (
            pd.Series(train_acc).rolling(window=window, min_periods=1).mean()
        )
        line2 = ax.plot(
            xs,
            train_acc_roll_mean,
            color="lightcoral",
            linewidth=1.5,
        )
        handles.append(line2[0])
        labels.append("Train accuracy")
    if wta:
        xs = sorted(train_acc_top)
        line = ax.plot(
            xs,
            [train_acc_top[e] for e in xs],
            label="train acc (top)",
            linestyle="none",
            marker="s",
            markersize=3,
            color="red",
        )
        handles2.append(line[0])
        labels2.append("Train accuracy")
        xs = sorted(val_acc_top)
        line = ax.plot(
            xs,
            [val_acc_top[e] for e in xs],
            label="val acc (top)",
            linestyle="none",
            marker="s",
            markersize=3,
            color="orange",
        )
        handles2.append(line[0])
        labels2.append("Val accuracy")

    # mcc (right axis)
    if mcc:
        xs = sorted(val_mcc)
        mcc_line = ax.plot(
            xs,
            [val_mcc[e] for e in xs],
            linestyle="solid",
            label="MCC",
            color="blue",
            marker="s",
            markersize=3,
            linewidth=1.5,
        )
        handles.append(mcc_line[0])
        labels.append("MCC")
        xs = sorted(train_mcc)
        mcc_line = ax.plot(
            xs,
            [train_mcc[e] for e in xs],
            linestyle="dashed",
            label="train mcc",
            color="blue",
            marker="s",
            markersize=3,
            linewidth=1.5,
        )
        handles.append(mcc_line[0])
        labels.append("train mcc")

    # add mean line from the first position
    if pca:
        import pandas as pd

        y_line = val_acc[0]
        # # window = max(1, val_acc.shape[0] // 5)
        # # y_line_current = pd.Series(val_acc).rolling(window=window, min_periods=1).mean()
        l1 = ax.axhline(
            y=y_line,
            linestyle="dashed",
            linewidth=0.5,
            color="grey",
            label="baseline val acc",
        )
        # l2 = ax.axhline(
        #     y=y_line_current,
        #     linestyle="solid",
        #     linewidth=0.5,
        #     color="red",
        #     label="average val acc",
        # )
        handles.append(l1)
        # handles.append(l2)
        labels.append("baseline val acc")
        # labels.append("average val acc")

    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Clustering")
    fig.supxlabel("Batches")
    ax.set_ylim(bottom=val_acc.min(), top=train_acc.max())

    # Phi (right axis)
    if phi:
        xs = sorted(val_phi)
        val_phi_arr = np.asarray([val_phi[e] for e in xs])
        train_phi_arr = np.asarray([train_phi[e] for e in xs])
        ax2.plot(
            xs,
            val_phi_arr,
            linestyle="none",
            color="skyblue",
            marker="p",
            markersize=1.0,
        )
        window = max(1, val_phi_arr.shape[0] // 5)
        val_phi_roll_mean = (
            pd.Series(val_phi_arr).rolling(window=window, min_periods=1).mean()
        )
        phi_line_val2 = ax2.plot(
            xs,
            val_phi_roll_mean,
            color="skyblue",
            linewidth=1.5,
        )
        ax2.plot(
            xs,
            train_phi_arr,
            linestyle="none",
            color="deepskyblue",
            marker="p",
            markersize=1.0,
        )
        train_phi_roll_mean = (
            pd.Series(train_phi_arr).rolling(window=window, min_periods=1).mean()
        )
        phi_line_train2 = ax2.plot(
            xs,
            train_phi_roll_mean,
            color="deepskyblue",
            linewidth=1.5,
        )
        ax2.set_ylabel("Clustering")
        handles2.append(phi_line_val2[0])
        labels2.append("Phi val")
        handles2.append(phi_line_train2[0])
        labels2.append("Phi train")

    # Create legend with white background, box, and smaller font
    if handles:
        ax.legend(handles, labels, loc="lower left", framealpha=1.0, fontsize=5)
    else:
        ax.legend(loc="lower left", framealpha=1.0, fontsize=5)

    if handles2:
        ax2.legend(handles2, labels2, loc="lower left", framealpha=1.0, fontsize=5)
    else:
        ax2.legend(loc="lower left", framealpha=1.0, fontsize=5)

    out_path = self._acc_log_file.replace(".jsonl", ".png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phi_acc(all_scores, epoch):
    import shutil
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    phi_means = all_scores.mean(axis=1)
    all_scores[:, :, 8] *= 100
    phi_means[:, 8] *= 100

    # extract mean sleep period for each sleep rate
    sleep_amount_mean = phi_means[:, 3]
    labels = np.char.mod("%.1f", sleep_amount_mean)[::-1]
    y1 = all_scores[:, :, 1][::-1]
    y2 = all_scores[:, :, 8][::-1]
    c1 = "#ffe5e1"
    c2 = "#c7fdf7"
    c11 = "#ffbfb3"
    c22 = "#6afae9"
    c111 = "#ff7f68"
    c222 = "#05af9b"

    # parameters for positioning
    positions = np.arange(len(labels))
    width1 = 0.35
    width2 = 0.25

    # prepare data lists
    data1 = [y1[i] for i in range(len(labels))]
    data2 = [y2[i] for i in range(len(labels))]
    linewidth = 2
    flier_size = 3
    marker_size = 5
    # boxplot on left axis
    box1 = ax1.boxplot(
        data1,
        positions=positions - width1 / 2,
        widths=width2,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=c1, edgecolor=c11, linewidth=linewidth),
        medianprops=dict(color=c11, linewidth=linewidth, zorder=4),
        whiskerprops=dict(linewidth=linewidth, zorder=2),
        capprops=dict(linewidth=linewidth, zorder=2),
        flierprops=dict(
            markerfacecolor=c1, markersize=flier_size, markeredgecolor=c11, zorder=2
        ),
    )
    # boxplot on right axis
    box2 = ax2.boxplot(
        data2,
        positions=positions + width1 / 2,
        widths=width2,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=c2, edgecolor=c22, linewidth=linewidth),
        medianprops=dict(color=c22, linewidth=linewidth, zorder=5),
        whiskerprops=dict(linewidth=linewidth, zorder=3),
        capprops=dict(linewidth=linewidth, zorder=3),
        flierprops=dict(
            markerfacecolor=c2, markersize=flier_size, markeredgecolor=c22, zorder=3
        ),
    )

    # Change outer box (edge) color
    for element in ["whiskers", "caps"]:
        for item in box1[element]:
            item.set_color(c11)  # Change 'red' to your preferred color
    for element in ["whiskers", "caps"]:
        for item in box2[element]:
            item.set_color(c22)  # Change 'red' to your preferred color

    # 1) calculate the medians yourself
    medians1 = np.array([np.median(d) for d in data1])
    medians2 = np.array([np.median(d) for d in data2])
    pos1 = positions - width1 / 2
    pos2 = positions + width1 / 2

    # 2) plot group1’s median‐connecting line behind everything (low zorder)
    ax1.plot(
        pos1,
        medians1,
        linestyle="-",
        marker="o",
        markersize=marker_size,
        color=c111,
        linewidth=linewidth,
        zorder=6,
    )

    # 3) plot group2’s median‐connecting line on top (high zorder)
    ax2.plot(
        pos2,
        medians2,
        linestyle="-",
        marker="o",
        markersize=marker_size,
        color=c222,
        linewidth=linewidth,
        zorder=6,
    )

    # labels, ticks, legends
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Sleep amount ($\\%$)", fontsize=16)
    ax1.set_ylabel("Clustering score ($\\phi$)", color=c11, fontsize=16)
    ax2.set_ylabel("Accuracy ($\\%$)", color=c22, fontsize=16)
    ax2.spines["left"].set_color(c11)
    ax2.spines["right"].set_color(c22)
    ax2.tick_params(axis="y", colors=c22)
    ax1.tick_params(axis="y", colors=c11)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for side in ["left", "right"]:
        ax1.spines[side].set_position(("outward", 10))

    for side in ["left", "right"]:
        ax2.spines[side].set_position(("outward", 10))

    ax2.spines["bottom"].set_position(("outward", 15))
    ax1.spines["bottom"].set_position(("outward", 15))
    # get the current numeric tick locations
    ticks1 = ax1.get_yticks()
    ticks2 = ax2.get_yticks()

    # set the y‐limits to the true ends of those ticks
    ax1.set_ylim(ticks1.min(), ticks1.max())
    ax2.set_ylim(ticks2.min(), ticks2.max())
    ax1.spines["bottom"].set_bounds(0, phi_means.shape[0] - 1)
    ax2.spines["bottom"].set_bounds(0, phi_means.shape[0] - 1)

    path = "figures"
    if not os.path.exists(path):
        os.makedirs("figures")

    plt.savefig(os.path.join(path, "sleep_subplots.png"))
    plt.tight_layout()
    plt.show()

    def _apply_labels(ax, latex_enabled: bool):
        plt.rcParams["text.usetex"] = bool(latex_enabled)
        ylabel = "Weight"

        ax.set_ylabel(ylabel, fontsize=26)
        ax.set_xlabel("time (ms)", fontsize=26)

        legend_lines = [
            Line2D(
                [0], [0], color="black", linestyle="-", linewidth=2.0, label="Exc mean"
            ),
            Line2D(
                [0], [0], color="black", linestyle="--", linewidth=2.0, label="Inh mean"
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                linewidth=1.2,
                label="Exc samples",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Inh samples",
            ),
        ]

    latex_available = shutil.which("latex") is not None
    _apply_labels(latex_enabled=latex_available)
    pdf_path = os.path.join("plots", f"weights_trajectories_epoch_{int(epoch):03d}.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=900, bbox_inches="tight")
    plt.rcParams["text.usetex"] = False
    plt.close(fig)


def plot_traces(
    random_selection=False,
    N_exc=200,
    N_inh=50,
    pre_traces=None,
    post_traces=None,
):
    if random_selection:
        ...
    else:
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].plot(
            pre_traces[:, :-N_inh], label="excitatory pre-traces", color="lightgreen"
        )
        axs[0, 0].set_title("excitatory pre-trace")
        axs[0, 1].plot(
            pre_traces[:, -N_inh:], label="inhibitory pre-traces", color="lightblue"
        )
        axs[0, 1].set_title("inhibitory pre-trace")
        axs[1, 0].plot(
            post_traces[:, :N_exc], label="excitatory post-traces", color="green"
        )
        axs[1, 0].set_title("excitatory post-trace")
        axs[1, 1].plot(
            post_traces[:, N_exc:], label="inhibitory post-trace", color="blue"
        )
        axs[1, 1].set_title("inhbitiory post-trace")
        plt.show()


def mp_plot(mp, N_exc):
    plt.plot(mp[:, :N_exc], color="green")
    plt.title("excitatory membrane potential during training")
    plt.xlabel("time (ms)")
    plt.ylabel("membrane potential (mV)")
    plt.show()

    plt.plot(mp[:, N_exc:], color="red")
    plt.title("inhibitory membrane potential during training")
    plt.xlabel("time (ms)")
    plt.ylabel("membrane potential (mV)")
    plt.show()


class PCAScatterDisplay:
    def __init__(self, scaler, pca=None):
        self.scaler = scaler
        self.pca = pca
        self.figure_ = None
        self.ax_ = None
        self.colors_ = None

    def plot(self, X, Y, *, epoch, run, dataset, phi=None):
        from datetime import datetime
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)

        if self.figure_ is None:
            self.figure_, self.ax_ = plt.subplots()
        if self.colors_ is None:
            self.colors_ = plt.cm.tab10(np.linspace(0, 1, len(np.unique(Y))))
        self.ax_.clear()
        for c in np.unique(Y).astype(int):
            mask = Y == c
            self.ax_.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=20, c=self.colors_[c])
            self.ax_.scatter(
                X[mask, 0].mean(),
                X[mask, 1].mean(),
                s=100,
                marker="x",
                c=self.colors_[c],
            )

        self.ax_.legend()
        title = f"Batch {epoch}"
        if phi is not None:
            title += f": $\\phi$={phi:.2f}"
        self.ax_.set_title(title)

        date = datetime.now().strftime("%m%d%Y")
        ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.dir = os.path.join("plots", "PCA", dataset, date, run)
        os.makedirs(self.dir, exist_ok=True)
        self.figure_.savefig(os.path.join(self.dir, f"{ts}.png"), dpi=100)


def plot_epoch_training(acc, cluster, val_acc=None, val_phi=None):
    fig, ax0 = plt.subplots()

    # Left y-axis
    (line0,) = ax0.plot(cluster, color="tab:blue", label="Cluster")
    ax0.set_ylabel("Cluster", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")

    # Right y-axis
    ax1 = ax0.twinx()
    (line1,) = ax1.plot(acc, color="tab:red", label="Train Acc")
    if val_acc is not None:
        (line1b,) = ax1.plot(
            val_acc, color="tab:orange", linestyle="--", label="Val Acc"
        )
    ax1.set_ylabel("Accuracy", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Common title and xlabel
    fig.suptitle("Epoch Training")
    fig.supxlabel("Epoch")

    # Common legend
    lines = [line0, line1]
    if val_acc is not None:
        lines.append(line1b)
    if val_phi is not None:
        # Overlay val Phi on left axis for simplicity
        (line2,) = ax0.plot(val_phi, color="tab:green", linestyle=":", label="Val Phi")
        lines.append(line2)
    labels = [line.get_label() for line in lines]
    ax0.legend(lines, labels, loc="upper center")

    plt.show()
