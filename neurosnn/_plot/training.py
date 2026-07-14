import os
import numpy as np
import matplotlib.pyplot as plt


def preview_loaded_data(
    self, dataset, num_image_samples: int = 9, save_path: str | None = None
):
    if dataset.lower() == "geomfig":
        from get_data import _geomfig_generate_one  # type: ignore

        try:
            classes = [0, 1, 2, 3]
            per_class = max(1, int(num_image_samples))
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
                    len(classes), per_class,
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

    if not hasattr(self, "image_streamer") or self.image_streamer is None:
        return
    try:
        self.image_streamer.show_preview(
            num_samples=num_image_samples, save_path=save_path
        )
    except Exception as exc:
        print(f"Dataset preview skipped ({exc})")
    finally:
        self._image_preview_done = True


def plot_stats(self, stats_log_file, read_jsonl):
    '''
    Iterative per-batch diagnostics figure, saved to <run>/stats/stats.png and
    overwritten each val step. One accuracy anchor (top, full width) plus six
    diagnostic panels: SE diversity, RF concentration, plasticity balance, E/I &
    activity, weight magnitudes, neuron dynamics.

    Reads the single unified stats jsonl: accuracy/phi records (carrying an
    'accuracy'/'phi' key) feed the anchor; the remaining per-batch diagnostics
    records feed the panels. No-ops gracefully if nothing was recorded.
    '''
    import matplotlib.gridspec as gridspec

    if not stats_log_file or not os.path.exists(stats_log_file):
        return
    all_recs = [r for r in read_jsonl(stats_log_file) if r.get("epoch") is not None]
    if not all_recs:
        return

    # Diagnostics records are those that are NOT accuracy/phi/mcc rows.
    recs = [
        r for r in all_recs
        if not any(k in r for k in ("accuracy", "phi", "mcc"))
    ]
    recs.sort(key=lambda r: r["epoch"])
    xs = [int(r["epoch"]) for r in recs]

    def col(key):
        return np.asarray([r.get(key, np.nan) for r in recs], dtype=float)

    # --- accuracy anchor data (same unified file) ---
    val_acc, train_acc, val_phi, train_phi = {}, {}, {}, {}
    for r in all_recs:
        e = int(r["epoch"])
        if r.get("accuracy") is not None and r.get("method") == "pca_lr":
            (val_acc if r.get("split") == "val" else train_acc)[e] = float(r["accuracy"])
        if r.get("phi") is not None:
            (val_phi if r.get("split") == "val" else train_phi)[e] = float(r["phi"])

    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)

    # Row 0 (full width): accuracy anchor
    ax = fig.add_subplot(gs[0, :])
    if val_acc:
        ex = sorted(val_acc)
        ax.plot(ex, [val_acc[e] for e in ex], "-o", ms=3, color="indianred", label="val acc")
    if train_acc:
        ex = sorted(train_acc)
        ax.plot(ex, [train_acc[e] for e in ex], "-o", ms=3, color="lightcoral", label="train acc")
    ax.set_ylabel("accuracy"); ax.set_title("accuracy (+ phi)")
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=7)
    if val_phi or train_phi:
        axp = ax.twinx()
        if val_phi:
            ep = sorted(val_phi); axp.plot(ep, [val_phi[e] for e in ep], "-", color="skyblue", lw=1.2, label="val phi")
        if train_phi:
            ep = sorted(train_phi); axp.plot(ep, [train_phi[e] for e in ep], "-", color="deepskyblue", lw=1.2, label="train phi")
        axp.set_ylabel("phi", color="steelblue"); axp.tick_params(axis="y", labelcolor="steelblue")
        axp.legend(loc="lower right", fontsize=7)

    def twin_panel(pos, title, left, right):
        a = fig.add_subplot(pos); a2 = a.twinx()
        for key, lbl, c in left:
            a.plot(xs, col(key), "-o", ms=2.5, color=c, label=lbl)
        for key, lbl, c in right:
            a2.plot(xs, col(key), "-o", ms=2.5, color=c, label=lbl)
        a.set_title(title); a.set_xlabel("batch"); a.grid(alpha=0.3)
        lc = left[0][2] if left else "k"; rc = right[0][2] if right else "k"
        a.set_ylabel("/".join(l[1] for l in left), color=lc, fontsize=8)
        a2.set_ylabel("/".join(r[1] for r in right), color=rc, fontsize=8)
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=6, loc="best")
        return a

    # P1 SE diversity (decorrelation): PR up + cosine down = good
    twin_panel(gs[1, 0], "SE diversity",
               [("rf_participation_ratio", "SE PR", "C0")],
               [("rf_mean_cosine", "SE cosine", "C3")])
    # P2 RF concentration
    twin_panel(gs[1, 1], "RF concentration",
               [("rf_gini", "Gini", "C0")],
               [("rf_entropy", "entropy", "C3")])
    # P3 plasticity balance
    twin_panel(gs[1, 2], "plasticity balance",
               [("ltp_ltd_ratio", "LTP/LTD", "C2")],
               [("mean_delta_w", "delta_w", "C1")])
    # P4 E/I & activity
    twin_panel(gs[2, 0], "E/I & activity",
               [("ei_ratio_median", "E/I med", "C1"), ("ei_ratio_p90", "E/I p90", "C5")],
               [("active_frac_exc", "active-E", "C0"), ("pop_sparseness", "sparse", "C2")])
    # P5 weight magnitudes (single axis, 4 lines)
    aw = fig.add_subplot(gs[2, 1])
    for key, lbl, c in [("w_se_mean", "SE", "C0"), ("w_ee_mean", "EE", "C2"),
                        ("w_ei_mean", "EI", "C1"), ("w_ie_mean", "|IE|", "C3")]:
        aw.plot(xs, col(key), "-o", ms=2.5, color=c, label=lbl)
    aw.set_title("weight magnitudes"); aw.set_xlabel("batch")
    aw.set_ylabel("mean |w|", fontsize=8); aw.grid(alpha=0.3); aw.legend(fontsize=6)
    # P6 neuron dynamics
    twin_panel(gs[2, 2], "neuron dynamics",
               [("mean_I_syn_exc", "Isyn E", "C0"), ("mean_I_syn_inh", "Isyn I", "C3")],
               [("mean_mp_exc", "mp E", "C2"), ("mean_mp_inh", "mp I", "C1")])

    out_path = stats_log_file.replace(".jsonl", ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_phi_acc(all_scores, epoch):
    import shutil
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    phi_means = all_scores.mean(axis=1)
    all_scores[:, :, 8] *= 100
    phi_means[:, 8] *= 100

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

    positions = np.arange(len(labels))
    width1 = 0.35
    width2 = 0.25

    data1 = [y1[i] for i in range(len(labels))]
    data2 = [y2[i] for i in range(len(labels))]
    linewidth = 2
    flier_size = 3
    marker_size = 5

    box1 = ax1.boxplot(
        data1, positions=positions - width1 / 2, widths=width2, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor=c1, edgecolor=c11, linewidth=linewidth),
        medianprops=dict(color=c11, linewidth=linewidth, zorder=4),
        whiskerprops=dict(linewidth=linewidth, zorder=2),
        capprops=dict(linewidth=linewidth, zorder=2),
        flierprops=dict(markerfacecolor=c1, markersize=flier_size, markeredgecolor=c11, zorder=2),
    )
    box2 = ax2.boxplot(
        data2, positions=positions + width1 / 2, widths=width2, patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=c2, edgecolor=c22, linewidth=linewidth),
        medianprops=dict(color=c22, linewidth=linewidth, zorder=5),
        whiskerprops=dict(linewidth=linewidth, zorder=3),
        capprops=dict(linewidth=linewidth, zorder=3),
        flierprops=dict(markerfacecolor=c2, markersize=flier_size, markeredgecolor=c22, zorder=3),
    )

    for element in ["whiskers", "caps"]:
        for item in box1[element]:
            item.set_color(c11)
    for element in ["whiskers", "caps"]:
        for item in box2[element]:
            item.set_color(c22)

    medians1 = np.array([np.median(d) for d in data1])
    medians2 = np.array([np.median(d) for d in data2])
    pos1 = positions - width1 / 2
    pos2 = positions + width1 / 2

    ax1.plot(pos1, medians1, linestyle="-", marker="o", markersize=marker_size,
             color=c111, linewidth=linewidth, zorder=6)
    ax2.plot(pos2, medians2, linestyle="-", marker="o", markersize=marker_size,
             color=c222, linewidth=linewidth, zorder=6)

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

    ticks1 = ax1.get_yticks()
    ticks2 = ax2.get_yticks()
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


def plot_traces(random_selection=False, N_exc=200, N_inh=50, pre_traces=None, post_traces=None):
    if random_selection:
        ...
    else:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(pre_traces[:, :-N_inh], label="excitatory pre-traces", color="lightgreen")
        axs[0, 0].set_title("excitatory pre-trace")
        axs[0, 1].plot(pre_traces[:, -N_inh:], label="inhibitory pre-traces", color="lightblue")
        axs[0, 1].set_title("inhibitory pre-trace")
        axs[1, 0].plot(post_traces[:, :N_exc], label="excitatory post-traces", color="green")
        axs[1, 0].set_title("excitatory post-trace")
        axs[1, 1].plot(post_traces[:, N_exc:], label="inhibitory post-trace", color="blue")
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
            self.colors_ = plt.cm.tab10(np.linspace(0, 1, int(np.max(Y)) + 1))
        self.ax_.clear()
        for c in np.unique(Y).astype(int):
            mask = Y == c
            self.ax_.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=20, c=self.colors_[c])
            self.ax_.scatter(X[mask, 0].mean(), X[mask, 1].mean(), s=100, marker="x", c=self.colors_[c])

        self.ax_.legend()
        title = f"Batch {epoch}"
        if phi is not None:
            title += f": $\\phi$={phi:.2f}"
        self.ax_.set_title(title)

        from neurosnn._utils.logger import tracking_run_dir

        ts = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.dir = os.path.join(tracking_run_dir(dataset, run), "plots", "PCA")
        os.makedirs(self.dir, exist_ok=True)
        self.figure_.savefig(os.path.join(self.dir, f"{ts}.png"), dpi=100)


def plot_epoch_training(acc, cluster, val_acc=None, val_phi=None):
    fig, ax0 = plt.subplots()
    (line0,) = ax0.plot(cluster, color="tab:blue", label="Cluster")
    ax0.set_ylabel("Cluster", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")

    ax1 = ax0.twinx()
    (line1,) = ax1.plot(acc, color="tab:red", label="Train Acc")
    if val_acc is not None:
        (line1b,) = ax1.plot(val_acc, color="tab:orange", linestyle="--", label="Val Acc")
    ax1.set_ylabel("Accuracy", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Epoch Training")
    fig.supxlabel("Epoch")

    lines = [line0, line1]
    if val_acc is not None:
        lines.append(line1b)
    if val_phi is not None:
        (line2,) = ax0.plot(val_phi, color="tab:green", linestyle=":", label="Val Phi")
        lines.append(line2)
    labels = [line.get_label() for line in lines]
    ax0.legend(lines, labels, loc="upper center")
    plt.show()
