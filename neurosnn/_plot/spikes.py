import matplotlib.pyplot as plt
import numpy as np
import os


def spike_plot(data, labels):
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Labels length ({len(labels)}) must match the number of time steps ({data.shape[0]})."
        )

    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data min/max: {data.min()}/{data.max()}")
    print(f"Number of non-zero elements: {np.count_nonzero(data)}")
    print(f"Unique values in data: {np.unique(data)}")

    if np.count_nonzero(data) == 0:
        print("WARNING: No spikes found in the data!")
        return

    valid_label_mask = labels != -1
    unique_labels = np.unique(labels[valid_label_mask])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    if np.any(data > 0):
        spike_threshold = 0
    else:
        spike_threshold = 1

    positions = [
        np.where(data[:, n] > spike_threshold)[0] for n in range(data.shape[1])
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.eventplot(positions, lineoffsets=np.arange(data.shape[1]), colors="black")
    ax.set_ylabel(f"{data.shape[1]} Units")
    ax.set_xlabel("Time (ms)")

    drawn_labels = set()
    y_offset = -10
    segment_start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            if current_label != -1:
                if current_label == -2:
                    label_text = "Sleep" if current_label not in drawn_labels else None
                    ax.hlines(y=y_offset, xmin=segment_start, xmax=i,
                              color=label_colors.get(current_label, "blue"),
                              linewidth=6, label=label_text)
                else:
                    label_text = (f"Class {current_label}"
                                  if current_label not in drawn_labels else None)
                    ax.hlines(y=y_offset, xmin=segment_start, xmax=i,
                              color=label_colors.get(current_label, "black"),
                              linewidth=6, label=label_text)
                drawn_labels.add(current_label)
            current_label = labels[i]
            segment_start = i

    if current_label != -1:
        if current_label == -2:
            label_text = "Sleep" if current_label not in drawn_labels else None
            ax.hlines(y=y_offset, xmin=segment_start, xmax=len(labels),
                      color=label_colors.get(current_label, "blue"),
                      linewidth=6, label=label_text)
        else:
            label_text = (f"Class {current_label}"
                          if current_label not in drawn_labels else None)
            ax.hlines(y=y_offset, xmin=segment_start, xmax=len(labels),
                      color=label_colors.get(current_label, "black"),
                      linewidth=6, label=label_text)
        drawn_labels.add(current_label)

    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(handles, labels_legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=len(unique_labels))
    plt.title("Spikes with Class-based Horizontal Lines")
    plt.tight_layout()
    plt.show()


def _meta_grid(n):
    """Most-square factor pair (rows, cols) for arranging n class tiles (e.g. 10 -> 2x5)."""
    best = (1, n)
    for r in range(1, int(np.sqrt(n)) + 1):
        if n % r == 0:
            best = (r, n // r)
    return best


def tiled_positions(n_exc, n_groups):
    """Neuron index (block order) -> (row, col) in the composite tiled heatmap.

    Layout: n_groups class tiles of k x k each (k = sqrt(N_exc/n_groups)), arranged
    in a most-square meta-grid. Returns (rows, cols, H, W, k, meta_r, meta_c).
    """
    g = n_exc // n_groups
    k = int(round(np.sqrt(g)))
    meta_r, meta_c = _meta_grid(n_groups)
    H, W = meta_r * k, meta_c * k
    idx = np.arange(n_exc)
    cls, p = idx // g, idx % g
    rows = (cls // meta_c) * k + p // k
    cols = (cls % meta_c) * k + p % k
    return rows, cols, H, W, k, meta_r, meta_c


def _tiled_composite(vec, rows, cols, H, W):
    img = np.full((H, W), np.nan)
    img[rows, cols] = vec[: len(rows)]
    return img


def tiled_spike_plot(
    spikes_exc, spikes_in, spikes_ih, label, run, dataset, num,
    weights_st_ex, weights_ih_ex, n_groups, **_ignored,
):
    """Class-tiled spike plot for the grouped/tiled architecture: excitatory and
    inhibitory mean activity re-arranged into a meta-grid of per-class k x k tiles
    (so a class 'lights up' as a contiguous region), plus per-exc-neuron incoming
    inhibition (|W_ie|, which regions are most suppressed), total input weight
    (|W_se|), and the pooled per-class readout. Works for non-square N_exc."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle
    from datetime import datetime
    from neurosnn._utils.logger import tracking_run_dir

    N_exc = spikes_exc.shape[1]
    rows, cols, H, W, k, meta_r, meta_c = tiled_positions(N_exc, n_groups)
    g = N_exc // n_groups

    exc = spikes_exc.mean(axis=0)                          # (N_exc,) mean rate
    inh = spikes_ih.mean(axis=0)                           # (N_inh==N_exc,) 1:1 paired
    inh_in = np.abs(weights_ih_ex).sum(axis=0)             # (N_exc,) inhibition received
    se_in = np.abs(weights_st_ex).sum(axis=0)              # (N_exc,) total input weight
    readout = np.array([exc[c * g:(c + 1) * g].mean() for c in range(n_groups)])
    pred = int(readout.argmax())
    inp = spikes_in.mean(axis=0)
    iside = int(round(np.sqrt(inp.size)))

    def tiled(ax, vec, title, cmap="viridis", mark_true=None, mark_pred=None):
        ax.imshow(_tiled_composite(vec, rows, cols, H, W), cmap=cmap, interpolation="nearest")
        for c in range(n_groups):
            r0, c0 = (c // meta_c) * k, (c % meta_c) * k
            ec = "#00d000" if c == mark_true else ("red" if c == mark_pred else "white")
            lw = 2.2 if ec != "white" else 0.4
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), k, k, fill=False, edgecolor=ec, lw=lw))
            ax.text(c0 + k / 2, r0 + 1.0, str(c), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2.4, 1.2], hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(inp.reshape(iside, iside), cmap="gray_r")
    ax.set_title(f"Input  label={label}", fontsize=9); ax.axis("off")

    tiled(fig.add_subplot(gs[0, 1]), exc, "Excitatory activity (class tiles; green=true red=pred)",
          mark_true=int(label), mark_pred=pred)
    tiled(fig.add_subplot(gs[1, 1]), inh, "Inhibitory activity (same layout, 1:1 paired)")

    ax = fig.add_subplot(gs[0, 2]); ax.imshow(readout.reshape(meta_r, meta_c), cmap="magma")
    for c in range(n_groups):
        ax.text(c % meta_c, c // meta_c, f"{c}", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold" if c == pred else "normal")
    ax.add_patch(Rectangle((pred % meta_c - 0.5, pred // meta_c - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    ax.set_title(f"Readout (pooled/class) pred={pred}", fontsize=9); ax.axis("off")

    tiled(fig.add_subplot(gs[1, 0]), inh_in, "I->E inhibition received (|W_ie|/neuron)", cmap="inferno")
    tiled(fig.add_subplot(gs[1, 2]), se_in, "Input weight (|W_se|/neuron)", cmap="cividis")

    fig.suptitle(f"Run {num} — label {label}", fontsize=12)
    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(tracking_run_dir(dataset, run), "spikes")
    for sub in (str(label), "all"):
        d = os.path.join(base, sub); os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{ts_spec}.png"), dpi=100)
    plt.close(fig)


def heatmap_spike_response(
    spikes_exc,
    spikes_in,
    spikes_ih,
    label,
    run,
    dataset,
    num,
    st,
    spike_trace,
    ex,
    x_target_se,
    x_target_ex,
    weights_st_ex,
    weights_ex_ex,
    weights_ex_ih,
    weights_ih_ex,
    n_groups=None,
):
    import matplotlib
    matplotlib.use("Agg")

    # grouped/tiled architecture: use the class-tiled layout (also handles
    # non-square N_exc, which the sqrt(N_exc) reshape below cannot).
    if n_groups:
        tiled_spike_plot(
            spikes_exc=spikes_exc, spikes_in=spikes_in, spikes_ih=spikes_ih,
            label=label, run=run, dataset=dataset, num=num,
            weights_st_ex=weights_st_ex, weights_ih_ex=weights_ih_ex, n_groups=n_groups,
        )
        return

    fig, axs = plt.subplots(figsize=(10, 6), nrows=3, ncols=4)

    def create_plot(spikes, ax, title, rows, cols, ax_flip):
        if spikes is None or np.sum(spikes) == 0:
            ax.set_title(title + " (empty)")
            ax.axis("off")
            return
        avg_spikes = np.mean(spikes, axis=ax_flip)
        expected = rows * cols
        if avg_spikes.size != expected:
            raise ValueError(
                f"{title}: cannot reshape avg_spikes of size {avg_spikes.size} into ({rows}, {cols})"
            )
        avg_spikes_reshaped = avg_spikes.reshape((rows, cols))
        im = ax.imshow(avg_spikes_reshaped, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    input_size = int(np.sqrt(spikes_in.shape[1]))
    exc_size = int(np.sqrt(spikes_exc.shape[1]))   # E grid side: 32 @1024, 64 @4096
    inh_size = int(np.sqrt(spikes_ih.shape[1]))    # I grid side: 15 @225, 30 @900
    create_plot(spikes_in, axs[0, 0], "Input activity", input_size, input_size, ax_flip=0)
    create_plot(spikes_exc, axs[0, 1], "Excitatory activity", exc_size, exc_size, ax_flip=0)
    create_plot(spikes_ih, axs[0, 2], "Inhibitory activity", inh_size, inh_size, ax_flip=0)

    n = len(spike_trace)
    colors = []
    for i in range(n):
        if i < st:
            colors.append("green")
        elif i < ex:
            colors.append("blue")
        else:
            colors.append("red")

    axs[0, 3].bar(np.arange(len(spike_trace)), spike_trace, color=colors)
    axs[0, 3].set_title("Spike trace distribution")
    axs[0, 3].set_ylabel("Spike count", fontsize=5)
    axs[0, 3].set_xticks([])
    axs[0, 3].axhline(y=x_target_se, color="blue", linestyle="--", linewidth=0.1)
    axs[0, 3].axhline(y=x_target_ex, color="green", linestyle="--", linewidth=0.1)

    create_plot(weights_st_ex, axs[1, 0], "St->Ex Outgoing Weights",
                input_size, input_size, ax_flip=1)
    create_plot(weights_ex_ex, axs[1, 1], "Ex->Ex Outgoing Weights", exc_size, exc_size, ax_flip=1)
    create_plot(weights_ex_ih, axs[1, 2], "Ex->Ih Outgoing Weights", exc_size, exc_size, ax_flip=1)
    create_plot(np.abs(weights_ih_ex), axs[1, 3], "Ih->Ex Outgoing Weights", inh_size, inh_size, ax_flip=1)

    create_plot(weights_st_ex, axs[2, 0], "St->Ex Incoming Weights", exc_size, exc_size, ax_flip=0)
    create_plot(weights_ex_ex, axs[2, 1], "Ex->Ex Incoming Weights", exc_size, exc_size, ax_flip=0)
    create_plot(weights_ex_ih, axs[2, 2], "Ex->Ih Incoming Weights", inh_size, inh_size, ax_flip=0)
    create_plot(np.abs(weights_ih_ex), axs[2, 3], "Ih->Ex Incoming Weights", exc_size, exc_size, ax_flip=0)

    row_labels = ["Spike activity", "Outgoing weights", "Incoming weights"]
    for i in range(3):
        bbox = axs[i, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2
        fig.text(0.02, y_center, row_labels[i], va="center", ha="left",
                 rotation=90, fontsize=8, fontweight="bold")

    fig.suptitle(f"Run: {num}, Label {label}")
    from datetime import datetime
    from neurosnn._utils.logger import tracking_run_dir

    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(tracking_run_dir(dataset, run), "spikes")
    directory = os.path.join(base, str(label))
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, f"{ts_spec}.png")
    fig.savefig(out_path, dpi=100)
    directory = os.path.join(base, "all")
    os.makedirs(directory, exist_ok=True)
    out_path_glob = os.path.join(directory, f"{ts_spec}.png")
    fig.savefig(out_path_glob, dpi=100)
    plt.close(fig)


def gif_spike_rate_by_label(frame_folder, output_filename="my_awesome.gif", duration=100, loop=0):
    import glob
    from PIL import Image

    files = glob.glob(f"{frame_folder}/*.png")
    files_sorted = sorted(files, key=os.path.basename)
    frames = [Image.open(image) for image in files_sorted]

    if not frames:
        print(f"No images found in {frame_folder}")
        return

    frame_one = frames[0]
    frame_one.save(output_filename, format="GIF", append_images=frames[1:],
                   save_all=True, duration=duration, loop=loop)
    print("gif made!")


def plot_floats_and_spikes(images, spikes, spike_labels, img_labels, num_steps):
    from neurosnn._utils.helper import get_contiguous_segment

    unique_labels = np.unique(img_labels)
    n_cols = len(unique_labels)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, label in enumerate(unique_labels):
        img_idx = np.where(np.array(img_labels) == label)[0][0]
        ax_img = axs[1, i]
        ax_img.imshow(np.squeeze(images[img_idx]), cmap="gray")
        ax_img.set_title(f"Digit {label}")
        ax_img.axis("off")

        spike_idx_all = np.where(np.array(spike_labels) == label)[0][:num_steps]
        if len(spike_idx_all) == 0:
            print(f"No spiking data found for label {label}.")
            continue

        segment = get_contiguous_segment(spike_idx_all)
        if segment is None or len(segment) == 0:
            print(f"No contiguous segment found for label {label}.")
            continue

        spike_segment = spikes[segment, :]
        positions = [
            np.where(spike_segment[:, n] == 1)[0] for n in range(spike_segment.shape[1])
        ]

        ax_spike = axs[0, i]
        ax_spike.eventplot(positions, colors="black")
        ax_spike.set_title(f"Spikes for {label}")
        ax_spike.set_xlabel("Time steps")
        ax_spike.set_ylabel("Neuron")
        ax_spike.set_ylim(-1, spike_segment.shape[1])

    plt.tight_layout()
    plt.savefig("plots/comparison_spike_img.png")
    plt.show()


def spike_threshold_plot(spike_threshold, N_exc):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(np.mean(spike_threshold[:N_exc], axis=1), label="excitatory", color="green")
    axs[1].plot(np.mean(spike_threshold[N_exc:], axis=1), label="inhibitory", color="red")
    axs[0].set_ylabel("spiking threshold (mV)")
    axs[1].set_ylabel("spiking threshold (mV)")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    fig.text(0.5, 0.04, "time (ms)", ha="center")
    fig.suptitle("Average spiking threshold per neuron group over time")
    plt.legend()
    plt.show()


def plot_spikes(self):
    label_for_plotting = input("Which label should we plot? ")
    while label_for_plotting != "stop":
        if label_for_plotting == "all":
            frame_folder = (
                f"plots\\spikes\\{self.image_dataset}\\all\\{self.ts}\\{self.ts_spec}"
            )
            output_filename = f"plots\\spikes\\{self.image_dataset}\\all\\{self.ts}\\{self.ts_spec}\\evolution.gif"
        else:
            frame_folder = f"plots\\spikes\\{self.image_dataset}\\{label_for_plotting}\\{self.ts}\\{self.ts_spec}"
            output_filename = f"plots\\spikes\\{self.image_dataset}\\{label_for_plotting}\\{self.ts}\\{self.ts_spec}\\evolution.gif"
        gif_spike_rate_by_label(
            frame_folder=frame_folder,
            output_filename=output_filename,
            duration=100,
            loop=0,
        )
        label_for_plotting = input("Which label should we plot? ")


class GenerateGif:
    def __init__(self, frame_folder, output_filename, duration=100, loop=0):
        self.frame_folder = frame_folder
        self.output_filename = output_filename
        self.duration = duration
        self.loop = loop

    @classmethod
    def from_PCAScatterDisplay(self, PCAS):
        from copy import deepcopy
        self.frame_folder = deepcopy(PCAS.dir)

    def create(self, frame_folder=None, output_filename=None):
        import glob
        import os
        from PIL import Image

        if frame_folder is None:
            frame_folder = self.frame_folder
        if output_filename is None:
            output_filename = self.output_filename

        files = glob.glob(f"{frame_folder}/*.png")
        if len(files) > 1:
            files_sorted = sorted(
                files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
        else:
            return

        frames = [Image.open(image) for image in files_sorted]
        if not frames:
            print(f"No images found in {frame_folder}")
            return

        frame_one = frames[0]
        frame_one.save(
            os.path.join(frame_folder, output_filename),
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=self.duration,
            loop=self.loop,
        )
        print("PCA gif made!")
