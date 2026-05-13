import matplotlib.pyplot as plt
import numpy as np
import os


def spike_plot(data, labels):
    # Validate dimensions
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Labels length ({len(labels)}) must match the number of time steps ({data.shape[0]})."
        )

    # Debug: Print data information
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data min/max: {data.min()}/{data.max()}")
    print(f"Number of non-zero elements: {np.count_nonzero(data)}")
    print(f"Unique values in data: {np.unique(data)}")

    # Check if there are any spikes at all
    if np.count_nonzero(data) == 0:
        print("WARNING: No spikes found in the data!")
        print("This could be because:")
        print(
            "1. The time window is too small (only last 5% of data is shown by default)"
        )
        print("2. The neurons selected don't have spikes")
        print("3. The spike data format is different than expected")
        print("4. The network hasn't learned to spike yet")
        print("\nSuggestions:")
        print(
            "- Try using a larger time window by setting start_time_spike_plot to an earlier time"
        )
        print("- Check if the network is actually producing spikes during training")
        print("- Verify that the spike data contains non-zero values")
        return

    # Assign colors to unique labels (excluding -1 if desired)
    valid_label_mask = labels != -1
    unique_labels = np.unique(labels[valid_label_mask])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    # Collect spike positions for each neuron
    # Try different spike representations
    if np.any(data > 0):
        # If there are positive values, use those as spikes
        spike_threshold = 0
        print(f"Using positive values as spikes (threshold > {spike_threshold})")
    else:
        # Default to looking for exactly 1
        spike_threshold = 1
        print(f"Using exact value {spike_threshold} as spikes")

    positions = [
        np.where(data[:, n] > spike_threshold)[0] for n in range(data.shape[1])
    ]

    # Debug: Print spike information
    total_spikes = sum(len(pos) for pos in positions)
    print(f"Total spikes found: {total_spikes}")
    print(
        f"Spikes per neuron: {[len(pos) for pos in positions[:10]]}..."
    )  # First 10 neurons

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spikes
    ax.eventplot(positions, lineoffsets=np.arange(data.shape[1]), colors="black")
    ax.set_ylabel(f"{data.shape[1]} Units")
    ax.set_xlabel("Time (ms)")

    """
    To plot the 
    """

    # We'll collect which labels we've drawn (for legend) so we don't add duplicates
    drawn_labels = set()

    # Add the horizontal line below the spikes
    y_offset = -10  # Position below the spike raster

    # We iterate through the time steps to identify contiguous segments
    segment_start = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        # If the label changes, we close off the old segment (unless it was -1)
        if labels[i] != current_label:
            if current_label != -1:
                if current_label == -2:
                    # For sleep segments, label as "Sleep" only once
                    label_text = "Sleep" if current_label not in drawn_labels else None
                    ax.hlines(
                        y=y_offset,
                        xmin=segment_start,
                        xmax=i,  # up to but not including i
                        color=label_colors.get(current_label, "blue"),
                        linewidth=6,
                        label=label_text,
                    )
                else:
                    label_text = (
                        f"Class {current_label}"
                        if current_label not in drawn_labels
                        else None
                    )
                    ax.hlines(
                        y=y_offset,
                        xmin=segment_start,
                        xmax=i,  # up to but not including i
                        color=label_colors.get(current_label, "black"),
                        linewidth=6,
                        label=label_text,
                    )
                drawn_labels.add(current_label)

            # Update to the new segment
            current_label = labels[i]
            segment_start = i

    # Handle the last segment after exiting the loop
    if current_label != -1:
        if current_label == -2:
            label_text = "Sleep" if current_label not in drawn_labels else None
            ax.hlines(
                y=y_offset,
                xmin=segment_start,
                xmax=len(labels),
                color=label_colors.get(current_label, "blue"),
                linewidth=6,
                label=label_text,
            )
        else:
            label_text = (
                f"Class {current_label}" if current_label not in drawn_labels else None
            )
            ax.hlines(
                y=y_offset,
                xmin=segment_start,
                xmax=len(labels),
                color=label_colors.get(current_label, "black"),
                linewidth=6,
                label=label_text,
            )
        drawn_labels.add(current_label)

    # Create a legend from the existing artists
    handles, labels_legend = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(unique_labels),
    )

    plt.title("Spikes with Class-based Horizontal Lines")
    plt.tight_layout()
    plt.show()


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
):
    import matplotlib

    matplotlib.use("Agg")

    # define subplot
    fig, axs = plt.subplots(figsize=(10, 6), nrows=3, ncols=4)

    def create_plot(spikes, ax, title, rows, cols, ax_flip):
        # average spike responses
        if spikes is None or np.sum(spikes) == 0:
            ax.set_title(title + " (empty)")
            ax.axis("off")
            return

        avg_spikes = np.mean(spikes, axis=ax_flip)

        # ✅ protect against reshape mismatch
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

    # top row heatmaps
    input_size = int(np.sqrt(spikes_in.shape[1]))
    create_plot(
        spikes_in, axs[0, 0], "Input activity", input_size, input_size, ax_flip=0
    )
    create_plot(spikes_exc, axs[0, 1], "Excitatory activity", 32, 32, ax_flip=0)
    create_plot(spikes_ih, axs[0, 2], "Inhibitory activity", 15, 15, ax_flip=0)

    # add barplot of spike_trace distribution
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

    # create heatmap plots
    create_plot(
        weights_st_ex,
        axs[1, 0],
        "St->Ex Outgoing Weights",
        input_size,
        input_size,
        ax_flip=1,
    )
    create_plot(weights_ex_ex, axs[1, 1], "Ex->Ex Outgoing Weights", 32, 32, ax_flip=1)
    create_plot(weights_ex_ih, axs[1, 2], "Ex->Ih Outgoing Weights", 32, 32, ax_flip=1)
    create_plot(
        np.abs(weights_ih_ex), axs[1, 3], "Ih->Ex Outgoing Weights", 15, 15, ax_flip=1
    )

    create_plot(weights_st_ex, axs[2, 0], "St->Ex Incoming Weights", 32, 32, ax_flip=0)
    create_plot(weights_ex_ex, axs[2, 1], "Ex->Ex Incoming Weights", 32, 32, ax_flip=0)
    create_plot(weights_ex_ih, axs[2, 2], "Ex->Ih Incoming Weights", 15, 15, ax_flip=0)
    create_plot(
        np.abs(weights_ih_ex), axs[2, 3], "Ih->Ex Incoming Weights", 32, 32, ax_flip=0
    )

    row_labels = ["Spike activity", "Outgoing weights", "Incoming weights"]

    for i in range(3):
        # get axis bounding box in figure coords
        bbox = axs[i, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2

        fig.text(
            0.02,  # x position (left margin)
            y_center,  # vertical center of row
            row_labels[i],
            va="center",
            ha="left",
            rotation=90,
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle(f"Run: {num}, Label {label}")
    from datetime import datetime

    ts = datetime.now().strftime("%Y.%m.%d")
    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.join("plots", "spikes", dataset, str(label), ts, str(run))
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, f"{ts_spec}.png")
    # save based on class
    fig.savefig(out_path, dpi=100)
    # save for global plotting
    directory = os.path.join("plots", "spikes", dataset, "all", ts, str(run))
    os.makedirs(directory, exist_ok=True)
    out_path_glob = os.path.join(directory, f"{ts_spec}.png")
    fig.savefig(out_path_glob, dpi=100)
    plt.close(fig)  # ✅ important if called in a loop


def gif_spike_rate_by_label(
    frame_folder,
    output_filename="my_awesome.gif",
    duration=100,
    loop=0,
):
    import glob
    from PIL import Image

    # Find all JPG or PNG files in the specified folder
    # Adjust the extension if your files have a different format (e.g., '*.png')
    files = glob.glob(f"{frame_folder}/*.png")
    files_sorted = sorted(files, key=lambda f: int(f.split("\\")[-1].split(".")[0]))
    frames = [Image.open(image) for image in files_sorted]

    if not frames:
        print(f"No images found in {frame_folder}")
        return

    frame_one = frames[0]
    frame_one.save(
        output_filename,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop,
    )

    print("gif made!")


def plot_floats_and_spikes(images, spikes, spike_labels, img_labels, num_steps):
    """
    Given:
      - images: an array of MNIST images (e.g., shape [num_images, H, W])
      - spikes: a 2D array of spike activity (shape: [time, neurons])
      - spike_labels: an array (length equal to the time dimension of spikes)
                      containing the label of the image that produced that spike train.
      - img_labels: an array of labels for the floating images
    This function plots, for each unique image label, the corresponding MNIST image
    (in the bottom row) and a raster plot of the spike data (in the top row).
    """
    # Determine the unique digit labels from the images.
    unique_labels = np.unique(img_labels)
    n_cols = len(unique_labels)

    # Create subplots: one column per digit, two rows (top for spikes, bottom for image)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6))

    # If there's only one column, make sure axs is 2D.
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, label in enumerate(unique_labels):
        # Find the first image with this label
        img_idx = np.where(np.array(img_labels) == label)[0][0]
        # Plot the image in the bottom row
        ax_img = axs[1, i]
        # Ensure the image is 2D (squeeze any singleton dimensions)
        ax_img.imshow(np.squeeze(images[img_idx]), cmap="gray")
        ax_img.set_title(f"Digit {label}")
        ax_img.axis("off")

        # Find all time indices in the spiking data that belong to this label.
        spike_idx_all = np.where(np.array(spike_labels) == label)[0][:num_steps]
        if len(spike_idx_all) == 0:
            print(f"No spiking data found for label {label}.")
            continue

        # Get a contiguous segment from the available indices.
        from src.utils.helper import get_contiguous_segment

        segment = get_contiguous_segment(spike_idx_all)
        if segment is None or len(segment) == 0:
            print(f"No contiguous segment found for label {label}.")
            continue

        # Extract the spike data for this segment.
        spike_segment = spikes[segment, :]  # shape: [time_segment, neurons]

        # For each neuron, determine the time steps (relative to the segment) where it spiked.
        positions = [
            np.where(spike_segment[:, n] == 1)[0] for n in range(spike_segment.shape[1])
        ]

        # Plot the spike raster on the top row.
        ax_spike = axs[0, i]
        ax_spike.eventplot(positions, colors="black")
        ax_spike.set_title(f"Spikes for {label}")
        ax_spike.set_xlabel("Time steps")
        ax_spike.set_ylabel("Neuron")
        # Optionally, adjust y-limits for clarity:
        ax_spike.set_ylim(-1, spike_segment.shape[1])

    plt.tight_layout()

    plt.savefig("plots/comparison_spike_img.png")
    plt.show()


def spike_threshold_plot(spike_threshold, N_exc):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(
        np.mean(spike_threshold[:N_exc], axis=1), label="excitatory", color="green"
    )
    axs[1].plot(
        np.mean(spike_threshold[N_exc:], axis=1), label="inhibitory", color="red"
    )
    axs[0].set_ylabel("spiking threshold (mV)")
    axs[1].set_ylabel("spiking threshold (mV)")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    fig.text(0.5, 0.04, "time (ms)", ha="center")
    fig.suptitle("Average spiking threshold per neuron group over time")
    plt.legend()
    plt.show()


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

        # Find all JPG or PNG files in the specified folder
        # Adjust the extension if your files have a different format (e.g., '*.png')
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
