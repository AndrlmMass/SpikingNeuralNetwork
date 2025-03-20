import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use("TkAgg")


def spike_plot(data, labels):
    # Validate dimensions
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Labels length ({len(labels)}) must match the number of time steps ({data.shape[0]})."
        )

    # Assign colors to unique labels (excluding -1 if desired)
    valid_label_mask = labels != -1
    unique_labels = np.unique(labels[valid_label_mask])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    # Collect spike positions for each neuron
    positions = [np.where(data[:, n] == 1)[0] for n in range(data.shape[1])]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spikes
    ax.eventplot(positions, lineoffsets=np.arange(data.shape[1]), colors="black")
    ax.set_ylabel(f"{data.shape[1]} Units")
    ax.set_xlabel("Time (ms)")

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


def plot_accuracy(spikes, ih, pp, pn, tp, labels, num_steps, num_classes, test):
    """
    spikes have shape: pp-pn-tp-tn-fp-fn
    """
    pp_ = spikes[:, ih:pp]
    tp_ = spikes[:, pn:tp]

    #### calculate precision (accuracy) ###

    # remove data from all breaks
    mask = labels != -1
    if mask.size != 0:
        labels = labels[mask]
        pp_ = pp_[mask]
        tp_ = tp_[mask]

    # loop through every num_steps time units and compare activity
    total_images = 0
    current_accuracy = 0
    accuracy = np.zeros((labels.shape[0] // num_steps))
    for t in range(0, labels.shape[0], num_steps):
        pp_label = np.sum(pp_[t : t + num_steps], axis=0)
        tp_label = np.sum(tp_[t : t + num_steps], axis=0)
        pp_label_pop = np.argmax(pp_label)
        tp_label_pop = np.argmax(tp_label)
        total_images += 1
        current_accuracy += int(pp_label_pop == tp_label_pop)
        accuracy[t // num_steps] = current_accuracy / total_images
        print(pp_label_pop, tp_label_pop)

    total_images = np.zeros(num_classes)
    current_accuracy = np.zeros(num_classes)
    accuracy2 = np.zeros(((labels.shape[0] // num_steps), num_classes))
    for t in range(0, labels.shape[0], num_steps):
        # calculate most popular class for each t
        pp_label = np.sum(pp_[t : t + num_steps], axis=0)
        tp_label = np.sum(tp_[t : t + num_steps], axis=0)

        # most popular class across all num_steps
        pp_label_pop = np.argmax(pp_label)
        tp_label_pop = np.argmax(tp_label)

        # update number of data points and accumulated accuracy
        total_images[pp_label_pop] += 1
        current_accuracy[pp_label_pop] += int(pp_label_pop == tp_label_pop)

        accuracy2[t // num_steps :, pp_label_pop] = (
            current_accuracy[pp_label_pop] / total_images[pp_label_pop]
        )

    # plot
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for c in range(num_classes):
        class_accuracy = accuracy2[:, c]
        plt.plot(
            class_accuracy,
            label=f"class:{c}",
            color=colors[c],
            linewidth=0.8,
            linestyle="dashed",
        )

    plt.plot(accuracy, label="All classes", linewidth=5, color="black")
    plt.legend(bbox_to_anchor=(1.1, 0.9), loc="upper right")
    plt.ylabel("Accuracy")
    plt.xlabel("Time (t)")
    if test:
        title = f"Testing accuracy: {accuracy[-1]}"
    else:
        title = f"Training accuracy: {accuracy[-1]}"
    plt.title(title)
    plt.show()

    return accuracy[-1]


def get_contiguous_segment(indices):
    """
    Given a sorted 1D array of indices, find contiguous segments
    and return the longest segment.
    """
    if len(indices) == 0:
        return None
    # Find gaps where consecutive indices differ by more than 1
    gaps = np.where(np.diff(indices) != 1)[0]
    segments = np.split(indices, gaps + 1)
    # Return the longest contiguous segment
    return max(segments, key=len)


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


def heat_map(data, pixel_size):
    data = data.numpy()
    summed_data = np.sum(data, axis=0)
    reshaped_summed_data = np.reshape(summed_data, (pixel_size, pixel_size))
    plt.imshow(reshaped_summed_data, cmap="hot", interpolation="nearest")
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


def weights_plot(
    weights_exc,
    weights_inh,
    N,
    N_x,
    N_exc,
    N_inh,
    max_weight_sum_inh,
    max_weight_sum_exc,
    random_selection,
):

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    mu_weights_exc = np.reshape(weights_exc, (weights_exc.shape[0], -1))
    mu_weights_inh = np.reshape(weights_inh, (weights_inh.shape[0], -1))

    idx_exc = np.where(np.sum(mu_weights_exc, axis=0) == 0)
    mu_weights_exc[:, idx_exc[0]] = None

    idx_inh = np.where(np.sum(mu_weights_inh, axis=0) == 0)
    mu_weights_inh[:, idx_inh[0]] = None

    # Define colormap gradients
    cmap_exc = plt.get_cmap("autumn")  # Excitatory in red-yellow shades
    cmap_inh = plt.get_cmap("winter")  # Inhibitory in blue-green shades

    # Plot each excitatory neuron with a unique color
    for i in range(mu_weights_exc.shape[1]):
        axs[0].plot(
            mu_weights_exc[:, i], color=cmap_exc(i / mu_weights_exc.shape[1]), alpha=0.7
        )

    # Plot each inhibitory neuron with a unique color
    for i in range(mu_weights_inh.shape[1]):
        axs[0].plot(
            mu_weights_inh[:, i], color=cmap_inh(i / mu_weights_inh.shape[1]), alpha=0.7
        )
    sum_weights_exc = np.nansum(np.abs(weights_exc), axis=0)
    sum_weights_inh = np.nansum(np.abs(weights_inh), axis=0)
    # plot sum of weights
    axs[1].plot(sum_weights_exc, color="red")
    axs[1].plot(sum_weights_inh, color="blue")
    axs[1].axhline(y=max_weight_sum_exc, color="red", linestyle="--", linewidth=2)
    axs[1].axhline(y=max_weight_sum_inh, color="blue", linestyle="--", linewidth=2)

    # Add legend and show the plot
    # Add titles and labels appropriately
    axs[0].set_title("Weight Evolution Over Time (Individual Neurons)")
    axs[1].set_title("Total Weight Evolution Over Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Synaptic Weight")
    plt.show()


def plot_traces(
    random_selection=False,
    num_exc=None,
    num_inh=None,
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
