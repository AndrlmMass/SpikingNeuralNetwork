import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.ticker import AutoMinorLocator

matplotlib.use("TkAgg")


def get_elite_nodes(spikes, labels, num_classes, narrow_top, st, ih):
    # remove unnecessary data periods
    mask_break = (labels != -1) & (labels != -2)
    spikes = spikes[mask_break, :]
    labels = labels[mask_break]

    # remove poisson-input & inhibition
    spikes = spikes[:, st:ih]

    # collect responses
    responses = np.zeros(
        (spikes.shape[1], num_classes), dtype=float
    )  # make responses float too

    for cl in range(num_classes):
        indices = np.where(labels == cl)[0]
        summed = np.sum(spikes[indices], axis=0)  # still int at this point
        response = summed.astype(float)  # now convert to float
        response[response == 0] = np.nan  # safe to assign NaN
        responses[:, cl] = response

    # compute discriminatory power
    total_responses = np.sum(spikes, axis=0, dtype=float)
    total_responses[total_responses == 0] = np.nan
    total_responses_reshaped = np.tile(total_responses, (num_classes, 1)).T
    ratio = responses / total_responses_reshaped
    responses *= ratio

    # sort performance
    responses_indices = np.argsort(responses, 0)[::-1, :]
    top_k = int(spikes.shape[1] * narrow_top)

    final_indices = responses_indices[:top_k]

    return final_indices, spikes, labels


def top_responders_plotted(
    spikes,
    labels,
    ih,
    st,
    num_classes,
    narrow_top,
    smoothening,
    train,
    compute_not_plot,
):
    fig, ax = plt.subplots(2, 1)

    # get indicess
    indices, spikes, labels = get_elite_nodes(
        spikes=spikes,
        labels=labels,
        ih=ih,
        st=st,
        num_classes=num_classes,
        narrow_top=narrow_top,
    )

    if compute_not_plot:
        block_size = smoothening

        # Calculate the number of complete blocks
        num_blocks = spikes.shape[0] // block_size

        # Initialize a list to hold the mean of each block
        means = []
        labs = []

        # Loop through each block, calculate mean along axis=0 (i.e. column-wise)
        for i in range(num_blocks):
            # add spikes
            block = spikes[i * block_size : (i + 1) * block_size]
            block_mean = np.mean(block, axis=0)
            means.append(block_mean)
            # add labels
            block_lab = labels[i * block_size : (i + 1) * block_size]
            block_maj = np.argmax(np.bincount(block_lab))
            labs.append(block_maj)

        # Optionally convert to a NumPy array for further processing
        spikes = np.array(means)
        labels = np.array(labs)

        acts = np.zeros((spikes.shape[0], num_classes))
        for c in range(num_classes):
            acts[:, c] = np.sum(spikes[:, indices[:, c]], axis=1)

        predictions = np.argmax(acts, axis=1)
        precision = np.zeros(spikes.shape[0])
        hit = 0
        for i in range(precision.shape[0]):
            hit += predictions[i] == labels[i]
            precision[i] = hit / (i + 1)

        # return the final accuracy measurement
        return precision[-1]

    # reduce samples
    cmap = plt.get_cmap("Set3", num_classes)
    colors = cmap.colors

    # Define an intensity factor (values between 0 and 1)
    intensity_factor = 0.5  # 70% of the original brightness

    # Reduce the intensity of each color by scaling its RGB components
    colors_adjusted = [
        tuple(np.clip(np.array(color) * intensity_factor, 0, 1)) for color in colors
    ]

    block_size = smoothening

    # Calculate the number of complete blocks
    num_blocks = spikes.shape[0] // block_size

    # Initialize a list to hold the mean of each block
    means = []
    labs = []

    # Loop through each block, calculate mean along axis=0 (i.e. column-wise)
    for i in range(num_blocks):
        # add spikes
        block = spikes[i * block_size : (i + 1) * block_size]
        block_mean = np.mean(block, axis=0)
        means.append(block_mean)
        # add labels
        block_lab = labels[i * block_size : (i + 1) * block_size]
        block_maj = np.argmax(np.bincount(block_lab))
        labs.append(block_maj)

    # Optionally convert to a NumPy array for further processing
    spikes = np.array(means)
    labels = np.array(labs)

    acts = np.zeros((spikes.shape[0], num_classes))
    for c in range(num_classes):
        activity = np.sum(spikes[:, indices[:, c]], axis=1)
        acts[:, c] = activity
        ax[0].plot(activity, color=colors[c], label=f"Class {c}")

    # Add the horizontal line below the spikes
    y_offset = 0
    box_height = np.max(acts)

    # We iterate through the time steps to identify contiguous segments
    segment_start = 0
    current_label = labels[0]
    labeled_classes = set()

    # Loop through the labels to draw segments
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # Draw a rectangle patch for the segment that just ended
            rect = patches.Rectangle(
                (segment_start, y_offset),
                i - segment_start,  # width of the rectangle
                box_height,  # height of the rectangle
                linewidth=2,
                facecolor=colors_adjusted[current_label],
            )
            ax[0].add_patch(rect)

            # Mark this class as having been labeled
            labeled_classes.add(current_label)

            # Update for the new segment
            current_label = labels[i]
            segment_start = i

    # Handle the final segment
    patch_label = (
        f"Class {current_label}" if current_label not in labeled_classes else None
    )
    rect = patches.Rectangle(
        (segment_start, y_offset),
        len(labels) - segment_start,
        box_height,
        linewidth=2,
        edgecolor=colors_adjusted[current_label],
        facecolor=colors_adjusted[current_label],
        label=patch_label,
    )
    ax[0].add_patch(rect)

    if train:
        title = "Top responding nodes by class during training"
    else:
        title = "Top responding nodes by class during testing"
    ax[0].set_ylabel("Spiking rate")

    """
    Plot accuracy in second plot
    """
    predictions = np.argmax(acts, axis=1)
    precision = np.zeros(spikes.shape[0])
    hit = 0
    for i in range(precision.shape[0]):
        hit += predictions[i] == labels[i]
        precision[i] = hit / (i + 1)

    ax[1].plot(precision)
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].set_xlabel(f"Time (intervals of {smoothening} ms)")
    ax[0].set_title(title)
    ax[0].legend(loc="upper right")
    plt.show()


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
        pp_ = pp_[mask, :]
        tp_ = tp_[mask, :]

    # loop through every num_steps time units and compare activity
    total_images = 0
    current_accuracy = 0
    accuracy = np.zeros((labels.shape[0] // num_steps) + 1)
    total_images2 = np.zeros(num_classes)
    current_accuracy2 = np.zeros(num_classes)
    accuracy2 = np.zeros(((labels.shape[0] // num_steps) + 1, num_classes))
    for t in range(0, labels.shape[0] + 1, num_steps):
        pp_label = np.sum(pp_[t : t + num_steps], axis=0)
        tp_label = np.sum(tp_[t : t + num_steps], axis=0)

        # check if there is no class preference
        if np.sum(tp_label) == 0:
            accuracy[t // num_steps] = accuracy[(t - 1) // num_steps]
        else:
            """
            Look over this logic again. I think argmax might be wrong.
            """
            pp_label_pop = np.argmax(pp_label)
            tp_label_pop = np.argmax(tp_label)
            total_images += 1
            current_accuracy += int(pp_label_pop == tp_label_pop)
            accuracy[t // num_steps] = current_accuracy / total_images

            # update number of data points and accumulated accuracy
            total_images2[tp_label_pop] += 1
            current_accuracy2[tp_label_pop] += int(pp_label_pop == tp_label_pop)
            acc = current_accuracy2[tp_label_pop] / total_images2[tp_label_pop]
            accuracy2[t // num_steps :, tp_label_pop] = acc

    # plot
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for c in range(num_classes):
        class_accuracy = accuracy2[:, c]
        # Add jitter to the x-values for this class
        jitter = np.random.normal(0, 0.001, size=class_accuracy.shape[0])
        plt.plot(
            class_accuracy + jitter,
            label=f"class:{c}",
            color=colors[c],
            linewidth=0.8,
            linestyle="dashed",
        )

    plt.plot(accuracy, label="All classes", linewidth=3, color="black")
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
    # plt.plot(mp[:, :N_exc], color="green")
    # plt.title("excitatory membrane potential during training")
    # plt.xlabel("time (ms)")
    # plt.ylabel("membrane potential (mV)")
    # plt.show()

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
    sum_weights_exc = np.abs(np.sum(weights_exc, axis=2))
    sum_weights_inh = np.abs(np.sum(weights_inh, axis=2))
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


def plot_phi_bars(phi_means, sleep_lengths, sleep_amount):
    """
    phi_results structure = (2, 6) # first dim: len(sleep_lengths), second dim: phi_train, phi_test, WCSS_train, WCSS_test, BCSS_train, BCSS_test
    """
    x = np.char.mod("%.2f%%", sleep_amount)
    y = phi_means[:, 1]
    colors = get_blue_colors(sleep_amount.shape[0])
    for i in range(sleep_lengths.shape[0]):
        plt.bar(x[i], y[i], label=f"$\\eta = {sleep_lengths[i]}$", color=colors[i])
    plt.ylabel("Clustering score ($\\phi$)", fontsize=14)
    plt.xlabel("Sleep amount ($\\%$)", fontsize=14)
    plt.legend(loc=(0.97, 0.75))
    plt.savefig("plot_phi_bars.png")
    plt.show()


def get_blue_colors(n):
    # Using the reversed Blues colormap ensures that:
    cmap = plt.cm.Blues_r
    # Generate n values from 0 (dark) to 1 (light) and get the corresponding color for each
    colors = [cmap(x) for x in np.linspace(0, 0.7, n)]
    return colors


def plot_phi_acc(all_scores):
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
