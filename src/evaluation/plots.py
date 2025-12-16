import os
import random
import argparse
import numpy as np
import matplotlib
from src.utils.platform import configure_matplotlib

configure_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
from src.evaluation.metrics import t_SNE


def get_elite_nodes(spikes, labels, num_classes, narrow_top):

    # remove unnecessary data periods
    mask_break = (labels != -1) & (labels != -2)
    spikes = spikes[mask_break, :]
    labels = labels[mask_break]

    print(f"Debug get_elite_nodes - spikes shape after filtering: {spikes.shape}")
    print(f"Debug get_elite_nodes - labels shape after filtering: {labels.shape}")
    print(f"Debug get_elite_nodes - unique labels after filtering: {np.unique(labels)}")
    print(f"Debug get_elite_nodes - narrow_top: {narrow_top}")

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

    # Now, assign nodes to their preferred class (highest response)
    responses_indices = np.argsort(responses, 0)[::-1, :]
    top_k = int(spikes.shape[1] * narrow_top)

    print(f"Debug get_elite_nodes - total neurons: {spikes.shape[1]}")
    print(f"Debug get_elite_nodes - top_k (neurons per class): {top_k}")
    print(f"Debug get_elite_nodes - total elite neurons: {top_k * num_classes}")

    # Assign top responders
    final_indices = responses_indices[:top_k]

    return final_indices, spikes, labels


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
    ax0.legend(lines, labels, loc="upper center", fontsize=14)

    plt.show()


def top_responders_plotted(
    spikes,
    labels,
    num_classes,
    narrow_top,
    smoothening,
    train,
    compute_not_plot,
    n_last_points=None,
):

    # get indicess
    indices, spikes, labels = get_elite_nodes(
        spikes=spikes,
        labels=labels,
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

        # Debug: Print some statistics
        print(f"Debug accuracy - Total samples: {len(predictions)}")
        print(f"Debug accuracy - Correct predictions: {hit}")
        print(f"Debug accuracy - Final accuracy: {precision[-1]}")
        print(f"Debug accuracy - Prediction distribution: {np.bincount(predictions)}")
        print(f"Debug accuracy - Label distribution: {np.bincount(labels)}")

        # return the final accuracy measurement
        return precision[-1]
    fig, ax = plt.subplots(2, 1)

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

    # Determine the range of points to plot for activity
    if n_last_points is not None and n_last_points < len(acts):
        start_idx = len(acts) - n_last_points
        plot_acts = acts[start_idx:]
        plot_labels = labels[start_idx:]
    else:
        plot_acts = acts
        plot_labels = labels

    # Plot activity for each class
    for c in range(num_classes):
        ax[0].plot(plot_acts[:, c], color=colors[c], label=f"Class {c}")

    # Add the horizontal line below the spikes
    y_offset = 0
    box_height = np.max(plot_acts)

    # We iterate through the time steps to identify contiguous segments
    segment_start = 0
    current_label = plot_labels[0]
    labeled_classes = set()

    # Loop through the labels to draw segments
    for i in range(1, len(plot_labels)):
        if plot_labels[i] != current_label:
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
            current_label = plot_labels[i]
            segment_start = i

    # Handle the final segment
    patch_label = (
        f"Class {current_label}" if current_label not in labeled_classes else None
    )
    rect = patches.Rectangle(
        (segment_start, y_offset),
        len(plot_labels) - segment_start,
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
    return precision[-1]


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
    plt.legend(bbox_to_anchor=(1.1, 0.9), loc="upper right", fontsize=14)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlabel("Time (t)", fontsize=18)
    if test:
        title = f"Testing accuracy: {accuracy[-1]}"
    else:
        title = f"Training and validation accuracy"
    # plt.title(title, fontsize=20, fontweight="bold")
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


def weights_plot():
    # Placeholder for weights_plot if it was missing from original
    pass

def heat_map():
    pass

def mp_plot():
    pass

def plot_weight_evolution():
    pass

def plot_weight_evolution_during_sleep():
    pass

def plot_weight_evolution_during_sleep_epoch():
    pass

def plot_weight_trajectories_with_sleep_epoch():
    pass

def save_weight_distribution_gif():
    pass

# Ensure we export what snn.py uses
__all__ = [
    'spike_plot',
    'weights_plot',
    'heat_map',
    'mp_plot',
    'plot_accuracy',
    'plot_weight_evolution',
    'plot_weight_evolution_during_sleep',
    'plot_weight_evolution_during_sleep_epoch',
    'plot_weight_trajectories_with_sleep_epoch',
    'plot_epoch_training',
    'top_responders_plotted',
    'get_elite_nodes',
    'save_weight_distribution_gif',
]
