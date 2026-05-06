import os
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display

import pandas as pd
import seaborn as sns
from PIL import Image
import glob

from typing import Optional


def plot_glmm_predictions(
    pred_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot GLMM predicted values with confidence intervals.

    Args:
        pred_path: Path to pred.xlsx (output from mixed_model2.r)
        output_path: Path to save the figure
        show_plot: Whether to display the plot

    Returns:
        Path to saved figure
    """
    data = pd.read_excel(pred_path)

    sleep_vals = sorted(data["x"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    data["x"] = data["x"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}

    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "v"}
    cluster_width = 0.8

    g = sns.FacetGrid(
        data=data,
        col="Dataset" if "Dataset" in data.columns else "facet",
        col_order=dataset_order,
        sharey=True,
        height=4,
        aspect=1.1,
    )

    base_index = {s: i for i, s in enumerate(order)}

    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = (
            data["Dataset"].iloc[0]
            if "Dataset" in data.columns
            else data["facet"].iloc[0]
        )
        subdf = data.copy()

        if subdf.empty:
            return

        models_here = [m for m in hue_order if m in subdf["group"].unique()]
        k = len(models_here)
        if k == 0:
            return

        offset = {
            m: (j - (k - 1) / 2) * (cluster_width / k)
            for j, m in enumerate(models_here)
        }

        for m in models_here:
            style = styles[m]
            sub_m = subdf[subdf["group"] == m].copy()
            sub_m["__ord__"] = sub_m["x"].map(order_map)
            sub_m = sub_m.sort_values("__ord__")

            if sub_m.empty:
                continue

            xs = [order_map[s] + offset[m] for s in sub_m["x"]]
            y_mean = sub_m["predicted"].to_numpy()
            y_max = sub_m["conf.high"].to_numpy()
            y_min = sub_m["conf.low"].to_numpy()

            yerr_lower = y_mean - y_min
            yerr_upper = y_max - y_mean

            plt.errorbar(
                xs,
                y_mean,
                yerr=[yerr_lower, yerr_upper],
                fmt=style,
                markersize=5,
                linestyle="none",
                capsize=4,
                elinewidth=1.2,
                color="black",
                label=m if dataset == dataset_order[0] else None,
            )
            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)

        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.2)

    g.map_dataframe(draw_errorbars)

    new_titles = {
        "mnist": "MNIST",
        "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST",
        "notmnist": "NotMNIST",
    }

    for ax, ds in zip(g.axes.flatten(), dataset_order):
        ax.set_title(new_titles.get(ds, ds), fontsize=18)

    g.set_ylabels("Accuracy (%)", fontsize=16)

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if handles:
        g.fig.legend(
            handles, labels, title="Model", bbox_to_anchor=(0.14, 0.4), framealpha=1.0
        )

    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    output_path = output_path or "sleep_duration_comparison_glmm.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")

    if show_plot:
        plt.show()
    plt.close()

    return output_path


def plot_glmm_with_raw_accuracy(
    pred_path: str,
    results_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot GLMM predictions overlaid with raw observed accuracy.

    Args:
        pred_path: Path to pred.xlsx
        results_path: Path to Results_.xlsx
        output_path: Path to save figure
        show_plot: Whether to display

    Returns:
        Path to saved figure
    """
    # Load GLMM predictions
    data = pd.read_excel(pred_path)
    data = data.rename(
        columns={"x": "Sleep_duration", "group": "Model", "facet": "Dataset"}
    )

    # Load raw results
    data2 = pd.read_excel(results_path)
    data2 = data2[data2["Run"] == 1]
    data2 = data2.drop(["Seed", "Run"], axis=1, errors="ignore")

    # Data is already in long format (Dataset and Accuracy columns)
    # No need to melt
    ldata = data2

    stats = (
        ldata.groupby(["Model", "Sleep_duration", "Dataset"])["Accuracy"]
        .agg(mean="mean", min="min", max="max")
        .reset_index()
    )

    sleep_vals = sorted(data["Sleep_duration"].dropna().unique())
    order = [str(int(v)) for v in sleep_vals]
    stats["Sleep_duration"] = stats["Sleep_duration"].astype(str)
    data["Sleep_duration"] = data["Sleep_duration"].astype(str)
    order_map = {v: i for i, v in enumerate(order)}

    dataset_order = ["mnist", "kmnist", "fmnist", "notmnist"]
    hue_order = ["SNN_sleepy", "snntorch"]
    styles = {"SNN_sleepy": "o", "snntorch": "v"}
    cluster_width = 0.8

    g = sns.FacetGrid(
        data=data,
        col="Dataset",
        col_order=dataset_order,
        sharey=True,
        height=4,
        aspect=1.1,
    )

    def draw_errorbars(data, color=None, **kwargs):
        ax = plt.gca()
        dataset = data["Dataset"].iloc[0]

        subdf = data.copy()
        subdf2 = stats[stats["Dataset"] == dataset].copy()

        if subdf.empty:
            return

        models_here = [m for m in hue_order if m in subdf["Model"].unique()]
        k = len(models_here)
        if k == 0:
            return

        offset = {
            m: (j - (k - 1) / 2) * (cluster_width / k)
            for j, m in enumerate(models_here)
        }

        for m in models_here:
            style = styles[m]
            sub_m = subdf[subdf["Model"] == m].copy()
            sub_m2 = subdf2[subdf2["Model"] == m].copy()

            sub_m["__ord__"] = sub_m["Sleep_duration"].map(order_map)
            sub_m = sub_m.sort_values("__ord__")
            sub_m2["__ord__"] = sub_m2["Sleep_duration"].map(order_map)
            sub_m2 = sub_m2.sort_values("__ord__")

            if sub_m.empty:
                continue

            xs = [order_map[s] + offset[m] for s in sub_m["Sleep_duration"]]
            y_mean2 = sub_m2["mean"].to_numpy() if not sub_m2.empty else []
            y_mean = sub_m["predicted"].to_numpy()
            y_max = sub_m["conf.high"].to_numpy()
            y_min = sub_m["conf.low"].to_numpy()

            yerr_lower = y_mean - y_min
            yerr_upper = y_max - y_mean

            # Predicted with CI
            plt.errorbar(
                xs,
                y_mean,
                yerr=[yerr_lower, yerr_upper],
                fmt=style,
                markersize=5,
                linestyle="none",
                capsize=4,
                elinewidth=1.2,
                color="black",
                label=f"{m} predicted mean" if dataset == dataset_order[0] else None,
            )

            # Observed means (hollow markers)
            if len(y_mean2) > 0:
                plt.scatter(
                    xs,
                    y_mean2,
                    marker=style,
                    facecolors="none",
                    edgecolors="black",
                    s=50,
                    label=f"{m} observed mean" if dataset == dataset_order[0] else None,
                )

            plt.plot(xs, y_mean, linestyle="--", color="black", linewidth=1)

        ax.set_xticks(list(order_map.values()))
        ax.set_xticklabels(order, fontsize=18)
        ax.set_xlabel("Sleep_duration", fontsize=16)
        ax.set_ylim(top=1.0, bottom=0.0)

    g.map_dataframe(draw_errorbars)

    new_titles = {
        "mnist": "MNIST",
        "kmnist": "KMNIST",
        "fmnist": "Fashion-MNIST",
        "notmnist": "NotMNIST",
    }

    for ax, ds in zip(g.axes.flatten(), dataset_order):
        ax.set_title(new_titles.get(ds, ds), fontsize=18)

    g.set_ylabels("Accuracy (%)", fontsize=16)

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if handles:
        g.fig.legend(
            handles, labels, title="Model", bbox_to_anchor=(0.23, 0.55), framealpha=1.0
        )

    sns.despine(offset=10, trim=True)

    output_path = output_path or "sleep_duration_comparison_with_accuracy.pdf"
    plt.savefig(output_path, dpi=900)
    print(f"Figure saved to {output_path}")

    if show_plot:
        plt.show()
    plt.close()

    return output_path


def block_reduce(spikes, labels, block_size, reduce="sum"):
    """Convert (T, N) spikes into (B, N) blocks and majority labels per block."""
    if block_size <= 1:
        return spikes, labels

    # Drop break / sleep markers (negative labels) before forming blocks
    valid_mask = labels >= 0
    spikes = spikes[valid_mask]
    labels = labels[valid_mask]

    num_blocks = spikes.shape[0] // block_size
    if num_blocks == 0:
        return spikes[:0], labels[:0]

    spikes = spikes[: num_blocks * block_size]
    labels = labels[: num_blocks * block_size]

    blocks = spikes.reshape(num_blocks, block_size, spikes.shape[1])
    if reduce == "mean":
        spikes_b = blocks.mean(axis=1)
    elif reduce == "sum":
        spikes_b = blocks.sum(axis=1)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")

    labels_b = np.zeros(num_blocks, dtype=int)
    for i in range(num_blocks):
        lab_block = labels[i * block_size : (i + 1) * block_size]
        # lab_block is now guaranteed non-negative
        labels_b[i] = np.argmax(np.bincount(lab_block))

    return spikes_b, labels_b


def get_elite_nodes_wta(spikes, labels, num_classes, min_total_spikes=10):
    mask = (labels >= 0) & (labels < num_classes)
    spikes = spikes[mask]
    labels = labels[mask]

    N = spikes.shape[1]
    # top_k = int(N * narrow_top)

    responses = np.zeros((N, num_classes), dtype=float)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size > 0:
            responses[:, c] = spikes[idx].sum(axis=0)

    total = responses.sum(axis=1)
    score = np.full_like(responses, -np.inf)
    valid = total >= min_total_spikes
    score[valid] = responses[valid] / total[valid, None]

    pref = np.full(N, -1, dtype=int)
    pref[valid] = np.argmax(score[valid], axis=1)

    return pref, score


def zscore_vote(
    spikes, pref, baseline_mu, baseline_sigma, num_classes, eps=1e-8, mu_min=1e-3
):
    """
    spikes: (B, N) block-reduced spikes
    pref: (N,) neuron->class assignment
    baseline_mu: (N,) mean firing from fit snippet
    baseline_sigma: (N,) std firing from fit snip    # 4) class activations
    acts = np.zeros((B, num_classes), dtype=float)pet
    """
    valid_neurons = baseline_mu >= mu_min

    # Avoid divide-by-zero
    sigma = np.maximum(baseline_sigma, eps)

    z = np.zeros_like(spikes, dtype=float)
    z[:, valid_neurons] = (spikes[:, valid_neurons] - baseline_mu[valid_neurons]) / (
        sigma[valid_neurons] + eps
    )
    z = np.maximum(z, 0.0)

    # 4) class activations
    acts = np.zeros((spikes.shape[0], num_classes), dtype=float)

    for c in range(num_classes):
        idx = np.where((pref == c) & valid_neurons)[0]
        if idx.size:
            acts[:, c] = z[:, idx].mean(axis=1)

    pred = np.argmax(acts, axis=1)
    return pred, acts


def WTA_accuracy(
    spikes,
    labels,
    num_classes,
    smoothening,
    split,
    state=None,
    fit_frac=0.8,
    reduce="sum",
    min_total_spikes=1,
):
    # ---- block reduce first (same unit for baseline & test) ----
    spikes_b, labels_b = block_reduce(spikes, labels, smoothening, reduce=reduce)

    if split == "train":
        cut = int(len(labels_b) * fit_frac)
        spikes_fit, labels_fit = spikes_b[:cut], labels_b[:cut]
        spikes_test, labels_test = spikes_b[cut:], labels_b[cut:]

        pref, score = get_elite_nodes_wta(
            spikes_fit, labels_fit, num_classes, min_total_spikes
        )
        baseline_mu = spikes_fit.mean(axis=0)
        baseline_sigma = spikes_fit.std(axis=0)

        state = {
            "pref": pref,
            "baseline_mu": baseline_mu,
            "baseline_sigma": baseline_sigma,
        }

    elif split == "test" or split == "val":
        if state is None:
            raise ValueError("Pass state from train (pref + baseline_mu).")
        pref = state["pref"]
        baseline_mu = state["baseline_mu"]
        baseline_sigma = state["baseline_sigma"]
        spikes_test, labels_test = spikes_b, labels_b

    else:
        raise ValueError("split must be 'train' or 'test'")

    # loop over each item and plot mean spikes per neuron
    # for i in range(len(labels_b)):
    #     fig, ax = plt.subplots()
    #     ax.bar(range(spikes_b.shape[1]), spikes_b[i])
    #     ax.set_ylabel("Spikes")
    #     ax.set_xlabel("Neuron")
    #     ax.set_title(f"Class {labels_b[i]}")
    #     idxs = np.where(pref == labels_b[i])[0]
    #     response = np.sum(spikes_b[i, idxs])
    #     population_response = np.sum(spikes_b[i])
    #     for p in idxs:
    #         ax.axvline(p, color="red", linewidth=1, alpha=0.2)
    #     ax.legend(["Pref", "Spikes", f"Response: {response:.2f}", f"Population: {population_response:.2f}"])
    #     plt.savefig(f"plots/spikes_item_{i}.png")
    #     plt.show()

    pred, acts = zscore_vote(
        spikes_test,
        pref,
        baseline_mu,
        baseline_sigma,
        num_classes,
    )
    # create bar plot of acts per item to gauge response in spikes and the comparable z-score

    acc = (pred == labels_test).mean()

    # Clip to non-negative before bincount to avoid errors
    pred_safe = np.clip(pred, 0, num_classes - 1)
    labels_test_safe = np.clip(labels_test, 0, num_classes - 1)
    pref_safe = np.clip(pref, 0, num_classes - 1)

    debug = {
        "acc": float(acc),
        "pred_dist": np.bincount(pred_safe, minlength=num_classes),
        "label_dist": np.bincount(labels_test_safe, minlength=num_classes),
        "neurons_per_class": np.bincount(pref_safe, minlength=num_classes),
    }
    print(debug)
    return acc, debug, state

    # fig, ax = plt.subplots(2, 1)

    # # reduce samples
    # cmap = plt.get_cmap("Set3", num_classes)
    # colors = cmap.colors

    # # Define an intensity factor (values between 0 and 1)
    # intensity_factor = 0.5  # 70% of the original brightness

    # # Reduce the intensity of each color by scaling its RGB components
    # colors_adjusted = [
    #     tuple(np.clip(np.array(color) * intensity_factor, 0, 1)) for color in colors
    # ]

    # block_size = smoothening

    # # Calculate the number of complete blocks
    # num_blocks = spikes.shape[0] // block_size

    # # Initialize a list to hold the mean of each block
    # means = []
    # labs = []

    # # Loop through each block, calculate mean along axis=0 (i.e. column-wise)
    # for i in range(num_blocks):
    #     # add spikes
    #     block = spikes[i * block_size : (i + 1) * block_size]
    #     block_mean = np.mean(block, axis=0)
    #     means.append(block_mean)
    #     # add labels
    #     block_lab = labels[i * block_size : (i + 1) * block_size]
    #     block_maj = np.argmax(np.bincount(block_lab))
    #     labs.append(block_maj)

    # # Optionally convert to a NumPy array for further processing
    # spikes = np.array(means)
    # labels = np.array(labs)

    # acts = np.zeros((spikes.shape[0], num_classes))
    # for c in range(num_classes):
    #     activity = np.sum(spikes[:, indices[:, c]], axis=1)
    #     acts[:, c] = activity

    # # Determine the range of points to plot for activity
    # if n_last_points is not None and n_last_points < len(acts):
    #     start_idx = len(acts) - n_last_points
    #     plot_acts = acts[start_idx:]
    #     plot_labels = labels[start_idx:]
    # else:
    #     plot_acts = acts
    #     plot_labels = labels

    # # Plot activity for each class
    # for c in range(num_classes):
    #     ax[0].plot(plot_acts[:, c], color=colors[c], label=f"Class {c}")

    # # Add the horizontal line below the spikes
    # y_offset = 0
    # box_height = np.max(plot_acts)

    # # We iterate through the time steps to identify contiguous segments
    # segment_start = 0
    # current_label = plot_labels[0]
    # labeled_classes = set()

    # # Loop through the labels to draw segments
    # for i in range(1, len(plot_labels)):
    #     if plot_labels[i] != current_label:
    #         # Draw a rectangle patch for the segment that just ended
    #         rect = patches.Rectangle(
    #             (segment_start, y_offset),
    #             i - segment_start,  # width of the rectangle
    #             box_height,  # height of the rectangle
    #             linewidth=2,
    #             facecolor=colors_adjusted[current_label],
    #         )
    #         ax[0].add_patch(rect)

    #         # Mark this class as having been labeled
    #         labeled_classes.add(current_label)

    #         # Update for the new segment
    #         current_label = plot_labels[i]
    #         segment_start = i

    # # Handle the final segment
    # patch_label = (
    #     f"Class {current_label}" if current_label not in labeled_classes else None
    # )
    # rect = patches.Rectangle(
    #     (segment_start, y_offset),
    #     len(plot_labels) - segment_start,
    #     box_height,
    #     linewidth=2,
    #     edgecolor=colors_adjusted[current_label],
    #     facecolor=colors_adjusted[current_label],
    #     label=patch_label,
    # )
    # ax[0].add_patch(rect)

    # if train:
    #     title = "Top responding nodes by class during training"
    # else:
    #     title = "Top responding nodes by class during testing"
    # ax[0].set_ylabel("Spiking rate")

    # """
    # Plot accuracy in second plot
    # """
    # predictions = np.argmax(acts, axis=1)
    # precision = np.zeros(spikes.shape[0])
    # hit = 0
    # for i in range(precision.shape[0]):
    #     hit += predictions[i] == labels[i]
    #     precision[i] = hit / (i + 1)

    # ax[1].plot(precision)
    # ax[1].set_ylabel("Accuracy (%)")
    # ax[1].set_xlabel(f"Time (intervals of {smoothening} ms)")
    # ax[0].set_title(title)
    # ax[0].legend(loc="upper right")
    # plt.show()
    # return precision[-1]


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


def plot_audio_spectrograms_and_spikes(
    audio_data, spikes, spike_labels, audio_labels, num_steps, sample_rate=22050
):
    """
    Given:
      - audio_data: an array of audio samples (e.g., shape [num_samples, audio_length])
      - spikes: a 2D array of spike activity (shape: [time, neurons])
      - spike_labels: an array (length equal to the time dimension of spikes)
                      containing the label of the audio that produced that spike train.
      - audio_labels: an array of labels for the audio samples
      - num_steps: number of time steps for spike generation
      - sample_rate: sample rate of the audio data
    This function plots, for each unique audio label, the corresponding spectrogram
    (in the bottom row) and a raster plot of the spike data (in the top row).
    """
    # Always target classes 0-3; keep empty columns if some are missing
    labels_to_plot = [0, 1, 2, 3]
    n_cols = 4

    # Create subplots: one column per digit, two rows (top for spikes, bottom for spectrogram)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))

    for i, label in enumerate(labels_to_plot):
        # Plot axes for this column
        ax_spec = axs[1, i]
        ax_spike = axs[0, i]
        # Find the first audio sample with this label (if none, show placeholder)
        idxs = np.where(np.array(audio_labels) == label)[0]
        if idxs.size == 0:
            ax_spec.text(
                0.5, 0.5, f"No audio for digit {label}", ha="center", va="center"
            )
            ax_spec.set_ylabel("Frequency (Hz)")
            ax_spec.set_xlabel("Time (s)")
            ax_spike.text(
                0.5, 0.5, f"No spikes for digit {label}", ha="center", va="center"
            )
            ax_spike.set_xlabel("Time steps")
            ax_spike.set_ylabel("Neuron")
            continue

        audio_idx = idxs[0]

        # Compute spectrogram
        audio_sample = audio_data[audio_idx]
        stft = librosa.stft(audio_sample)
        magnitude = np.abs(stft)
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Display spectrogram
        librosa.display.specshow(
            log_magnitude,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec,
            cmap="viridis",
        )
        # Reduce x-axis label density (wider spacing)
        try:
            from matplotlib import ticker as _ticker

            ax_spec.xaxis.set_major_locator(_ticker.MaxNLocator(nbins=4))
        except Exception:
            pass
        ax_spec.set_title(f"Digit {label}")

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
        ax_spike.eventplot(positions, colors="black")
        ax_spike.set_title(f"Digit {label}")
        # Optionally, adjust y-limits for clarity:
        ax_spike.set_ylim(-1, spike_segment.shape[1])

    plt.tight_layout()

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/comparison_spike_audio.png")
    plt.show()


def plot_audio_preview_from_streamer(
    audio_streamer,
    num_steps,  # e.g., 1000
    N_x,
    training_mode="audio_only",
    max_batches=50,
    batch_size=500,
    sample_rate=22050,
):
    """
    Collect up to one sample for each class in [0,1,2,3] from the streamer,
    generate spikes, and render the spectrogram + spikes grid.
    """
    import numpy as np
    from get_data import (
        cochlear_to_spikes_1s,
    )  # expects: (wav_batch_list, sr=..., n_channels=..., out_T_ms=...)

    targets = [0, 1, 2, 3]
    have = {k: False for k in targets}
    audio_samples = []
    audio_labels = []
    spike_chunks = []
    spike_labels = []

    # How many cochlear channels to allocate (must match your plotting grid expectations)
    if training_mode == "multimodal":
        num_audio_neurons = int(np.sqrt(N_x // 2)) ** 2
    else:
        num_audio_neurons = int(np.sqrt(N_x)) ** 2

    start = 0
    for _ in range(max_batches):
        batch, labels = audio_streamer.get_batch(start, batch_size)
        if batch is None:
            break

        # batch is a list of 1D waveforms; labels is a numpy array
        for t in targets:
            if have[t]:
                continue
            idxs = np.where(labels == t)[0]
            if idxs.size == 0:
                continue

            i0 = idxs[0]
            wav = batch[i0]  # 1D np.array
            audio_samples.append(wav)
            audio_labels.append(t)

            # IMPORTANT: pass a *list* of waveforms; set sr, n_channels, and out_T_ms
            spikes_bt = cochlear_to_spikes_1s(
                [wav],
                sr=sample_rate,
                n_channels=num_audio_neurons,
                out_T_ms=num_steps,
                return_rates=False,
            )  # shape: (B=1, num_steps, num_audio_neurons)

            spike_chunks.append(spikes_bt)  # list of (1, T, F)
            spike_labels.extend(
                [t] * num_steps
            )  # one label per timestep for this sample
            have[t] = True

        if all(have.values()):
            break
        start += batch_size

    if len(audio_samples) == 0:
        print("No audio samples found for preview.")
        return

    # Concatenate on batch axis -> (num_found, T, F), then flatten time axis -> (num_found*T, F)
    spike_data_3d = np.concatenate(spike_chunks, axis=0)
    spike_data = spike_data_3d.reshape(-1, spike_data_3d.shape[-1])

    # Hand off to your plotting util
    plot_audio_spectrograms_and_spikes(
        audio_data=np.array(audio_samples, dtype=object),  # variable-length waves
        spikes=spike_data,
        spike_labels=np.array(spike_labels),
        audio_labels=np.array(audio_labels),
        num_steps=num_steps,
        sample_rate=sample_rate,
    )


def plot_audio_spectrograms_and_spikes_simple(
    audio_samples, spike_data, spike_labels, audio_labels, num_steps, sample_rate=22050
):
    """
    Simplified version that works with pre-loaded data.
    Given:
      - audio_samples: list of audio samples (each is a 1D array)
      - spike_data: 2D array of spike activity (shape: [time, neurons])
      - spike_labels: array of labels for spike data
      - audio_labels: array of labels for audio samples
      - num_steps: number of time steps for spike generation
      - sample_rate: sample rate of the audio data
    """
    # Determine labels to plot (restrict to classes 0-3)
    unique_labels = np.unique(audio_labels)
    labels_to_plot = [lbl for lbl in [0, 1, 2, 3] if lbl in unique_labels]
    if len(labels_to_plot) == 0:
        labels_to_plot = list(unique_labels[:4])
    n_cols = max(1, len(labels_to_plot))

    # Create subplots: one column per digit, two rows (top for spikes, bottom for spectrogram)
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))

    # If there's only one column, make sure axs is 2D.
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, label in enumerate(labels_to_plot):
        # Find the first audio sample with this label
        audio_idx = np.where(np.array(audio_labels) == label)[0][0]

        # Plot the spectrogram in the bottom row
        ax_spec = axs[1, i]

        # Compute spectrogram
        audio_sample = audio_samples[audio_idx]
        stft = librosa.stft(audio_sample)
        magnitude = np.abs(stft)
        log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Display spectrogram
        librosa.display.specshow(
            log_magnitude,
            sr=sample_rate,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec,
            cmap="viridis",
        )
        # Reduce x-axis label density (wider spacing)
        try:
            from matplotlib import ticker as _ticker

            ax_spec.xaxis.set_major_locator(_ticker.MaxNLocator(nbins=4))
        except Exception:
            pass
        ax_spec.set_title(f"Audio Spectrogram - Digit {label}")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_xlabel("Time (s)")

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
        spike_segment = spike_data[segment, :]  # shape: [time_segment, neurons]

        # For each neuron, determine the time steps (relative to the segment) where it spiked.
        positions = [
            np.where(spike_segment[:, n] == 1)[0] for n in range(spike_segment.shape[1])
        ]

        # Plot the spike raster on the top row.
        ax_spike = axs[0, i]
        ax_spike.eventplot(positions, colors="black")
        ax_spike.set_title(f"Spikes for Audio {label}")
        ax_spike.set_xlabel("Time steps")
        ax_spike.set_ylabel("Neuron")
        # Optionally, adjust y-limits for clarity:
        ax_spike.set_ylim(-1, spike_segment.shape[1])

    plt.tight_layout()

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/comparison_spike_audio.png")
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
    sum_weights_exc = np.nansum(mu_weights_exc, axis=1)
    sum_weights_inh = np.nansum(mu_weights_inh, axis=1)
    # plot sum of weights
    axs[1].plot(sum_weights_exc, color="red", label="excitatory")
    axs[1].plot(sum_weights_inh, color="blue", label="inhibitory")

    # Add legend and show the plot
    # Add titles and labels appropriately
    axs[0].set_title("Weight Evolution Over Time (Individual Neurons)")
    axs[1].set_title("Total Weight Evolution Over Time")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Synaptic Weight")
    plt.show()


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


def plot_weight_evolution_during_sleep_epoch(weight_tracking_epoch, epoch):
    """Plot weight changes during sleep for a single epoch, overlaying all sleep periods."""
    import matplotlib.pyplot as plt
    import shutil
    from matplotlib.lines import Line2D

    times = np.array(weight_tracking_epoch["times"])
    exc_mean = np.array(weight_tracking_epoch["exc_mean"])
    exc_samples = np.array(weight_tracking_epoch["exc_samples"])  # (n_t, n_s)
    inh_mean = np.array(weight_tracking_epoch["inh_mean"])
    inh_samples = np.array(weight_tracking_epoch["inh_samples"])  # (n_t, n_s)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Excitatory
    ax = axes[0]
    ax.plot(
        times, exc_mean, color="black", linestyle="-", linewidth=2.0, label="Exc mean"
    )
    # Sampled connections (thin)
    if exc_samples.ndim == 2 and exc_samples.shape[0] == times.shape[0]:
        # Use mono-color (black) and vary only linestyle; avoid solid which is used by the mean
        for i in range(min(10, exc_samples.shape[1])):
            ax.plot(
                times,
                exc_samples[:, i],
                color="black",
                alpha=0.6,
                linewidth=0.8,
                linestyle="-",
            )
    ax.set_ylabel("Exc weight")

    # Inhibitory
    ax = axes[1]
    ax.plot(
        times, inh_mean, color="black", linestyle="--", linewidth=2.0, label="Inh mean"
    )
    if inh_samples.ndim == 2 and inh_samples.shape[0] == times.shape[0]:
        # Use mono-color (black) and vary only linestyle; avoid dashed which is used by the mean
        for i in range(min(10, inh_samples.shape[1])):
            ax.plot(
                times,
                inh_samples[:, i],
                color="black",
                alpha=0.6,
                linewidth=0.8,
                linestyle="--",
            )
    ax.set_ylabel("Inh weight")
    ax.set_xlabel("Sleep time (ms)")

    os.makedirs("figures", exist_ok=True)
    pdf_path = os.path.join("plots", f"weight_sleep_epoch_{epoch:03d}.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight", dpi=900)
    plt.close(fig)


def plot_weight_evolution(
    weight_evolution: dict, output_path: str = "plots/weights_evolution.pdf"
):
    """
    Render the per-epoch weight statistics (mean/std and min/max) for excitatory
    and inhibitory synapses.
    """
    epochs = np.array(weight_evolution.get("epochs", []), dtype=float)
    if epochs.size == 0:
        raise ValueError("weight_evolution contains no epochs to plot")

    exc_mean = np.array(weight_evolution.get("exc_mean", []), dtype=float)
    exc_min = np.array(weight_evolution.get("exc_min", []), dtype=float)
    exc_max = np.array(weight_evolution.get("exc_max", []), dtype=float)

    inh_mean = np.array(weight_evolution.get("inh_mean", []), dtype=float)
    inh_min = np.array(weight_evolution.get("inh_min", []), dtype=float)
    inh_max = np.array(weight_evolution.get("inh_max", []), dtype=float)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Excitatory mean ± std
    ax.plot(
        epochs, exc_mean, label="Exc Mean", linewidth=2, linestyle="-", color="black"
    )
    ax.fill_between(
        epochs,
        exc_min,
        exc_max,
        facecolor="none",
        hatch="//",
        edgecolor="black",
        label="Exc min/max",
    )
    # Inhibitory mean ± std
    ax.plot(
        epochs, inh_mean, label="Inh Mean", linewidth=2, linestyle="--", color="black"
    )
    ax.fill_between(
        epochs,
        inh_min,
        inh_max,
        facecolor="none",
        hatch="\\\\",
        edgecolor="black",
        label="Inh min/max",
    )

    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Weight", fontsize=26)
    ax.set_xticks(fontsize=16)
    ax.set_yticks(fontsize=16)
    legend = ax.legend(
        facecolor="white",
        edgecolor="black",
        fontsize=22,
        loc="lower left",
        framealpha=1.0,
    )
    legend.get_frame().set_alpha(1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=900)
    plt.close(fig)


def plot_weight_evolution_during_sleep(weight_tracking_sleep):
    """Plot weight changes during sleep periods only (accumulated across all epochs)."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    if len(weight_tracking_sleep["exc_mean"]) == 0:
        print("Warning: No sleep weight tracking data available")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot excitatory weights during sleep
    ax = axes[0]
    sleep_times = np.array(weight_tracking_sleep["times"])  # concatenated
    sleep_exc = np.array(weight_tracking_sleep["exc_mean"])  # concatenated

    # Smooth the data for better visualization
    try:
        sleep_exc_sm = uniform_filter1d(sleep_exc, size=5)
    except Exception:
        sleep_exc_sm = sleep_exc
    ax.plot(sleep_times, sleep_exc_sm, color="#ff7f68", label="Exc mean (smoothed)")
    if "exc_std" in weight_tracking_sleep:
        exc_std = np.array(weight_tracking_sleep["exc_std"])
        try:
            exc_std_sm = uniform_filter1d(exc_std, size=5)
        except Exception:
            exc_std_sm = exc_std
        ax.fill_between(
            sleep_times,
            sleep_exc_sm - exc_std_sm,
            sleep_exc_sm + exc_std_sm,
            color="#ffe5e1",
            alpha=0.6,
        )
    ax.set_ylabel("Exc weight")

    # Plot inhibitory weights during sleep
    ax = axes[1]
    sleep_inh = np.array(weight_tracking_sleep["inh_mean"])  # concatenated
    try:
        sleep_inh_sm = uniform_filter1d(sleep_inh, size=5)
    except Exception:
        sleep_inh_sm = sleep_inh
    ax.plot(sleep_times, sleep_inh_sm, color="#05af9b", label="Inh mean (smoothed)")
    if "inh_std" in weight_tracking_sleep:
        inh_std = np.array(weight_tracking_sleep["inh_std"])  # concatenated
        try:
            inh_std_sm = uniform_filter1d(inh_std, size=5)
        except Exception:
            inh_std_sm = inh_std
        ax.fill_between(
            sleep_times,
            sleep_inh_sm - inh_std_sm,
            sleep_inh_sm + inh_std_sm,
            color="#c7fdf7",
            alpha=0.6,
        )
    ax.set_ylabel("Inh weight")
    ax.set_xlabel("Sleep time (ms)")
    ax.set_xticks(fontsize=16)
    ax.set_yticks(fontsize=16)

    os.makedirs("figures", exist_ok=True)
    save_path = os.path.join("figures", "weight_sleep_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_weight_trajectories_with_sleep_epoch(
    weight_tracking_epoch, epoch, max_lines=8
):
    """
    Plot sampled weight trajectories across snapshot time with sleeps shaded (combined on one axis, BW).
    - Exc mean: solid black; Exc samples: dotted black
    - Inh mean: dashed black; Inh samples: dash-dot black
    Sleep segments are provided as (start,end) in the same snapshot-time reference.
    """
    import matplotlib.pyplot as plt
    import shutil
    from matplotlib.lines import Line2D

    times = np.array(weight_tracking_epoch.get("times", []), dtype=float)
    if times.size == 0:
        return

    exc_samples = weight_tracking_epoch.get("exc_samples", [])
    inh_samples = weight_tracking_epoch.get("inh_samples", [])
    exc_mean = np.array(weight_tracking_epoch.get("exc_mean", []), dtype=float)
    inh_mean = np.array(weight_tracking_epoch.get("inh_mean", []), dtype=float)
    sleep_segments = weight_tracking_epoch.get("sleep_segments", [])

    # Convert list-of-lists to arrays (n_snapshots x K)
    def _to_array(list_of_lists):
        if not list_of_lists:
            return np.zeros((0, 0), dtype=float)
        max_len = max(len(row) for row in list_of_lists)
        if max_len == 0:
            return np.zeros((len(list_of_lists), 0), dtype=float)
        arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
        for r, row in enumerate(list_of_lists):
            k = min(max_len, len(row))
            if k > 0:
                arr[r, :k] = np.array(row[:k], dtype=float)
        return arr

    exc_arr = _to_array(exc_samples)
    inh_arr = _to_array(inh_samples)

    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Shade sleep segments
    for s, e in sleep_segments:
        try:
            patch = ax.axvspan(
                float(s),
                float(e),
                facecolor="0.92",
                edgecolor="0.92",
                alpha=1.0,
                zorder=0,
            )
            try:
                patch.set_hatch("//")
            except Exception:
                pass
        except Exception:
            pass

    # Excitatory trajectories (samples: dotted; mean: solid)
    if exc_arr.size > 0 and exc_arr.shape[0] == times.size:
        lines_to_plot = min(max_lines, exc_arr.shape[1])
        for i in range(lines_to_plot):
            ax.plot(
                times,
                exc_arr[:, i],
                color="black",
                alpha=0.6,
                linewidth=0.9,
                linestyle="-",
            )
    if exc_mean.size == times.size:
        ax.plot(
            times,
            exc_mean,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="Exc mean",
        )

    # Inhibitory trajectories (samples: dash-dot; mean: dashed)
    if inh_arr.size > 0 and inh_arr.shape[0] == times.size:
        lines_to_plot = min(max_lines, inh_arr.shape[1])
        for i in range(lines_to_plot):
            ax.plot(
                times,
                inh_arr[:, i],
                color="black",
                alpha=0.6,
                linewidth=0.9,
                linestyle="--",
            )
    if inh_mean.size == times.size:
        ax.plot(
            times,
            inh_mean,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Inh mean",
        )

    def _apply_labels(latex_enabled: bool):
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
