import os
import numpy as np
from ..utils.platform import configure_matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import datetime
configure_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
# this is active
def preview_loaded_data(
    save_path: str | None = None, images=None, labels=None
):
    """
    Plot a 2x2 grid of images from the loaded dataset (one per class).
    
    Uses cached train data from data/datasets/geomfig/image_cache/train.
    
    Args:
        images: Cached images array (required)
        labels: Cached labels array (required)
        save_path: Optional path to save the plot
    """
    if images is None or labels is None:
        raise ValueError("images and labels must be provided (use cached train data)")
    
    classes = [0, 1, 2, 3]
    
    # Convert to numpy if torch tensors
    if hasattr(images, 'numpy'):
        images_np = images.numpy()
    elif hasattr(images, 'cpu'):
        images_np = images.cpu().numpy()
    else:
        images_np = np.array(images)
    
    labels_np = np.array(labels)
    
    # Remove channel dimension if present: (N, 1, H, W) -> (N, H, W)
    if images_np.ndim == 4 and images_np.shape[1] == 1:
        images_np = images_np.squeeze(1)
    
    # Select one sample per class
    selected_images = []
    selected_labels = []
    for cls in classes:
        cls_indices = np.where(labels_np == cls)[0]
        if len(cls_indices) == 0:
            raise ValueError(f"No samples found for class {cls}")
        # Take first sample of this class
        idx = cls_indices[0]
        selected_images.append(images_np[idx])
        selected_labels.append(cls)
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(4.0, 4.0))
    axes = axes.flatten()
    titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
    
    for idx, (img, cls) in enumerate(zip(selected_images, selected_labels)):
        ax = axes[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(titles[cls], fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    try:
        if save_path is None:
            os.makedirs("plots", exist_ok=True)
            save_path = os.path.join("plots", "geomfig_preview.png")
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Dataset preview saved to {save_path}")
    except Exception as exc:
        print(f"Failed to save dataset preview ({exc})")
    plt.show()
    plt.close(fig)

def tsne(tsne_results, segment_labels, train, show_plot, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
    unique_labels = np.unique(segment_labels)

    for i, label in enumerate(unique_labels):
        indices = segment_labels == label
        marker = marker_list[i % len(marker_list)]
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"Class {label}",
            marker=marker,
            color="black",
            s=60,
        )

    plt.xlabel("t-SNE dimension 1", fontsize=26)
    plt.ylabel("t-SNE dimension 2", fontsize=26)
    os.makedirs("plots", exist_ok=True)
    suffix = "train" if train else "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_path = os.path.join("plots\\tsne", f"tsne_{suffix}_{timestamp}.pdf")
    plt.tight_layout()
    plt.savefig(tsne_path, bbox_inches="tight")
    print(f"t-SNE plot saved to {tsne_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

def weight_trajectories(
    weight_tracking_epoch, epoch, dataset, max_lines=32, sleep_enabled=True, 
):
    """
    Plot sampled weight trajectories across snapshot time with sleeps shaded (combined on one axis, BW).
    - Exc mean: solid black; Exc samples: dotted black
    - Inh mean: dashed black; Inh samples: dash-dot black
    Sleep segments are provided as (start,end) in the same snapshot-time reference.
    """
    from matplotlib.lines import Line2D

    times = np.array(weight_tracking_epoch.get("times", []), dtype=float)
    if times.size == 0:
        return

    exc_samples = weight_tracking_epoch.get("exc_samples", [])
    inh_samples = weight_tracking_epoch.get("inh_samples", [])
    
    # Ensure exc_mean and inh_mean are flat arrays (not nested lists)
    exc_mean_raw = weight_tracking_epoch.get("exc_mean", [])
    inh_mean_raw = weight_tracking_epoch.get("inh_mean", [])
    
    # Convert to numpy arrays, handling both flat lists and nested structures
    if exc_mean_raw and isinstance(exc_mean_raw[0], (list, np.ndarray)):
        # If it's a list of lists, flatten it (shouldn't happen, but handle it)
        exc_mean = np.array([item for sublist in exc_mean_raw for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])], dtype=float)
    else:
        exc_mean = np.array(exc_mean_raw, dtype=float)
    
    if inh_mean_raw and isinstance(inh_mean_raw[0], (list, np.ndarray)):
        # If it's a list of lists, flatten it (shouldn't happen, but handle it)
        inh_mean = np.array([item for sublist in inh_mean_raw for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])], dtype=float)
    else:
        inh_mean = np.array(inh_mean_raw, dtype=float)
    
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
    
    # Plot excitatory mean (only once, handle length mismatches)
    if exc_mean.size > 0:
        min_len = min(exc_mean.size, times.size)
        if min_len > 0:
            ax.plot(
                times[:min_len],
                exc_mean[:min_len],
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
    
    # Plot inhibitory mean (handle length mismatches)
    if inh_mean.size > 0:
        min_len = min(inh_mean.size, times.size)
        if min_len > 0:
            ax.plot(
                times[:min_len],
                inh_mean[:min_len],
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Inh mean",
            )

    def _apply_labels(latex_enabled: bool):
        plt.rcParams["text.usetex"] = bool(latex_enabled)
        ylabel = r"$\Delta w$" if latex_enabled else "Δw"
        title = "Sleep + STDP learning" if sleep_enabled else "No sleep"

        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel("time (ms)", fontsize=16)
        ax.set_title(title, fontsize=16)

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
        legend = ax.legend(
            handles=legend_lines,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fontsize=12,
        )
        legend.get_frame().set_alpha(1.0)

    latex_available = shutil.which("latex") is not None
    _apply_labels(latex_enabled=latex_available)
    pdf_dir = os.path.join("plots", "weights", "Trajectory", dataset)
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"weights_trajectories_epoch_{int(epoch):03d}.pdf")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.rcParams["text.usetex"] = False
    plt.close(fig)

def weight_evolution(
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
    ax.set_xlabel("Epoch", size=18)
    ax.set_ylabel("Average weight", size=18)
    legend = ax.legend(
        facecolor="white",
        edgecolor="black",
        fontsize=14,
        loc="upper right",
        framealpha=1.0,
    )

    legend.get_frame().set_alpha(1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(size=14)
    ax.set_yticklabels(size=14)
    fig.suptitle("No sleep", fontsize=20, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=900)
    plt.close(fig)


def plot_weight_distribution(N_x, st, ih, weights, N_exc, N_inh, e, weight_evolution, epochs):
    _st, _ex, _ih = N_x, st + 0 - 0, ih + 0 - 0
    _ex = _st + N_exc
    _ih = _ex + N_inh
    W_exc = weights[:_ex, _st:_ih]
    W_inh = weights[_ex:_ih, _st:_ex]

    weight_evolution["epochs"].append(e + 1)

    # Compute stats on non-zero weights only to avoid zero-bias
    W_exc_nz = W_exc[W_exc != 0]
    if W_exc_nz.size > 0:
        weight_evolution["exc_mean"].append(
            float(np.mean(W_exc_nz))
        )
        weight_evolution["exc_std"].append(float(np.std(W_exc_nz)))
        weight_evolution["exc_min"].append(float(np.min(W_exc_nz)))
        weight_evolution["exc_max"].append(float(np.max(W_exc_nz)))
    else:
        weight_evolution["exc_mean"].append(0.0)
        weight_evolution["exc_std"].append(0.0)
        weight_evolution["exc_min"].append(0.0)
        weight_evolution["exc_max"].append(0.0)

    W_inh_nz = W_inh[W_inh != 0]
    if W_inh_nz.size > 0:
        weight_evolution["inh_mean"].append(
            float(np.mean(W_inh_nz))
        )
        weight_evolution["inh_std"].append(float(np.std(W_inh_nz)))
        weight_evolution["inh_min"].append(float(np.min(W_inh_nz)))
        weight_evolution["inh_max"].append(float(np.max(W_inh_nz)))
    else:
        weight_evolution["inh_mean"].append(0.0)
        weight_evolution["inh_std"].append(0.0)
        weight_evolution["inh_min"].append(0.0)
        weight_evolution["inh_max"].append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Plot only non-zero weights for informative histograms
    exc_vals = W_exc.flatten()
    exc_vals = exc_vals[exc_vals != 0]
    if exc_vals.size > 0:
        axes[0].hist(exc_vals, bins=50, color="tomato", alpha=0.8)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No non-zero weights",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].set_title("Excitatory weights")
    axes[0].set_xlabel("Weight")
    axes[0].set_ylabel("Count")
    inh_vals = W_inh.flatten()
    inh_vals = inh_vals[inh_vals != 0]
    if inh_vals.size > 0:
        axes[1].hist(
            inh_vals, bins=50, color="steelblue", alpha=0.8
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No non-zero weights",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_title("Inhibitory weights")
    axes[1].set_xlabel("Weight")
    fig.suptitle(
        f"Epoch {e+1}/{epochs} - Weight Distributions",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"plots/weights/weights_epoch_{e+1:03d}.png", bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"  Saved weights snapshot: plots/weights_epoch_{e+1:03d}.png"
    )


def _plot_weight_matrix(weights):
    """Visualize the weight matrix."""
    boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]
    cmap = ListedColormap(["red", "white", "green"])
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    plt.imshow(weights, cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    plt.title("Weights")
    plt.show()

def plot_accuracy_history(save_dir: str = "plots", filename_suffix: str = "", acc_history: dict = None):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    if acc_history.get("train"):
        plt.plot(
            range(1, len(acc_history["train"]) + 1),
            acc_history["train"],
            label="Train",
            color="gray",   
            linestyle="-",
        )
    if acc_history.get("val"):
        plt.plot(
            range(1, len(acc_history["val"]) + 1),
            acc_history["val"],
            label="Val",
            color="black",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    tv_path = os.path.join(save_dir, f"acc_train_val{suffix_str}.png")
    try:
        plt.savefig(tv_path, bbox_inches="tight")
        print(f"Saved train/val accuracy plot to {tv_path}")
    except Exception as exc:
        print(f"Failed to save train/val plot ({exc})")
    plt.close()


def plot_network_graph(weights, N_x, N_exc, N_inh):
    """Visualize the network as a graph."""
    total_nodes = N_x + N_exc + N_inh

    G = nx.from_numpy_array(weights)

    # Partition nodes
    input_nodes = list(range(N_x))
    exc_nodes = list(range(N_x, N_x + N_exc))
    inh_nodes = list(range(N_x + N_exc, total_nodes))

    # Assign positions (vertical columns)
    pos = {}
    for i, node in enumerate(input_nodes):
        y = 1 - (i / (len(input_nodes) - 1)) if len(input_nodes) > 1 else 0.5
        pos[node] = (0, y)

    for i, node in enumerate(exc_nodes):
        y = 1 - (i / (len(exc_nodes) - 1)) if len(exc_nodes) > 1 else 0.5
        pos[node] = (1, y)

    for i, node in enumerate(inh_nodes):
        y = 1 - (i / (len(inh_nodes) - 1)) if len(inh_nodes) > 1 else 0.5
        pos[node] = (2, y)

    # Node colors
    node_colors = {}
    for node in input_nodes:
        node_colors[node] = "skyblue"
    for node in exc_nodes:
        node_colors[node] = "lightgreen"
    for node in inh_nodes:
        node_colors[node] = "salmon"

    colors = [node_colors[node] for node in G.nodes()]

    # Draw
    plt.figure(figsize=(8, 4))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=100)
    edges = G.edges(data=True)
    edge_weights = [data["weight"] for (u, v, data) in edges]
    nx.draw_networkx_edges(G, pos, width=[5 * w for w in edge_weights], alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=5, font_color="black")
    plt.title("Partitioned Graph: Input, Excitatory, Inhibitory")
    plt.axis("off")
    plt.show()


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
    n_last_points=None,
):

    # get indicess
    indices, spikes, labels = get_elite_nodes(
        spikes=spikes,
        labels=labels,
        num_classes=num_classes,
        narrow_top=narrow_top,
    )

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

    # Assign colors to unique labels (excluding -1 if desired)
    valid_label_mask = labels != -1
    unique_labels = np.unique(labels[valid_label_mask])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    # Collect spike positions for each neuron
    # Try different spike representations
    spike_threshold = 0 if np.any(data > 0) else 1

    positions = [
        np.where(data[:, n] > spike_threshold)[0] for n in range(data.shape[1])
    ]

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

# Ensure we export what snn.py uses
__all__ = [
    'plot_tsne',
    'plot_weight_distribution',
    'plot_accuracy_history',
    'plot_network_graph',
    'plot_epoch_training',
    'top_responders_plotted',
    'spike_plot',
    'plot_floats_and_spikes',
    'plot_accuracy',
    'get_contiguous_segment',
    'preview_loaded_data',
    'plot_weight_matrix',
    'plot_weight_evolution',
    'plot_weight_evolution_during_sleep',
    'plot_weight_evolution_during_sleep_epoch',
    'plot_weight_trajectories_with_sleep_epoch',
    'save_weight_distribution_gif',
]
