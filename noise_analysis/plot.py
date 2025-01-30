import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
                # Plot from segment_start to i for current_label
                ax.hlines(
                    y=y_offset,
                    xmin=segment_start,
                    xmax=i,  # up to but not including i
                    color=label_colors.get(current_label, "black"),
                    linewidth=6,
                    label=(
                        None
                        if current_label in drawn_labels
                        else f"Class {current_label}"
                    ),
                )
                drawn_labels.add(current_label)

            # Update to the new segment
            current_label = labels[i]
            segment_start = i

    # Handle the last segment after exiting the loop
    if current_label != -1:
        ax.hlines(
            y=y_offset,
            xmin=segment_start,
            xmax=len(labels),
            color=label_colors.get(current_label, "black"),
            linewidth=6,
            label=None if current_label in drawn_labels else f"Class {current_label}",
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


def weights_plot(weights_plot, N_x, N_inh):
    # Simplify weights
    mu_weights = np.mean(weights_plot, axis=2)

    # Plot the data
    plt.plot(mu_weights[:, N_x:-N_inh], color="blue", label="excitatory")
    plt.plot(mu_weights[:, -N_inh:], color="red", label="inhibitory")

    # Ensure only unique legends are displayed
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates by creating a dictionary
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def t_SNE(
    spikes,
    labels_spike,
    timesteps,
    items,
    N_x,
    n_components,
    perplexity,
    max_iter,
    random_state,
):

    # Remove ISI
    filler_mask = labels_spike != -1
    spikes = spikes[filler_mask]
    spikes = spikes[:, N_x:]
    labels_spike = labels_spike[filler_mask]
    diff_indices = np.where(np.diff(labels_spike) != 0)[0] + 1
    labels = np.concatenate(([labels_spike[0]], labels_spike[diff_indices]))

    # Temporal binning to create features
    n_time_steps, n_neurons = spikes.shape
    n_bins = n_time_steps // timesteps
    bin_size = timesteps
    features = np.zeros((n_bins, n_neurons))

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        features[i, :] = np.mean(spikes[start:end, :], axis=0)

    if perplexity < items:
        perplexity = items - 1

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    tsne_results = tsne.fit_transform(features)

    # Visualize the results with labels
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        if label == -1:
            continue
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label)
    plt.title("t-SNE results of SNN firing rates")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()
