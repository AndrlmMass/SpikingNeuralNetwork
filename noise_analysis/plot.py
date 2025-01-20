import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def spike_plot(data, labels):
    # Validate dimensions
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Labels length ({len(labels)}) must match the number of time steps ({data.shape[0]})."
        )

    # Assign colors to unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: color for label, color in zip(unique_labels, colors)}

    # Collect spike positions for each neuron
    positions = [np.where(data[:, n] == 1)[0] for n in range(data.shape[1])]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the spikes
    ax.eventplot(positions, lineoffsets=np.arange(data.shape[1]), colors="black")
    ax.set_ylabel(f"{data.shape[1]} Units")
    ax.set_xlabel("Time (ms)")

    # Add the horizontal line below the spikes
    y_offset = -10  # Position below the spike raster
    for label in unique_labels:
        # Get the indices where this label is active
        label_indices = np.where(labels == label)[0]

        # Plot a horizontal line for the label
        if len(label_indices) > 0:
            ax.hlines(
                y=y_offset,  # Position below the raster
                xmin=label_indices[0],  # Start of the label
                xmax=label_indices[-1] + 1,  # End of the label
                color=label_colors[label],  # Color for the label
                linewidth=6,
                label=f"Class {label}",
            )

    # Add legend for the labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(unique_labels))

    plt.title("Spikes with Class-based Horizontal Lines")
    plt.tight_layout()
    plt.show()


def heat_map(data, pixel_size):
    data = data.numpy()
    summed_data = np.sum(data, axis=0)
    reshaped_summed_data = np.reshape(summed_data, (pixel_size, pixel_size))
    plt.imshow(reshaped_summed_data, cmap="hot", interpolation="nearest")
    plt.show()


def mp_plot(mp, N_exc):
    plt.plot(mp[:, :N_exc], color="green")
    plt.title("membrane potential during training")
    plt.xlabel("time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.show()

    plt.plot(mp[:, N_exc:], color="red")
    plt.title("membrane potential during training")
    plt.xlabel("time (ms)")
    plt.ylabel("Membrane potential (mV)")
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
