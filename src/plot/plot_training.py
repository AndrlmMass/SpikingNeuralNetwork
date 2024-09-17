# This is the plotting functions script

# Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE


def plot_membrane_activity(
    MemPot: np.ndarray,
    MemPot_th: int,
    t_start: int,
    t_stop: int,
    N_excit_neurons: int,
):
    t = np.arange(t_start, t_stop)
    print(t.shape[0])
    MemPot_th_ = np.full(shape=t.shape[0], fill_value=MemPot_th)

    # Get membrane potentials for excitatory and inhibitory neurons
    MemPot_exc = MemPot[t_start:t_stop, :N_excit_neurons]
    MemPot_inh = MemPot[t_start:t_stop, N_excit_neurons:]

    # Compute mean, min, and max membrane potentials over neurons for each time step
    exc_mean = np.mean(MemPot_exc, axis=1)
    exc_min = np.min(MemPot_exc, axis=1)
    exc_max = np.max(MemPot_exc, axis=1)

    inh_mean = np.mean(MemPot_inh, axis=1)
    inh_min = np.min(MemPot_inh, axis=1)
    inh_max = np.max(MemPot_inh, axis=1)

    # Plot mean membrane potentials
    plt.plot(t, exc_mean, color="red", label="Excitatory Mean")
    plt.plot(t, inh_mean, color="blue", label="Inhibitory Mean")
    plt.plot(t, MemPot_th_, color="grey", linestyle="dashed", label="Threshold")

    # Fill between min and max to create shaded area
    plt.fill_between(
        t, exc_min, exc_max, color="red", alpha=0.2, label="Excitatory Range"
    )
    plt.fill_between(
        t, inh_min, inh_max, color="blue", alpha=0.2, label="Inhibitory Range"
    )

    plt.xlabel("ms")
    plt.ylabel("Membrane Potential")
    plt.title("Membrane Potential Changes Over Time")

    plt.show()


def plot_weights_and_spikes(spikes, weights, t_start, t_stop):

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Get firing times for each neuron
    Firing_times = [
        np.where(spikes[t_start:t_stop, n])[0] for n in range(spikes.shape[1])
    ]

    # Add item and neuronal layer indicators
    axs[0].axhline(y=484, color="red", linestyle="-")
    axs[0].axhline(y=968, color="red", linestyle="-")

    # Plot spike raster
    axs[0].eventplot(Firing_times, colors="black")
    axs[0].set_title("Spikes during training")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Neuron index")

    # Define se, ee and ie indices for weights
    w_se_start, w_se_stop = 0, 9
    w_ee_start, w_ee_stop = 10, 19
    w_ie_start, w_ie_stop = 20, 29

    # Define color and label mapping for plots
    weight_plots = {
        "W_se": {"data": weights[t_start:t_stop, w_se_start:w_se_stop], "color": "red"},
        "W_ee": {
            "data": weights[t_start:t_stop, w_ee_start:w_ee_stop],
            "color": "blue",
        },
        "W_ie": {
            "data": weights[t_start:t_stop, w_ie_start:w_ie_stop],
            "color": "green",
        },
    }

    # Plot weights with different colors for each weight matrix
    for key, info in weight_plots.items():
        for i, weights in enumerate(info["data"].T):
            if i == 0:
                axs[1].plot(
                    weights, color=info["color"], label=key
                )  # Label only the first line of each type
            else:
                axs[1].plot(weights, color=info["color"])
    axs[1].set_title("Weight Matrix Changes")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Weight Value")
    axs[1].legend()

    plt.show()


def plot_clusters(spikes, labels, N_input_neurons, N_excit_neurons, N_inhib_neurons):
    # Create list for class-preference
    class_preference = np.zeros(spikes.shape[1])
    intensity = np.zeros(spikes.shape[1])

    # Loop through each neuron and assign class-preference
    for n in range(spikes.shape[1]):
        total_class_spikes = [0, 0, 0, 0]

        # Count spikes for each class
        for i in range(4):
            total_class_spikes[i] = np.sum(spikes[np.where(labels[:, i] == 1)[0], n])

        # Append class-preference to list
        idx = np.argmax(total_class_spikes)
        class_preference[n] = idx
        total_spikes = np.sum(total_class_spikes)
        if total_spikes > 0:
            intensity[n] = total_class_spikes[idx] / total_spikes
        else:
            intensity[n] = 0

    # Define base colors for each class
    base_colors = np.array(
        [
            [1, 0, 0],  # Red for class 0
            [0, 1, 0],  # Green for class 1
            [0, 0, 1],  # Blue for class 2
            [1, 1, 0],  # Yellow for class 3
        ]
    )

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 8))

    # Reshape class-preference and intensity for plotting
    W_ee_pref = class_preference[
        N_input_neurons : N_input_neurons + N_excit_neurons
    ].reshape(int(math.sqrt(N_excit_neurons)), int(math.sqrt(N_excit_neurons)))
    W_ee_intensity = intensity[
        N_input_neurons : N_input_neurons + N_excit_neurons
    ].reshape(int(math.sqrt(N_excit_neurons)), int(math.sqrt(N_excit_neurons)))

    W_ie_pref = class_preference[N_input_neurons + N_excit_neurons :].reshape(
        int(math.sqrt(N_inhib_neurons)), int(math.sqrt(N_inhib_neurons))
    )
    W_ie_intensity = intensity[N_input_neurons + N_excit_neurons :].reshape(
        int(math.sqrt(N_inhib_neurons)), int(math.sqrt(N_inhib_neurons))
    )

    W_se_pref = class_preference[:N_input_neurons].reshape(
        int(math.sqrt(N_input_neurons)), int(math.sqrt(N_input_neurons))
    )
    W_se_intensity = intensity[:N_input_neurons].reshape(
        int(math.sqrt(N_input_neurons)), int(math.sqrt(N_input_neurons))
    )

    # Function to create the combined color array
    def create_color_array(class_pref, intensity):
        color_array = np.zeros((*class_pref.shape, 3))
        for i in range(4):  # Assuming classes are 0 to 3
            class_mask = class_pref == i
            for j in range(3):  # For RGB channels
                color_array[..., j] += class_mask * base_colors[i, j] * intensity
        return color_array

    # Create color arrays
    W_ee_colors = create_color_array(W_ee_pref, W_ee_intensity)
    W_ie_colors = create_color_array(W_ie_pref, W_ie_intensity)
    W_se_colors = create_color_array(W_se_pref, W_se_intensity)

    # Plot the heatmaps with intensity scaling
    ax[0].imshow(W_ee_colors, interpolation="nearest")
    ax[0].set_title("Class preference in excitatory layer")

    ax[1].imshow(W_ie_colors, interpolation="nearest")
    ax[1].set_title("Class preference in inhibitory layer")

    ax[2].imshow(W_se_colors, interpolation="nearest")
    ax[2].set_title("Class preference in stimulation layer")

    # Create a custom legend for the classes
    class_labels = ["Triangle class", "Circle class", "Square class", "X class"]
    class_colors = ["red", "green", "blue", "yellow"]
    legend_patches = [
        Patch(color=class_colors[i], label=class_labels[i]) for i in range(4)
    ]

    fig.legend(handles=legend_patches, loc="upper right", title="Class Legends")

    plt.show()


def plot_traces(
    pre_synaptic_trace, post_synaptic_trace, slow_pre_synaptic_trace, N_input_neurons
):
    trace_plots = {
        "pre": {"data": pre_synaptic_trace, "color": "red"},
        "post": {"data": post_synaptic_trace, "color": "blue"},
        "slow": {"data": slow_pre_synaptic_trace, "color": "green"},
    }

    # Plot weights with different colors for each weight matrix
    for key, info in trace_plots.items():
        for i, weights in enumerate(info["data"].T):
            if i == 0:
                plt.plot(
                    weights, color=info["color"], label=key
                )  # Label only the first line of each type
            else:
                plt.plot(weights, color=info["color"])
    plt.legend(loc="upper right")
    plt.show()


def t_SNE(N_classes, spikes, labels_spike, labels, timesteps, N_input_neurons):
    # Reshape labels to match spikes
    labels_spike_simpl = np.argmax(labels_spike, axis=1)

    # Remove ISI
    filler_mask = labels_spike_simpl != N_classes
    spikes = spikes[filler_mask]
    spikes = spikes[:, N_input_neurons:]
    labels_spike = labels_spike[filler_mask]
    labels = labels[labels != N_classes]

    # Temporal binning to create features
    n_time_steps, n_neurons = spikes.shape
    n_bins = n_time_steps // timesteps
    bin_size = timesteps
    features = np.zeros((n_bins, n_neurons))

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        features[i, :] = np.mean(spikes[start:end, :], axis=0)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Label names
    label_names = ["Triangle", "Circle", "Square", "X"]

    # Visualize the results with labels
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(
            tsne_results[indices, 0], tsne_results[indices, 1], label=label_names[label]
        )
    plt.title("t-SNE results of SNN firing rates")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()


def plot_cluster_activity(spikes, labels, freq_threshold):
    # Compute average spiking activity per neuron per class for training period
    cluster_spikes = {
        "triangle_class": [],
        "square_class": [],
        "circle_class": [],
        "X_class": [],
    }

    for n, (key, value) in enumerate(cluster_spikes):  # Does it start from 1 or 0?
        firing_freq = np.sum(spikes[1000][labels == n], axis=1) / np.count_nonzero(
            int(labels == n)
        )  # Need to set '1000' to something more meaningful
        indices = np.where(firing_freq >= freq_threshold)

        # Store cluster spikes
        cluster_spikes[key] = np.sum(spikes[:, indices], axis=0)

    #
