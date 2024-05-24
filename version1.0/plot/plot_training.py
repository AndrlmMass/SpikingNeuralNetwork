# This is the plotting functions script

# Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt


def plot_membrane_activity(
    MemPot: np.ndarray,
    idx_start: int,
    idx_stop: int,
):
    plt.plot(MemPot[:, idx_start:idx_stop])

    plt.xlabel("ms")
    plt.ylabel("Membrane Potential")
    plt.title("Membrane Potential Changes Over Time")
    plt.show()


def plot_weights_and_spikes(spikes, W_se, W_ee, W_ie):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Get firing times for each neuron
    Firing_times = [np.where(spikes[:, n])[0] for n in range(spikes.shape[1])]

    # Add item and neuronal layer indicators
    for i in range(0, spikes.shape[0], 100):  # Adjust for dt scaling
        axs[0].axvline(x=i, color="black", linestyle="--")
    axs[0].axhline(y=484, color="red", linestyle="-")
    axs[0].axhline(y=968, color="red", linestyle="-")

    # Plot spike raster
    axs[0].eventplot(Firing_times, colors="black")
    axs[0].set_title("Spikes during training")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Neuron index")

    # Reshape weight matrices
    W_se = W_se.reshape(W_se.shape[0], -1)[:, ::500]
    W_ee = W_ee.reshape(W_ee.shape[0], -1)[:, ::500]
    W_ie = W_ie.reshape(W_ie.shape[0], -1)[:, ::500]

    print(f"W_se: {W_se.shape}, W_ee: {W_ee.shape}, W_ie: {W_ie.shape}")

    # Define color and label mapping for plots
    weight_plots = {
        "W_se": {"data": W_se, "color": "red"},
        "W_ee": {"data": W_ee, "color": "blue"},
        "W_ie": {"data": W_ie, "color": "green"},
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

    cax3 = ax[2].imshow(W_se_colors, interpolation="nearest")
    ax[2].set_title("Class preference in stimulation layer")

    # Create a colorbar for the last subplot
    cbar3 = fig.colorbar(cax3, ax=ax[2], orientation="vertical")
    cbar3.set_ticks([0.25, 0.75, 1.25, 1.75])
    cbar3.ax.set_yticklabels(["Class 0", "Class 1", "Class 2", "Class 3"])
    cbar3.set_label("Class preference")

    plt.show()


def plot_traces(
    pre_synaptic_trace, post_synaptic_trace, slow_pre_synaptic_trace, N_input_neurons
):
    # Draw traces
    plt.plot(
        pre_synaptic_trace[:, 40 : 60 + 20],
        label="pre_synaptic_trace",
        color="red",
    )
    plt.plot(post_synaptic_trace[:, :20], label="post_synaptic_trace", color="blue")
    plt.plot(
        slow_pre_synaptic_trace[:, N_input_neurons : N_input_neurons + 20],
        label="slow_pre_synaptic_trace",
        color="green",
    )
    plt.legend()
    plt.show()
