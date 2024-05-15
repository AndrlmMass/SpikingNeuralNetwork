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


def plot_weights_and_spikes(spikes, W_se, W_ee, W_ie, dt, update_interval=10):
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

    # Loop through each neuron and assign class-preference
    for n in range(spikes.shape[1]):
        total_class_spikes = [0, 0, 0, 0]

        # Count spikes for each class
        for i in range(4):
            total_class_spikes[i] = np.sum(spikes[np.where(labels[:, i] == 1)[0], n])

        # Append class-preference to list
        class_preference[n] = np.argmax(total_class_spikes)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Reshape class-preference for plotting
    W_ee_pref = class_preference[
        N_input_neurons : N_input_neurons + N_excit_neurons
    ].reshape(int(math.sqrt(N_excit_neurons)), int(math.sqrt(N_excit_neurons)))

    W_ie_pref = class_preference[N_input_neurons + N_excit_neurons :].reshape(
        int(math.sqrt(N_inhib_neurons)), int(math.sqrt(N_inhib_neurons))
    )

    # Create a heatmap for class-preference of the weights
    ax[0].imshow(W_ee_pref, cmap="viridis", interpolation="nearest")
    ax[0].set_title("Class preference in excitatory layer")

    ax[1].imshow(W_ie_pref, cmap="viridis", interpolation="nearest")
    ax[1].set_title("Class preference in inhibitory layer")

    plt.show()
