# This is the plotting functions script

# Import libraries
import numpy as np
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
    W_se = W_se.reshape(W_se.shape[0], -1)[:, ::1000]
    W_ee = W_ee.reshape(W_ee.shape[0], -1)[:, ::1000]
    W_ie = W_ie.reshape(W_ie.shape[0], -1)[:, ::1000]

    # Create x-variable for weight matrix
    x = np.arange(0, W_se.shape[0])

    print("Shape of x:", x.shape)
    print(
        "Example shape of weights:", W_se.T.shape
    )  # Transposed to match plotting dimensions

    # Define color and label mapping for plots
    weight_plots = {
        "W_se": {"data": W_se, "color": "red"},
        "W_ee": {"data": W_ee, "color": "blue"},
        "W_ie": {"data": W_ie, "color": "green"},
    }

    # Plot weights with different colors for each weight matrix
    for key, info in weight_plots.items():
        for i, weights in enumerate(info["data"]):
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


# Example usage would require you to define spikes, W_se, W_ee, W_ie, and dt variables.
