# This is the plotting functions script

# Import libraries
import numpy as np
import imageio
from sklearn.manifold import (
    TSNE,
)  # This will be applied later to illustrate clustering in the SNN
import matplotlib.pyplot as plt


def plot_membrane_activity(MemPot, num_neurons, num_items, input_idx, timesteps):
    # Reshape array to plot items continuously
    MemPot = MemPot.reshape(MemPot.shape[0] * MemPot.shape[2], MemPot.shape[1])
    fig, ax = plt.subplots()  # Corrected here
    neurons = np.arange(num_neurons)
    for j in input_idx:
        if j < num_neurons:
            neurons = np.delete(neurons, j)

    for j in range(len(neurons)):
        y = MemPot[:, neurons[j]]
        x = np.arange(len(y))
        ax.plot(x, y, label=f"Neuron {neurons[j]}")

    for item in range(1, num_items):
        ax.axvline(x=item * timesteps, color="r", linestyle=":", linewidth=1)

    plt.xlabel("Time")
    plt.ylabel("mV Value")
    plt.title("Membrane Potential Changes Over Time")
    plt.legend()
    plt.show()


def plot_weights_and_spikes(spikes, W_se, W_ee, dt, update_interval=10):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 16)
    )  # Large figure to accommodate both plots

    # Get firing times for each neuron
    Firing_times = [np.where(spikes[t, 484:])[0] for t in range(spikes.shape[0])]

    # Plot spike raster
    axs[0].eventplot(Firing_times, colors="black")
    axs[0].set_title(f"Spikes during training")
    axs[0].set_xlabel("Neuron Index")
    axs[0].set_ylabel("Spikes")

    # Concatenate W_se and W_ee along the neuron axis
    weights = np.concatenate(
        (W_se, W_ee), axis=2
    )  # Using axis=2 to concatenate along the feature dimension
    weights = np.reshape(
        weights, (weights.shape[0], -1)
    )  # Flatten the last two dimensions

    # Time steps array, adjust based on your dt
    dt = 1  # Set your time step duration
    time_steps = np.arange(0, weights.shape[0] * dt, dt)

    # Find indices of nonzero weights in the first row (assuming sparsity)
    non_zero_indices = np.nonzero(weights[0])[0]

    for j in non_zero_indices:
        axs[1].plot(time_steps, weights[:, j], label=f"Weight {j}")

    # Adding legend, labels, and title
    axs[1].legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Weight Value")
    axs[1].set_title("Weight Changes Over Time")

    plt.tight_layout()
    plt.show()
