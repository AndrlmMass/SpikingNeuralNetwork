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


def plot_weights_and_spikes(spikes, t, W_se, W_ee, dt, update_interval=10):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 16)
    )  # Large figure to accommodate both plots

    # Plot spike raster
    axs[0].eventplot([np.where(spikes[t])[0]], colors="black")
    axs[0].set_title(f"Spike Raster at Time {t}")
    axs[0].set_xlabel("Neuron Index")
    axs[0].set_ylabel("Spikes")

    # Only update weights if t is a multiple of update_interval
    if t % update_interval == 0:
        # Concatenate W_se and W_ee along the neuron axis
        weights = np.concatenate((W_se, W_ee), axis=1)

        # Time steps array, adjusted to plot weights up to current time
        time_steps = np.arange(0, (t + 1) * dt, dt)

        # Plot weight changes over time
        for i in range(weights.shape[0]):  # Iterate over all starting neurons
            for j in range(weights.shape[1]):  # Iterate over all ending neurons
                if np.any(
                    weights[i, j, : t + 1] != 0
                ):  # Check if there are any non-zero weights
                    axs[1].plot(
                        time_steps[
                            : len(weights[i, j, : t + 1])
                        ],  # Correct time range for existing data
                        weights[i, j, : t + 1],
                        label=f"Weight from Neuron {i+1} to {j+1}",
                    )

        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Weight Value")
        axs[1].set_title("Weight Changes Over Time")
        axs[1].legend()  # Optional, can be omitted for clarity if too many lines

    plt.tight_layout()
    plt.show()
