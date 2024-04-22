# This is the plotting functions script

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


def plot_membrane_activity(
    MemPot: np.ndarray, idx_start: int, idx_stop: int, update_interval: int
):
    # MemPot shape: (time, num_neurons-N_input_neurons)
    time_units = np.arange(idx_start, idx_stop, update_interval)
    for neuron in range(idx_start, idx_stop):
        mv_ls = []
        for t in time_units:
            mv_ls.append(MemPot[t, neuron])
        print(mv_ls)
        print(len(mv_ls))
        plt.plot((time_units, mv_ls), label=f"Neuron {neuron}")

    plt.xlabel("Time")
    plt.ylabel("Membrane Potential")
    plt.title("Membrane Potential Changes Over Time")
    plt.legend(title="Legend")
    plt.show()


def plot_weights_and_spikes(spikes, W_se, W_ee, dt, update_interval=10):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Get firing times for each neuron
    Firing_times = [
        np.where(spikes[t, 484:])[0] for t in range(0, spikes.shape[0], update_interval)
    ]

    # Plot spike raster
    axs[0].eventplot(Firing_times, colors="black")
    axs[0].set_title(f"Spikes during training")
    axs[0].set_xlabel("Neuron Index")
    axs[0].set_ylabel("Spikes")

    # Concatenate W_se and W_ee along the neuron axis
    weights = np.concatenate((W_se, W_ee), axis=2)
    weights = np.reshape(weights, (weights.shape[0], -1))

    # Convert weight matrix to a tenth of its current width
    nu_ws = np.zeros((weights.shape[0], weights.shape[1] // 10))
    step_size = weights.shape[1] // 10

    for id in range(nu_ws.shape[1]):
        start_index = id * step_size
        end_index = start_index + step_size
        if weights[:, start_index:end_index].shape[1] != 0:
            nu_ws[:, id] = np.mean(weights[:, start_index:end_index], axis=1)

    # Create list of time_units for each weight to reduce computation
    time_units = np.arange(0, nu_ws.shape[0], update_interval)

    # Time steps array, adjust based on your dt
    time_steps = time_units * dt

    # Find indices of nonzero weights in the first row (assuming sparsity)
    non_zero_indices = np.nonzero(nu_ws[0])[0]

    for j in non_zero_indices:
        axs[1].plot(time_steps, weights[time_units, j], label=f"Weight {j}")

    # Adding legend, labels, and title
    axs[1].legend(title="Legend", loc="upper right")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Weight Value")
    axs[1].set_title("Weight Changes Over Time")

    plt.show()
