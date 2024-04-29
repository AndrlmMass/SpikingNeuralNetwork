# This is the plotting functions script

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


def plot_membrane_activity(
    MemPot: np.ndarray,
    idx_start: int,
    idx_stop: int,
    update_interval: int,
    time_start: int,
    time_stop: int,
):
    # MemPot shape: (time, num_neurons-N_input_neurons)
    time_units = np.arange(time_start, time_stop - 1, update_interval)
    t_unit = np.arange(0, MemPot.shape[0])
    print(MemPot.shape)
    # for neuron in range(idx_start, idx_stop):
    plt.plot(t_unit, MemPot)

    plt.xlabel("ms")
    plt.ylabel("Membrane Potential")
    plt.title("Membrane Potential Changes Over Time")
    plt.show()


def plot_weights_and_spikes(spikes, W_se, W_ee, W_ie, dt, update_interval=10):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Get firing times for each neuron
    Firing_times = [np.where(spikes[:, n])[0] for n in range(0, 1089)]

    # Plot spike raster
    axs[0].eventplot(Firing_times, colors="black")
    axs[0].set_title(f"Spikes during training")
    axs[0].set_xlabel("time (ms)")
    axs[0].set_ylabel("Neuron index")

    # Reshape weight matrix
    W_se = W_se.reshape(W_se.shape[0], -1)
    W_ee = W_ee.reshape(W_ee.shape[0], -1)
    W_ie = W_ie.reshape(W_ie.shape[0], -1)

    # Reduce complexity of weight matrix
    W_se = W_se[:300, :10]
    W_ee = W_ee[:300, :10]
    W_ie = W_ie[:300, :10]

    # Create x-variable for weight matrix
    x = np.arange(0, W_se.shape[0])

    # Define color and label mapping for plots
    weight_plots = {
        "W_se": {"data": W_se, "color": "red"},
        "W_ee": {"data": W_ee, "color": "blue"},
        "W_ie": {"data": W_ie, "color": "green"},
    }

    # Plot weights
    for key, info in weight_plots.items():
        for i, weights in enumerate(info["data"].T):
            if i == 0:
                axs[1].plot(
                    x, weights, color=info["color"], label=key
                )  # Label only the first line of each type
            else:
                axs[1].plot(x, weights, color=info["color"])

    axs[1].set_title("Weight matrix changes")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Weight value")
    axs[1].legend()

    plt.show()
