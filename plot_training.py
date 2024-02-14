# This is the plotting functions script

# Import libraries
import numpy as np
import imageio
import matplotlib.pyplot as plt


def plot_spikes(
    num_neurons_to_plot=None,
    num_items_to_plot=None,
    t_since_spike=None,
    input_indices=None,
):
    num_items = t_since_spike.shape[2]
    num_neurons = t_since_spike.shape[1]

    # If the arguments are not provided, plot all neurons and items
    if num_neurons_to_plot is None:
        num_neurons_to_plot = num_neurons
    if num_items_to_plot is None:
        num_items_to_plot = num_items

    # Define colors for each neuron
    colors = plt.cm.jet(np.linspace(0, 1, num_neurons_to_plot))

    # Time shift to represent each item sequentially
    time_shift_per_item = t_since_spike.shape[
        0
    ]  # Assuming the time window is equal to the number of time steps in each item

    # Initialize spike data for all neurons and items
    spike_data = [[] for _ in range(num_neurons_to_plot)]

    for item in range(min(num_items_to_plot, num_items)):
        for neuron in range(min(num_neurons_to_plot, num_neurons)):
            # Find the time steps where the neuron fired
            neuron_spike_times = np.where(t_since_spike[:, neuron, item] == 0)[0]
            # Shift spike times for sequential representation and append to spike data
            spike_data[neuron].extend(neuron_spike_times + item * time_shift_per_item)

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set lineoffsets and linelengths for spacing
    lineoffsets = np.arange(num_neurons_to_plot)
    linelengths = 0.8  # Adjust this value to control the length of spikes

    # Spike Raster Plot
    ax.eventplot(
        spike_data, lineoffsets=lineoffsets, linelengths=linelengths, colors=colors
    )
    ax.set_yticks(np.arange(num_neurons_to_plot))
    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron")
    ax.set_title(f"Item {item+1} - Spike Raster Plot")

    # Convert the y ticks to red if they are the input neurons
    print(input_indices)
    for idx in input_indices:
        if idx in np.arange(num_neurons_to_plot):
            ax.get_yticklabels()[idx].set_color("red")

    plt.tight_layout()
    plt.show()


def plot_gif_evolution(avg_spike_counts, epochs, num_neurons, num_items):
    for epoch in range(epochs):
        fig, ax = plt.subplots()
        for item in range(num_items):
            ax.bar(
                np.arange(num_neurons) + 0.35 * item,
                avg_spike_counts[:, item],
                width=0.35,
                label=f"Item {item}",
            )

        ax.set_xlabel("Neuron")
        ax.set_ylabel("Average Spike Count")
        ax.set_title(f"Epoch {epoch}")
        ax.legend()
        plt.savefig(f"epoch_{epoch}.png")
        plt.close()

    images = []
    filenames = [f"epoch_{epoch}.png" for epoch in range(epochs)]

    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave("neuron_activity.gif", images, fps=1)


import numpy as np
import matplotlib.pyplot as plt


def plot_weights(weights, dt_items):
    # Assuming `weights` is a 3D numpy array of shape (n, m, time_steps)
    # where n is the number of neurons, m is the number of connections per neuron, and
    # time_steps is the number of time steps recorded.

    time_steps = np.arange(dt_items)  # Create an array of time steps

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            # Check if the weight is non-zero at any time step in the range of interest
            if np.any(weights[i, j, :dt_items] != 0):
                plt.plot(
                    time_steps, weights[i, j, :dt_items], label=f"Weight {i+1},{j+1}"
                )

    plt.xlabel("Time")
    plt.ylabel("Weight Value")
    plt.title("Weight Changes Over Time")

    # Optional: Comment out the legend if there are too many lines to make it readable
    # plt.legend()

    plt.show()


def plot_membrane_activity(MemPot, num_neurons, num_items):
    fig, ax = plt.subplots()  # Corrected here
    for j in range(num_neurons):  # Assume num_neurons is an int and use range()
        for i in range(num_items):
            y = MemPot[:, j, i]  # Assuming MemPot is a 3D array; check this matches your data structure
            x = np.arange(len(y))
            ax.plot(x, y, label=f'Neuron {j}')  # Plot on the same axis, added label for clarity
    plt.xlabel("Time")
    plt.ylabel("mV Value")
    plt.title("Membrane Potential Changes Over Time")
    plt.legend()  # Optional, to show legend if labels are added
    plt.show()


def plot_activity_scatter(spikes, classes, num_classes):
    print(spikes.shape, classes.shape)

    # Get the indices for each class and calculate the mean activity for each neuron
    mean_activities = [
        np.mean(spikes[:, :, classes == t], axis=(0, 2)) for t in range(num_classes)
    ]

    # Calculate differences and sort neurons by it
    differences = mean_activities[0] - mean_activities[1]
    sorted_neurons = np.argsort(-np.abs(differences))  # Sort by decreasing difference

    # Create the plot
    plt.figure(figsize=(10, 8))
    for i, neuron_idx in enumerate(sorted_neurons):
        # Plot a line between the points for each neuron
        plt.plot(
            [i, i],
            [mean_activities[0][neuron_idx], mean_activities[1][neuron_idx]],
            "grey",
            zorder=1,
        )
        # Plot the points for each class
        plt.scatter(i, mean_activities[0][neuron_idx], color="blue", zorder=2)
        plt.scatter(i, mean_activities[1][neuron_idx], color="red", zorder=2)

    # Adding labels and title
    plt.title("Sorted Neuronal Activity by Class Difference")
    plt.xlabel("Sorted Neuron Index")
    plt.ylabel("Average Activity")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_relative_activity(spikes, classes, input_idx, num_neurons):
    # Compute the mean activity across all timesteps for each neuron, for each item
    mean_activity = np.mean(spikes, axis=0)  # Resulting shape: (neurons, items)

    # Colors for different classes
    colors = ["blue", "red"]  # Assuming class 0 is blue, class 1 is red

    # Prepare to plot
    plt.figure(figsize=(14, 10))

    neurons = np.arange(num_neurons)
    for j in input_idx:
        neurons = np.delete(neurons, j)
    # Iterate over each neuron
    for neuron_idx in neurons:
        # Handle each class separately
        for class_val in np.unique(classes):
            x_positions = []
            y_positions = []

            # Filter items by class
            class_items = np.where(classes == class_val)[0]

            # Iterate over items of the current class for this neuron
            for item_idx in class_items:
                # Compute x position with slight offset for each neuron to avoid overlap
                x_pos = (
                    item_idx + (neuron_idx * 0.01) - (mean_activity.shape[0] / 2 * 0.01)
                )
                x_positions.append(x_pos)
                y_positions.append(mean_activity[neuron_idx, item_idx])

                # Plot point
                plt.scatter(
                    x_pos,
                    mean_activity[neuron_idx, item_idx],
                    color=colors[class_val],
                    alpha=0.6,
                    edgecolor="none",
                    s=30,
                )

            # Draw lines connecting points for this neuron within the same class
            if len(x_positions) > 1:  # Only draw lines if there are at least two points
                plt.plot(
                    x_positions,
                    y_positions,
                    color=colors[class_val],
                    alpha=0.5,
                    linestyle="-",
                    linewidth=1,
                )

    plt.title("Change in Activity of Each Neuron Over Items by Class")
    plt.xlabel("Item Index (with slight offset for each neuron)")
    plt.ylabel("Mean Activity")
    plt.ylim((0, 0.1))
    plt.grid(True)
    plt.show()
