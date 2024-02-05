# This is the plotting functions script

# Import libraries
import numpy as np
import imageio
import matplotlib.pyplot as plt

def plot_spikes(num_neurons_to_plot=None, num_items_to_plot=None, t_since_spike=None, weights=None, input_indices=None):
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
    time_shift_per_item = t_since_spike.shape[0]  # Assuming the time window is equal to the number of time steps in each item

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
    ax.eventplot(spike_data, lineoffsets=lineoffsets, linelengths=linelengths, colors=colors)
    ax.set_yticks(np.arange(num_neurons_to_plot))
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Item {item+1} - Spike Raster Plot')

    # Convert the y ticks to red if they are the input neurons
    for idx in input_indices:
        ax.get_yticklabels()[idx].set_color("red")

    plt.tight_layout()
    plt.show()

def plot_gif_evolution(avg_spike_counts, epochs, num_neurons, num_items):
    for epoch in range(epochs):
        fig, ax = plt.subplots()
        for item in range(num_items):
            ax.bar(np.arange(num_neurons) + 0.35*item, avg_spike_counts[:, item], width=0.35, label=f'Item {item}')
        
        ax.set_xlabel('Neuron')
        ax.set_ylabel('Average Spike Count')
        ax.set_title(f'Epoch {epoch}')
        ax.legend()
        plt.savefig(f'epoch_{epoch}.png')
        plt.close()

    images = []
    filenames = [f'epoch_{epoch}.png' for epoch in range(epochs)]

    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave('neuron_activity.gif', images, fps=1)

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
                plt.plot(time_steps, weights[i, j, :dt_items], label=f'Weight {i+1},{j+1}')

    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.title('Weight Changes Over Time')

    # Optional: Comment out the legend if there are too many lines to make it readable
    # plt.legend()

    plt.show()
