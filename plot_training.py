# This is the plotting functions script

# Import libraries
import numpy as np
import imageio
import matplotlib.pyplot as plt

def plot_spikes(num_neurons_to_plot=None, num_items_to_plot=None, t_since_spike=None):
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Item {item+1} - Spike Raster Plot')
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

def plot_weights(weights, num_weights):
    # Flatten the first two dimensions
    flattened_weights = weights.reshape(-1, num_weights)

    # Plotting
    time_steps = range(num_weights)
    for i in range(flattened_weights.shape[0]):
        plt.plot(time_steps, flattened_weights[i, :], label=f'Weight {i+1}')

    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.title('Weight Changes Over Time')
    plt.legend()
    plt.show()