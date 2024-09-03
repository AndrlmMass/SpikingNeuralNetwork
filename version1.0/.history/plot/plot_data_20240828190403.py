# define function to create a raster plot of the input data
import matplotlib.pyplot as plt
import numpy as np
import os

if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"


def raster_plot_other(data, labels, save):
    # Create raster plot with dots
    plt.figure(figsize=(10, 6))
    for neuron_index in range(data.shape[1]):
        spike_times = np.where(data[:, neuron_index] == 1)[0]
        plt.scatter(
            spike_times, np.ones_like(spike_times) * neuron_index, color="black", s=10
        )
    t = 0

    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.show()

    if save:
        plt.savefig(f"\\plot_files\\data_plots\\init_plot")


# Plot input_data structure to ensure realistic creation
def input_space_plotted_single(data):

    # The function receives a 2D array of values
    sqr_side = int(np.sqrt(data.shape))

    # Convert 1D array to 2D
    data = np.reshape(data, (sqr_side, sqr_side))

    # Create a plt subplot
    fig, ax = plt.subplots()

    # Create plot
    ax.imshow(data, cmap="Greys", interpolation="nearest")

    plt.grid(visible=True, which="both")
    plt.show()


def raster_plot(spikes):
    # Get firing times for each neuron
    Firing_times = [np.where(spikes[:, n])[0] for n in range(spikes.shape[1])]

    # Plot spike raster
    plt.eventplot(Firing_times, colors="black")
    plt.title("Spikes during training")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")

    plt.show()
