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
    labels_name = ["t", "o", "s", "x", " "]
    indices = np.argmax(labels, axis=1)

    # Create raster plot with dots
    plt.figure(figsize=(10, 6))
    for neuron_index in range(data.shape[1]):
        spike_times = np.where(data[:, neuron_index] == 1)[0]
        plt.scatter(
            spike_times, np.ones_like(spike_times) * neuron_index, color="black", s=10
        )
    t = 0

    for item_boundary in range(0, data.shape[0], 100):
        # Get label name
        plt.axvline(x=item_boundary, color="red", linestyle="--")
        plt.text(
            x=item_boundary + 25,
            y=1700,
            s=labels_name[indices[t]],
            size=12,
        )

        t += 1

    # Calculate the frequency of the spikes to check that it is acceptable
    t_counts = sum([np.sum(data[j : j + 100]) for j in range(0, data.shape[0], 800)])
    c_counts = sum([np.sum(data[j : j + 100]) for j in range(200, data.shape[0], 800)])
    s_counts = sum([np.sum(data[j : j + 100]) for j in range(400, data.shape[0], 800)])
    x_counts = sum([np.sum(data[j : j + 100]) for j in range(600, data.shape[0], 800)])
    b_counts = sum([np.sum(data[j : j + 100]) for j in range(100, data.shape[0], 200)])

    # Calculate the frquency of firing according to this formula: (spikes / possible spikes (units)) * timeunit (this is used to convert the unit to seconds, not milieconds)
    t_hz = t_counts / (100 * (t_counts // 100))
    c_hz = c_counts / (100 * (c_counts // 100))
    s_hz = s_counts / (100 * (s_counts // 100))
    x_hz = x_counts / (100 * (x_counts // 100))
    b_hz = b_counts / (100 * (b_counts // 100))

    print(f"t: {t_hz} Hz\nc: {c_hz} Hz\ns: {s_hz} Hz\nx: {x_hz} Hz\nb: {b_hz} Hz")

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
