import numpy as np
import matplotlib.pyplot as plt


def spike_plot(data):
    positions = [np.where(data[:, n] == 1)[0] for n in range(data.shape[1])]
    plt.eventplot(positions=positions)
    plt.title("Spikes")
    plt.show()


def heat_map(data, pixel_size):
    data = data.numpy()
    summed_data = np.sum(data, axis=0)
    reshaped_summed_data = np.reshape(summed_data, (pixel_size, pixel_size))
    plt.imshow(reshaped_summed_data, cmap="hot", interpolation="nearest")
    plt.show()


def mp_plot(mp):
    plt.plot(mp)
    plt.show()
