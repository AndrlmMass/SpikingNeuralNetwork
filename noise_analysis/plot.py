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
    plt.title("membrane potential during training")
    plt.xlabel("time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.show()


def weights_plot(weights_plot, N_x, N_inh):
    # simplify weights
    mu_weights = np.mean(weights_plot, axis=2)

    plt.plot(mu_weights[:, N_x:-N_inh], color="blue", label="excitatory")
    plt.plot(mu_weights[:, -N_inh:], color="red", label="inhibitory")
    plt.legend()
    plt.show()
