import matplotlib.pyplot as plt
import numpy as np


# Create weight array
def create_weigths(
    N_exc,
    N_inh,
    N_x,
    weight_affinity_hidden,
    weight_affinity_input,
    pos_weight,
    neg_weight,
    plot_weights,
):
    N = N_exc + N_inh + N_x

    # Create weights based on affinity rate
    mask_hidden = np.random.random((N, N)) < weight_affinity_hidden
    mask_input = np.random.random((N, N)) < weight_affinity_input
    weights = np.zeros(shape=(N, N))

    # input_weights
    weights[:N_x, N_x:-N_inh][mask_input[:N_x, N_x:-N_inh]] = pos_weight
    # excitatory weights
    weights[N_x:-N_inh, N_x:][mask_hidden[N_x:-N_inh, N_x:]] = pos_weight
    # inhibitory weights
    weights[-N_inh:, N_x:][mask_hidden[-N_inh:, N_x:]] = neg_weight

    if plot_weights:
        plt.imshow(weights)
        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.show()

    return weights


def create_arrays(N, resting_membrane, total_time, max_time, data, N_x):
    membrane_potential = np.zeros((total_time, N - N_x))
    membrane_potential[0] = resting_membrane

    trace = np.zeros((total_time, N))
    trace[0] = 1

    spikes = np.zeros((total_time, N))
    spikes[:, :N_x] = data

    spike_times = np.full(N, fill_value=max_time + 1)

    return membrane_potential, trace, spikes, spike_times