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


def create_arrays(N, resting_membrane, total_time, data, N_x):
    membrane_potential = np.full((N, total_time), fill_value=resting_membrane)

    trace = np.zeros((N, total_time))

    spikes = np.zeros((N, total_time))
    spikes[:N_x] = np.transpose(data)

    return membrane_potential, trace, spikes
