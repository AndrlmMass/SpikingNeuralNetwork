import matplotlib.pyplot as plt
import numpy as np


# Create weight array
def create_weights(
    N_exc,
    N_inh,
    N_x,
    weight_affinity_hidden_exc,
    weight_affinity_hidden_inh,
    weight_affinity_input,
    pos_weight,
    neg_weight,
    plot_weights,
):
    N = N_exc + N_inh + N_x

    # Create weights based on affinity rate
    mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
    mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
    mask_input = np.random.random((N, N)) < weight_affinity_input
    weights = np.zeros(shape=(N, N))

    # input_weights
    weights[:N_x, N_x:-N_inh][mask_input[:N_x, N_x:-N_inh]] = pos_weight
    # excitatory weights
    weights[N_x:-N_inh, N_x:][mask_hidden_exc[N_x:-N_inh, N_x:]] = pos_weight
    # inhibitory weights
    weights[-N_inh:, N_x:-N_inh][mask_hidden_inh[-N_inh:, N_x:-N_inh]] = neg_weight
    # remove self-connections (diagonal) to 0 for excitatory weights
    np.fill_diagonal(weights[N_x:-N_inh, N_x:-N_inh], 0)

    if plot_weights:
        plt.imshow(weights)
        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.show()

    return weights


def create_arrays(
    N,
    resting_membrane,
    total_time_train,
    total_time_test,
    max_time,
    data_train,
    data_test,
    N_x,
):
    membrane_potential_train = np.zeros((total_time_train, N - N_x))
    membrane_potential_train[0] = resting_membrane

    membrane_potential_test = np.zeros((total_time_test, N - N_x))
    membrane_potential_test[0] = resting_membrane

    pre_trace = np.zeros((N))
    post_trace = np.zeros((N - N_x))

    spikes_train = np.zeros((total_time_train, N), dtype="int64")
    spikes_train[:, :N_x] = data_train

    spikes_test = np.zeros((total_time_test, N), dtype="int64")
    spikes_test[:, :N_x] = data_test

    spike_times = np.random.randint(low=max_time, high=max_time**2, size=N)

    return (
        membrane_potential_train,
        membrane_potential_test,
        pre_trace,
        post_trace,
        spikes_train,
        spikes_test,
        spike_times,
    )
