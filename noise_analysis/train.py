import os
import numpy as np
import pickle as pkl


def update_weights(weights, spikes, elig_trace):
    ...
    # will add Hebbian learning rule here

    return weights


def update_membrane_potential(
    spikes,
    weights,
    mp,
    resting_potential,
    membrane_resistance,
    membrane_conductance,
    dt,
    very_small_value,
):
    I_in = np.dot(weights.T, spikes)
    mp_delta = (
        (I_in - ((mp - resting_potential + very_small_value) / membrane_resistance))
        * dt
        / membrane_conductance
    )
    mp += mp_delta

    return mp


def train_network(
    weights,
    mp,
    spikes,
    elig_trace,
    resting_potential,
    membrane_resistance,
    membrane_conductance,
    very_small_value,
    dt,
    spike_threshold,
    reset_potential,
    tau_trace,
    save,
    N_x,
    T,
):

    for t in range(1, T):
        # update membrane potential
        mp[t] = update_membrane_potential(
            mp=mp[t - 1],
            weights=weights[:, N_x:],
            spikes=spikes[t - 1],
            resting_potential=resting_potential,
            membrane_resistance=membrane_resistance,
            membrane_conductance=membrane_conductance,
            very_small_value=very_small_value,
            dt=dt,
        )

        # update spikes array
        spikes[t, N_x:][mp[t] > spike_threshold] = 1
        mp[t][spikes[t, N_x:] == 1] = reset_potential

        # update eligibility trace
        elig_trace[t] = elig_trace[t - 1] * (1 - (1 / tau_trace))
        elig_trace[t][spikes[t] == 1] += 1

        # update weights
        weights = update_weights(
            weights=weights, spikes=spikes[t - 1], elig_trace=elig_trace[t - 1]
        )

    if save:
        file_name = "trained_weights/weights.pkl"

        if not os.path.exists(file_name):
            os.makedirs("trained_weights")

        with open(file_name, "wb") as file:
            pkl.dump(weights, file)

    return weights, spikes, elig_trace, mp
