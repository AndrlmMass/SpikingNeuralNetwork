import os
from tqdm import tqdm
import numpy as np
import pickle as pkl


def update_weights(
    weights,
    spike_times,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    N_inh,
    learning_rate_exc,
    learning_rate_inh,
    tau_pre,
    tau_post,
):
    """
    Apply the STDP rule to update synaptic weights using a fully vectorized approach.

    Parameters:
    - weights: Matrix of weights (2D or 1D, depending on connections).
    - spike_times: Array of time since the last spike for each neuron (0 indicates a spike at the current timestep).
    - min_weight_exc, max_weight_exc: Min and max weights for excitatory synapses.
    - min_weight_inh, max_weight_inh: Min and max weights for inhibitory synapses.
    - N_inh: Number of inhibitory neurons.
    - learning_rate_exc, learning_rate_inh: Learning rates for excitatory and inhibitory weights.
    - tau_pre, tau_post: Time constants for pre- and post-synaptic STDP components.

    Returns:
    - Updated weights.
    """
    # Find the neurons that spiked in the current timestep
    spiking_neurons = spike_times == 0

    # If no neurons spiked, return weights unchanged
    if not np.any(spiking_neurons):
        return weights

    # Compute pairwise time differences for all neurons
    time_diff = np.subtract.outer(spike_times, spike_times)

    # Mask time differences to only consider interactions involving spiking neurons
    spike_mask = (
        spiking_neurons[:, None] | spiking_neurons[None, :]
    )  # Broadcast mask for any interaction with spiking neurons
    masked_time_diff = (
        time_diff * spike_mask
    )  # Mask time differences; irrelevant entries become 0

    # STDP update rule
    stdp_update = np.zeros_like(masked_time_diff)

    # Potentiation for Δt > 0 (pre-spike before post-spike)
    stdp_update[masked_time_diff > 0] = np.exp(
        -masked_time_diff[masked_time_diff > 0] / tau_pre
    )

    # Depression for Δt < 0 (post-spike before pre-spike)
    stdp_update[masked_time_diff < 0] = -np.exp(
        masked_time_diff[masked_time_diff < 0] / tau_post
    )

    # Separate updates for excitatory and inhibitory neurons
    delta_weights_exc = (
        learning_rate_exc * stdp_update[:, :-N_inh]
    )  # For excitatory connections
    delta_weights_inh = (
        learning_rate_inh * stdp_update[:, -N_inh:]
    )  # For inhibitory connections

    # print(
    #     f"\rmean exc delta: {np.mean(delta_weights_exc)}, mean inh delta: {np.mean(delta_weights_inh)}",
    #     end="",
    # )

    # Update weights
    weights[:-N_inh] += np.sum(
        delta_weights_exc.T, axis=0
    )  # Summing contributions from all spikes
    weights[:-N_inh] = np.clip(weights[:-N_inh], min_weight_exc, max_weight_exc)

    weights[-N_inh:] += np.sum(delta_weights_inh.T, axis=0)
    weights[-N_inh:] = np.clip(weights[-N_inh:], min_weight_inh, max_weight_inh)

    return weights


def update_membrane_potential(
    spikes,
    weights,
    mp,
    resting_potential,
    membrane_resistance,
    tau_m,
    dt,
    mean_noise,
    var_noise,
    max_mp,
    min_mp,
):
    mp_new = mp.copy()
    I_in = np.dot(weights.T, spikes)
    mp_delta = (-(mp - resting_potential) + membrane_resistance * I_in) / tau_m * dt
    mp_new += mp_delta + np.random.normal(
        loc=mean_noise, scale=var_noise, size=mp.shape
    )  # Gaussian noise

    return mp_new


def train_network(
    weights,
    mp,
    spikes,
    elig_trace,
    resting_potential,
    membrane_resistance,
    spike_times,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    N_inh,
    N_exc,
    learning_rate_exc,
    learning_rate_inh,
    tau_pre,
    tau_post,
    max_mp,
    min_mp,
    dt,
    N,
    tau_m,
    spike_threshold,
    reset_potential,
    interval,
    save,
    N_x,
    T,
    mean_noise,
    var_noise,
):
    # create weights_plotting_array
    weights_4_plotting = np.zeros((T // interval, N_exc + N_inh, N))

    for t in tqdm(range(1, T)):
        # update membrane potential
        mp[t] = update_membrane_potential(
            mp=mp[t - 1].copy(),
            weights=weights[:, N_x:],
            spikes=spikes[t - 1],
            resting_potential=resting_potential,
            membrane_resistance=membrane_resistance,
            tau_m=tau_m,
            dt=dt,
            mean_noise=mean_noise,
            var_noise=var_noise,
            max_mp=max_mp,
            min_mp=min_mp,
        )

        # update spikes array
        mp[t] = np.clip(mp[t], a_min=min_mp, a_max=max_mp)
        spikes[t, N_x:][mp[t] > spike_threshold] = 1
        mp[t][mp[t] > spike_threshold] = reset_potential
        spike_times = np.where(spikes[t] == 1, 0, spike_times)

        d = np.min(mp[t])
        if d < min_mp:
            print(d)

        # update eligibility trace
        # elig_trace[t] = elig_trace[t - 1] - (elig_trace[t - 1] / tau_trace)
        # elig_trace[t][spikes[t] == 1] += 1

        # update weights
        weights = update_weights(
            weights,
            spike_times,
            min_weight_exc,
            max_weight_exc,
            min_weight_inh,
            max_weight_inh,
            N_inh,
            learning_rate_exc,
            learning_rate_inh,
            tau_pre,
            tau_post,
        )

        # save weights for plotting
        if t % interval == 0:
            weights_4_plotting[t // interval] = weights[N_x:]

    if save:
        file_name = "trained_weights/weights.pkl"

        if not os.path.exists(file_name):
            os.makedirs("trained_weights")

        with open(file_name, "wb") as file:
            pkl.dump(weights, file)

    return weights, spikes, elig_trace, mp, weights_4_plotting
