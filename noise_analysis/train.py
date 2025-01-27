import os
from tqdm import tqdm
import numpy as np
import pickle as pkl


def update_weights(
    weights,
    spike_times,
    timing_update,
    pre_trace,
    post_trace,
    trace_update,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    weight_decay,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    tau_pre_trace_exc,
    tau_pre_trace_inh,
    tau_post_trace_exc,
    tau_post_trace_inh,
    noisy_weights,
    weight_mean_noise,
    weight_var_noise,
    N_inh,
    dt,
    N_x,
    A_plus,
    A_minus,
    learning_rate_exc,
    learning_rate_inh,
    spikes,
    tau_LTP,
    tau_LTD,
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
    # Add weight decay
    if weight_decay:
        decay_exc = np.exp(-weights[:-N_inh] / weight_decay_rate_exc)
        decay_inh = np.exp(-weights[-N_inh:] / weight_decay_rate_inh)
        weights[:-N_inh] *= decay_exc
        weights[-N_inh:] *= decay_inh

    # Find the neurons that spiked in the current timestep
    spike_idx = spike_times == 0

    # If no neurons spiked, return weights unchanged
    if not np.any(spike_idx):
        return weights

    if timing_update:
        # Compute pairwise time differences for all neurons
        time_diff = np.subtract.outer(spike_times, spike_times)

        # Mask time differences to only consider interactions involving spiking neurons
        spike_mask = spike_idx[:, None] | spike_idx[None, :]
        masked_time_diff = np.where(spike_mask == True, time_diff, float("nan"))

        # STDP update rule
        stdp_update = np.zeros_like(masked_time_diff)

        # Potentiation for Δt > 0 (pre-spike before post-spike)
        stdp_update[masked_time_diff >= 0] = np.exp(
            -masked_time_diff[masked_time_diff >= 0] / tau_LTP
        )

        # Depression for Δt < 0 (post-spike before pre-spike)
        stdp_update[masked_time_diff < 0] = -np.exp(
            masked_time_diff[masked_time_diff < 0] / tau_LTD
        )

        # Separate updates for excitatory and inhibitory neurons
        delta_weights_exc = (
            learning_rate_exc * stdp_update[:-N_inh]
        )  # For excitatory connections
        delta_weights_inh = (
            learning_rate_inh * stdp_update[-N_inh:]
        )  # For inhibitory connections

    if trace_update:
        # update eligibility trace
        pre_trace[:, :-N_inh] *= np.exp(-pre_trace[:, :-N_inh] / tau_pre_trace_exc) * dt
        pre_trace[:, -N_inh:] *= np.exp(-pre_trace[:, -N_inh:] / tau_pre_trace_inh) * dt
        post_trace[:-N_inh] *= np.exp(-post_trace[:-N_inh] / tau_post_trace_exc) * dt
        post_trace[-N_inh:] *= np.exp(-post_trace[-N_inh:] / tau_post_trace_inh) * dt

        # create 2d spike array
        spikes_reshaped = spikes[np.newaxis, :]
        spiked_2d = np.repeat(spikes_reshaped, repeats=spikes.shape[0], axis=0)

        # increase trace if neuron spiked
        pre_trace += spiked_2d * dt
        post_trace += spikes[N_x:] * dt

        # update weights for post-neurons that spiked
        weight_mask = weights != 0
        delta_weight = (
            A_plus * pre_trace[weight_mask] * spikes - A_minus * post_trace * spikes
        )

        """
        pre_trace: N x N
        spikes: N
        post_trace: N-N_x
        
        """

    # print(np.sum(delta_weights_inh), np.sum(delta_weights_exc))
    if noisy_weights:
        delta_weight_noise = np.random.normal(
            loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
        )
    else:
        delta_weight_noise = np.zeros(weights.shape)

    # Update weights
    weights += delta_weight_noise * dt

    weights[:-N_inh] = np.clip(weights[:-N_inh], min_weight_exc, max_weight_exc)
    weights[-N_inh:] = np.clip(weights[-N_inh:], min_weight_inh, max_weight_inh)

    return weights, pre_trace, post_trace


def update_membrane_potential(
    spikes,
    weights,
    mp,
    noisy_potential,
    resting_potential,
    membrane_resistance,
    tau_m,
    dt,
    mean_noise,
    var_noise,
):
    if noisy_potential:
        gaussian_noise = np.random.normal(
            loc=mean_noise, scale=var_noise, size=mp.shape
        )
    else:
        gaussian_noise = 0

    mp_new = mp.copy()
    I_in = np.dot(weights.T, spikes)
    mp_delta = (-(mp - resting_potential) + membrane_resistance * I_in) / tau_m * dt
    mp_new += mp_delta + gaussian_noise
    return mp_new


def update_spikes(
    N_x,
    mp,
    dt,
    spikes,
    spike_times,
    max_mp,
    min_mp,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    noisy_threshold,
    spike_slope,
    spike_intercept,
    spike_threshold,
    spike_threshold_default,
    reset_potential,
):
    # update spike threshold
    if spike_adaption:
        spike_threshold += (spike_threshold_default - spike_threshold) / tau_adaption

    # update spikes array
    mp = np.clip(mp, a_min=min_mp, a_max=max_mp)
    spikes[N_x:][mp > spike_threshold] = 1

    # Add Solve's noisy membrane potential
    if noisy_threshold:
        delta_potential = spike_threshold - mp
        p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
            1 + np.exp(spike_slope * delta_potential + spike_intercept)
        )
        additional_spikes = np.random.binomial(n=1, p=p_fire)
        spikes[N_x:] = spikes[N_x:] | additional_spikes

    # add spike adaption
    if spike_adaption:
        spike_threshold[spikes[N_x:] == 1] += (
            spike_threshold[spikes[N_x:] == 1] * delta_adaption * dt
        )
        print(f"\r{np.mean(spike_threshold)}", end="")

    mp[spikes[N_x:] == 1] = reset_potential
    spike_times = np.where(spikes == 1, 0, spike_times + 1)

    return mp, spikes, spike_times, spike_threshold


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
    pre_trace,
    post_trace,
    resting_potential,
    membrane_resistance,
    spike_times,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    train_weights,
    weight_decay,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    N_inh,
    N_exc,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    max_mp,
    min_mp,
    tau_pre_trace_exc,
    tau_pre_trace_inh,
    tau_post_trace_exc,
    tau_post_trace_inh,
    dt,
    N,
    A_plus,
    A_minus,
    tau_m,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold_default,
    reset_potential,
    spike_slope,
    spike_intercept,
    noisy_threshold,
    noisy_weights,
    weight_mean_noise,
    weight_var_noise,
    timing_update,
    trace_update,
    w_interval,
    interval,
    save,
    N_x,
    T,
    mean_noise,
    var_noise,
):
    # create weights_plotting_array
    weights_4_plotting = np.zeros((T // interval, N_exc + N_inh, N))
    weights_4_plotting[0] = weights[N_x:]

    # create spike threshold array
    spike_threshold = np.full(
        shape=(T, N - N_x), fill_value=spike_threshold_default, dtype=float
    )

    print("Training network...")
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
            noisy_potential=noisy_potential,
            mean_noise=mean_noise,
            var_noise=var_noise,
        )

        # update spikes array
        mp[t], spikes[t], spike_times, spike_threshold[t] = update_spikes(
            N_x=N_x,
            mp=mp[t].copy(),
            dt=dt,
            spikes=spikes[t].copy(),
            spike_times=spike_times.copy(),
            spike_intercept=spike_intercept,
            spike_slope=spike_slope,
            noisy_threshold=noisy_threshold,
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            max_mp=max_mp,
            min_mp=min_mp,
            spike_threshold=spike_threshold[t - 1],
            spike_threshold_default=spike_threshold_default,
            reset_potential=reset_potential,
        )

        # update weights
        if train_weights:
            weights, pre_trace, post_trace = update_weights(
                weights=weights,
                spikes=spikes[t],
                pre_trace=pre_trace,
                post_trace=post_trace,
                timing_update=timing_update,
                trace_update=trace_update,
                spike_times=spike_times,
                weight_decay=weight_decay,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                tau_pre_trace_exc=tau_pre_trace_exc,
                tau_pre_trace_inh=tau_pre_trace_inh,
                tau_post_trace_exc=tau_post_trace_exc,
                tau_post_trace_inh=tau_post_trace_inh,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                N_x=N_x,
                N_inh=N_inh,
                A_plus=A_plus,
                A_minus=A_minus,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                dt=dt,
            )
            # print(f"\r{np.max(weights[:N_exc])}, {np.min(weights[N_exc:])}", end="")

        # save weights for plotting
        if t % interval == 0:
            weights_4_plotting[t // interval] = weights[N_x:]

    if save:
        file_name = "trained_weights/weights.pkl"

        if not os.path.exists(file_name):
            os.makedirs("trained_weights")

        with open(file_name, "wb") as file:
            pkl.dump(weights, file)

    return (
        weights,
        spikes,
        pre_trace,
        post_trace,
        mp,
        weights_4_plotting,
        spike_threshold,
    )
