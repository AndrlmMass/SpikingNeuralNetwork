from numba.typed import List
from tqdm import tqdm
import pickle as pkl
import numpy as np
import os
from weight_funcs import sleep_func, spike_timing, vectorized_trace_func, trace_STDP


def update_weights(
    weights,
    spike_times,
    timing_update,
    non_weight_mask,
    pre_trace,
    post_trace,
    trace_update,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    nonzero_pre_idx,
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
    weight_mask,
    w_target_exc,
    w_target_inh,
    max_sum_exc,
    max_sum_inh,
    clip_exc_weights,
    clip_inh_weights,
    vectorized_trace,
    baseline_sum_exc,
    baseline_sum_inh,
    check_sleep_interval,
    spikes,
    N_inh,
    N,
    N_exc,
    sleep,
    sleep_now,
    delta_w,
    N_x,
    nz_cols,
    nz_rows,
    t,
    dt,
    A_plus,
    A_minus,
    learning_rate_exc,
    learning_rate_inh,
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
    if sleep:
        now = t % check_sleep_interval
        if now == 0 or sleep_now:
            weights, sleep_now = sleep_func(
                weights=weights,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                sleep_now=sleep_now,
                N_inh=N_inh,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
            )

    # Find the neurons that spiked in the current timestep
    spike_idx = spikes == 1

    # If no neurons spiked, return weights unchanged
    if not np.any(spike_idx):
        # Precompute exponential decay factors
        decay_pre_exc = np.exp(-dt / tau_pre_trace_exc)
        decay_pre_inh = np.exp(-dt / tau_pre_trace_inh)
        decay_post_exc = np.exp(-dt / tau_post_trace_exc)
        decay_post_inh = np.exp(-dt / tau_post_trace_inh)

        # Update eligibility traces
        pre_trace[:-N_inh] *= decay_pre_exc
        pre_trace[-N_inh:] *= decay_pre_inh
        post_trace[:-N_inh] *= decay_post_exc
        post_trace[-N_inh:] *= decay_post_inh

        return weights, pre_trace, post_trace, sleep_now

    if timing_update:
        weights = spike_timing(
            spike_times=spike_times,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            learning_rate_exc=learning_rate_exc,
            learning_rate_inh=learning_rate_inh,
            N_inh=N_inh,
            weights=weights,
            N_x=N_x,
        )

    if trace_update:
        post_trace, pre_trace, weights = trace_STDP(
            spikes=spikes,
            weights=weights,
            pre_trace=pre_trace,
            post_trace=post_trace,
            learning_rate_exc=learning_rate_exc,
            learning_rate_inh=learning_rate_inh,
            dt=dt,
            N_x=N_x,
            A_plus=A_plus,
            A_minus=A_minus,
            N_inh=N_inh,
            N=N,
            N_exc=N_exc,
            nonzero_pre_idx=nonzero_pre_idx,
            tau_pre_trace_exc=tau_pre_trace_exc,
            tau_pre_trace_inh=tau_pre_trace_inh,
            tau_post_trace_exc=tau_post_trace_exc,
            tau_post_trace_inh=tau_post_trace_inh,
        )

    if vectorized_trace:
        weights, pre_trace, post_trace = vectorized_trace_func(
            check_sleep_interval=check_sleep_interval,
            spikes=spikes,
            N_x=N_x,
            nz_rows=nz_rows,
            nz_cols=nz_cols,
            delta_w=delta_w,
            A_plus=A_plus,
            dt=dt,
            A_minus=A_minus,
            pre_trace=pre_trace,
            post_trace=post_trace,
            tau_pre_trace_exc=tau_pre_trace_exc,
            tau_pre_trace_inh=tau_pre_trace_inh,
            tau_post_trace_exc=tau_post_trace_exc,
            tau_post_trace_inh=tau_post_trace_inh,
            N_inh=N_inh,
        )

    # add noise to weights if desired
    if noisy_weights:
        delta_weight_noise = np.random.normal(
            loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
        )
        weights += delta_weight_noise

    # Update weights
    # print(
    #     f"\rexc weight change:{np.round(np.mean(delta_weights_exc),6)}, inh weight change:{np.round(np.mean(delta_weights_inh),6)}",
    #     end="",
    # )

    if clip_exc_weights:
        weights[:-N_inh] = np.clip(
            weights[:-N_inh], a_min=min_weight_exc, a_max=max_weight_exc
        )

    if clip_inh_weights:
        weights[-N_inh:] = np.clip(
            weights[-N_inh:], a_min=min_weight_inh, a_max=max_weight_inh
        )

    weights[non_weight_mask] = 0

    return weights, pre_trace, post_trace, sleep_now


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
        threshold_decay = (spike_threshold_default - spike_threshold) / tau_adaption
        spike_threshold += threshold_decay

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
        delta_adapt_dt = delta_adaption * dt
        delta_spike_threshold = spike_threshold[spikes[N_x:] == 1] * delta_adapt_dt
        spike_threshold[spikes[N_x:] == 1] -= delta_spike_threshold
        # print(f"\r{np.mean(spike_threshold)}", end="")

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
    clip_exc_weights,
    clip_inh_weights,
    check_sleep_interval,
    w_target_exc,
    w_target_inh,
    dt,
    N,
    A_plus,
    A_minus,
    tau_m,
    sleep,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold_default,
    reset_potential,
    spike_slope,
    spike_intercept,
    noisy_threshold,
    noisy_weights,
    vectorized_trace,
    spike_labels,
    alpha,
    weight_mean_noise,
    weight_var_noise,
    timing_update,
    trace_update,
    w_interval,
    interval,
    save,
    N_x,
    T,
    beta,
    mean_noise,
    var_noise,
):
    weight_mask = weights != 0
    non_weight_mask = weights == 0

    # create weights_plotting_array
    weights_4_plotting = np.zeros((T // interval, N, N - N_x))
    weights_4_plotting[0] = weights[:, N_x:]
    pre_trace_4_plot = np.zeros((T // interval, N))
    post_trace_4_plot = np.zeros((T // interval, N - N_x))

    # create spike threshold array
    spike_threshold = np.full(
        shape=(T, N - N_x), fill_value=spike_threshold_default, dtype=float
    )

    sum_weights_exc = np.sum(np.abs(weights[:-N_inh]))
    sum_weights_inh = np.sum(np.abs(weights[-N_inh:]))

    baseline_sum_exc = sum_weights_exc * beta
    baseline_sum_inh = sum_weights_inh * beta
    max_sum_exc = baseline_sum_exc * alpha
    max_sum_inh = baseline_sum_inh * alpha
    delta_w = np.zeros(shape=weights.shape)

    nz_rows, nz_cols = np.nonzero(weights)
    sleep_now = False

    # Suppose weights is your initial 2D numpy array of weights.
    # Here, we assume that the columns correspond to post-neurons.
    N, N_post = weights.shape

    # Create a typed list to hold, for each post neuron, the indices of pre neurons
    nonzero_pre_idx = List()
    for i in range(N_post):
        # np.nonzero returns a tuple; we take the first element (the indices)
        pre_idx = np.nonzero(weights[:, i])[0]
        # Ensure the pre_idx array is of type int64 (or another fixed int type)
        nonzero_pre_idx.append(pre_idx.astype(np.int64))

    for t in tqdm(range(1, T), desc="Training network:"):

        # update membrane potential
        mp[t] = update_membrane_potential(
            mp=mp[t - 1],
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
        (
            mp[t],
            spikes[t],
            spike_times,
            spike_threshold[t],
        ) = update_spikes(
            N_x=N_x,
            mp=mp[t],
            dt=dt,
            spikes=spikes[t],
            spike_times=spike_times,
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
            weights, pre_trace, post_trace, sleep_now = update_weights(
                spikes=spikes[t - 1],
                weights=weights,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                non_weight_mask=non_weight_mask,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                check_sleep_interval=check_sleep_interval,
                sleep=sleep,
                nonzero_pre_idx=nonzero_pre_idx,
                N_x=N_x,
                vectorized_trace=vectorized_trace,
                delta_w=delta_w,
                N_exc=N_exc,
                weight_mask=weight_mask,
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
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                clip_exc_weights=clip_exc_weights,
                clip_inh_weights=clip_inh_weights,
                sleep_now=sleep_now,
                t=t,
                N=N,
                nz_cols=nz_cols,
                nz_rows=nz_rows,
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
        if t % interval == 0 and t != T:
            weights_4_plotting[t // interval] = weights[:, N_x:]
            pre_trace_4_plot[t // interval] = pre_trace
            post_trace_4_plot[t // interval] = post_trace

        if sleep_now:
            spikes[t, :N_x] = 0
            spike_labels[t] = -2

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
        pre_trace_4_plot,
        post_trace_4_plot,
        spike_threshold,
        weight_mask,
    )
