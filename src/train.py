from numba.typed import List
from numba import njit
from tqdm import tqdm
import numpy as np
from weight_funcs import sleep_func, spike_timing, vectorized_trace_func


@njit
def clip_weights(
    weights,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
):
    for i_ in range(nz_rows_exc.shape[0]):
        i, j = nz_rows_exc[i_], nz_cols_exc[i_]
        if weights[i, j] < min_weight_exc:
            weights[i, j] = min_weight_exc
        elif weights[i, j] > max_weight_exc:
            weights[i, j] = max_weight_exc
    for i_ in range(nz_rows_inh.shape[0]):
        i, j = nz_rows_inh[i_], nz_cols_inh[i_]
        if weights[i, j] < min_weight_inh:
            weights[i, j] = min_weight_inh
        elif weights[i, j] > max_weight_inh:
            weights[i, j] = max_weight_inh
    return weights


def update_weights(
    weights,
    spike_times,
    timing_update,
    trace_update,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    nonzero_pre_idx,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    noisy_weights,
    weight_mean_noise,
    weight_var_noise,
    w_target_exc,
    w_target_inh,
    max_sum_exc,
    max_sum_inh,
    vectorized_trace,
    baseline_sum_exc,
    baseline_sum_inh,
    check_sleep_interval,
    sleep_synchronized,
    baseline_sum,
    nz_rows,
    nz_cols,
    max_sum,
    spikes,
    N_inh,
    N,
    N_exc,
    sleep,
    sleep_now_inh,
    sleep_now_exc,
    delta_w,
    N_x,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
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
    # Clip weights before sleep to prevent zero/negative weights
    weights = clip_weights(
        weights=weights,
        nz_cols_exc=nz_cols_exc,
        nz_cols_inh=nz_cols_inh,
        nz_rows_exc=nz_rows_exc,
        nz_rows_inh=nz_rows_inh,
        min_weight_exc=min_weight_exc,
        max_weight_exc=max_weight_exc,
        min_weight_inh=min_weight_inh,
        max_weight_inh=max_weight_inh,
    )

    if sleep:
        now = t % check_sleep_interval
        if now == 0 or sleep_now_inh or sleep_now_exc:
            # Recompute nonzero indices to reflect current weight state
            st = N_x
            ex = st + N_exc
            ih = ex + N_inh

            weights, sleep_now_inh, sleep_now_exc = sleep_func(
                weights=weights,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                sleep_now_inh=sleep_now_inh,
                sleep_now_exc=sleep_now_exc,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                sleep_synchronized=sleep_synchronized,
                baseline_sum=baseline_sum,
                nz_rows=nz_rows,
                nz_cols=nz_cols,
                max_sum=max_sum,
                nz_rows_exc=nz_rows_exc,
                nz_rows_inh=nz_rows_inh,
                nz_cols_exc=nz_cols_exc,
                nz_cols_inh=nz_cols_inh,
            )

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
            spikes=spikes,
            nonzero_pre_idx=nonzero_pre_idx,
        )

    if vectorized_trace:
        weights = vectorized_trace_func(
            check_sleep_interval=check_sleep_interval,
            spikes=spikes,
            N_x=N_x,
            nz_rows=nz_rows,
            nz_cols=nz_cols,
            delta_w=delta_w,
            A_plus=A_plus,
            dt=dt,
            A_minus=A_minus,
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

    return weights, sleep_now_inh, sleep_now_exc


def update_membrane_potential(
    mp,
    weights,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance,
    tau_m,
    dt,
    I_syn,
    tau_syn,
    mean_noise,
    var_noise,
    sleep_exc,
    sleep_inh,
):
    if noisy_potential and (sleep_exc or sleep_inh):
        gaussian_noise = np.random.normal(
            loc=mean_noise, scale=var_noise, size=mp.shape
        )
    else:
        gaussian_noise = 0

    mp_new = mp.copy()
    # Update synaptic current (avoid in-place modification)
    I_syn_new = I_syn + (-I_syn + np.dot(weights.T, spikes)) * dt / tau_syn
    mp_delta = (
        (-(mp - resting_potential) + membrane_resistance * I_syn_new) / tau_m * dt
    )
    mp_new += mp_delta + gaussian_noise
    return mp_new, I_syn_new


def update_spikes(
    st,
    ih,
    mp,
    dt,
    a,
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

    # update spikes array
    mp = np.clip(mp, a_min=min_mp, a_max=max_mp)
    spikes[st:ih][mp > spike_threshold] = 1

    # Add Solve's noisy membrane potential
    if noisy_threshold:
        delta_potential = spike_threshold - mp
        p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
            1 + np.exp(spike_slope * delta_potential + spike_intercept)
        )
        additional_spikes = np.random.binomial(n=1, p=p_fire)
        spikes[st:ih] = spikes[st:ih] | additional_spikes

    # add spike adaption
    if spike_adaption:
        # --- each timestep ---
        a += (-a / tau_adaption) * dt
        a[spikes[st:ih] == 1] += delta_adaption

        spike_threshold = spike_threshold_default + a
        spike_threshold = np.clip(spike_threshold, -90, 0)

    mp[spikes[st:ih] == 1] = reset_potential
    spike_times = np.where(spikes == 1, 0, spike_times + 1)

    return mp, spikes, spike_times, spike_threshold, a


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
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
    sleep_synchronized,
    baseline_sum,
    max_sum,
    N_inh,
    N_exc,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    max_mp,
    min_mp,
    check_sleep_interval,
    w_target_exc,
    w_target_inh,
    baseline_sum_exc,
    baseline_sum_inh,
    max_sum_exc,
    max_sum_inh,
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
    weight_mean_noise,
    weight_var_noise,
    timing_update,
    trace_update,
    interval,
    N_x,
    T,
    tau_syn,
    beta,
    mean_noise,
    var_noise,
    num_exc,
    num_inh,
    I_syn,
    a,
    spike_threshold,
):

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory
    exc_interval = np.arange(st, ex)
    inh_interval = np.arange(ex, ih)
    idx_exc = np.random.choice(exc_interval, size=num_exc, replace=False)
    idx_inh = np.random.choice(inh_interval, size=num_inh, replace=False)

    plot_positions = np.arange(1, T, interval)
    weights_4_plotting_exc = np.zeros((plot_positions.shape[0], num_exc, ih - st))
    weights_4_plotting_inh = np.zeros((plot_positions.shape[0], num_inh, ex - st))
    weights_4_plotting_exc[0] = weights[idx_exc, st:ih]
    weights_4_plotting_inh[0] = weights[idx_inh, st:ex]

    delta_w = np.zeros(shape=weights.shape)

    nz_rows_inh, nz_cols_inh = np.nonzero(weights[ex:ih, st:ex])
    nz_rows_exc, nz_cols_exc = np.nonzero(weights[:ex, :ih])
    nz_rows_inh += ex
    nz_cols_inh += st
    nz_rows, nz_cols = np.nonzero(weights)
    sleep_now_inh = False
    sleep_now_exc = False

    # Suppose weights is your initial 2D numpy array of weights.
    # Here, we assume that the columns correspond to post-neurons.
    if train_weights:
        desc = "Training network:"
    else:
        desc = "Testing network:"

    # Compute for neurons N_x to N_post-1
    nonzero_pre_idx = List()
    for i in range(st, ih):
        pre_idx = np.nonzero(weights[:ih, i])[0]
        nonzero_pre_idx.append(pre_idx.astype(np.int64))

    idx = 0
    sleep_amount = 0
    for t in tqdm(range(1, T), desc=desc, leave=False):
        # update membrane potential
        mp[t], I_syn = update_membrane_potential(
            mp=mp[t - 1],
            weights=weights[:, st:ih],
            spikes=spikes[t - 1],
            resting_potential=resting_potential,
            membrane_resistance=membrane_resistance,
            tau_m=tau_m,
            dt=dt,
            noisy_potential=noisy_potential,
            mean_noise=mean_noise,
            var_noise=var_noise,
            I_syn=I_syn,
            tau_syn=tau_syn,
            sleep_exc=sleep_now_exc,
            sleep_inh=sleep_now_inh,
        )

        # update spikes array
        (
            mp[t],
            spikes[t],
            spike_times,
            spike_threshold,
            a,
        ) = update_spikes(
            st=st,
            ih=ih,
            mp=mp[t],
            dt=dt,
            a=a,
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
            spike_threshold=spike_threshold,
            spike_threshold_default=spike_threshold_default,
            reset_potential=reset_potential,
        )
        # print(np.mean(spike_threshold))

        # update weights
        if train_weights:
            weights, sleep_now_inh, sleep_now_exc = update_weights(
                spikes=spikes[t - 1],
                weights=weights,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                check_sleep_interval=check_sleep_interval,
                sleep=sleep,
                nonzero_pre_idx=nonzero_pre_idx,
                N_x=N_x,
                vectorized_trace=vectorized_trace,
                delta_w=delta_w,
                N_exc=N_exc,
                timing_update=timing_update,
                trace_update=trace_update,
                spike_times=spike_times,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                sleep_now_inh=sleep_now_inh,
                sleep_now_exc=sleep_now_exc,
                t=t,
                N=N,
                sleep_synchronized=sleep_synchronized,
                baseline_sum=baseline_sum,
                nz_rows=nz_rows,
                nz_cols=nz_cols,
                max_sum=max_sum,
                nz_cols_exc=nz_cols_exc,
                nz_cols_inh=nz_cols_inh,
                nz_rows_exc=nz_rows_exc,
                nz_rows_inh=nz_rows_inh,
                N_inh=N_inh,
                A_plus=A_plus,
                A_minus=A_minus,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                dt=dt,
            )

            if t == plot_positions[idx]:
                weights_4_plotting_exc[idx] = weights[idx_exc, st:ih]
                weights_4_plotting_inh[idx] = weights[idx_inh, st:ex]
                idx += 1
                if plot_positions.size <= idx:
                    idx -= 1

        # remove training data during sleep
        if sleep:
            if sleep_now_exc and t < T - 2:
                spikes[t + 1, :st] = 0
                spike_labels[t] = -2
                sleep_amount += 1
            if sleep_now_inh and t < T - 2:
                spike_labels[t] = -2

    sleep_percent = (sleep_amount / T) * 100

    return (
        weights,
        spikes,
        mp,
        weights_4_plotting_exc,
        weights_4_plotting_inh,
        spike_threshold,
        max_sum_inh,
        max_sum_exc,
        spike_labels,
        sleep_percent,
        I_syn,
        spike_times,
        a,
    )
