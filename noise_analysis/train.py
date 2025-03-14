from numba.typed import List
from numba import njit
from tqdm import tqdm
import numpy as np
from numba import prange
from weight_funcs import sleep_func, spike_timing, vectorized_trace_func, trace_STDP


@njit(parallel=True)
def clip_weights(
    weights,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
    min_weight_exc,
    max_weight_inh,
):
    for i_ in prange(nz_rows_exc.shape[0]):
        i, j = nz_rows_exc[i_], nz_cols_exc[i_]
        if weights[i, j] < min_weight_exc:
            weights[i, j] = min_weight_exc
    for i_ in prange(nz_rows_inh.shape[0]):
        i, j = nz_rows_inh[i_], nz_cols_inh[i_]
        if weights[i, j] > max_weight_inh:
            weights[i, j] = max_weight_inh
    return weights


# @njit(parallel=True)
def update_weights(
    weights,
    spike_times,
    min_weight_exc,
    max_weight_inh,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    w_target_exc,
    w_target_inh,
    max_sum_exc,
    max_sum_inh,
    baseline_sum_exc,
    baseline_sum_inh,
    check_sleep_interval,
    spikes,
    sleep,
    sleep_now_inh,
    sleep_now_exc,
    nz_cols_exc,  # maybe wasteful
    nz_cols_inh,  # maybe wasteful
    nz_rows_exc,  # maybe wasteful
    nz_rows_inh,  # maybe wasteful
    t,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    spiking_nzw_exc,
    spiking_nzw_inh,
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
        if now == 0 or sleep_now_inh or sleep_now_exc:
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
                nz_rows_exc=nz_rows_exc,
                nz_rows_inh=nz_rows_inh,
                nz_cols_exc=nz_cols_exc,
                nz_cols_inh=nz_cols_inh,
            )

    # if timing_update:
    # weights = spike_timing(
    #     spike_times=spike_times,
    #     tau_LTP=tau_LTP,
    #     tau_LTD=tau_LTD,
    #     learning_rate_exc=learning_rate_exc,
    #     learning_rate_inh=learning_rate_inh,
    #     weights=weights,
    #     spiking_nzw_exc=spiking_nzw_exc,
    #     spiking_nzw_inh=spiking_nzw_inh,
    # )

    # if trace_update:
    #     post_trace, pre_trace, weights = trace_STDP(
    #         spikes=spikes,
    #         weights=weights,
    #         pre_trace=pre_trace,
    #         post_trace=post_trace,
    #         learning_rate_exc=learning_rate_exc,
    #         learning_rate_inh=learning_rate_inh,
    #         dt=dt,
    #         N_x=st,
    #         A_plus=A_plus,
    #         A_minus=A_minus,
    #         N_inh=N_inh,
    #         N=N,
    #         N_exc=N_exc,
    #         nonzero_pre_idx=nonzero_pre_idx,
    #         tau_pre_trace_exc=tau_pre_trace_exc,
    #         tau_pre_trace_inh=tau_pre_trace_inh,
    #         tau_post_trace_exc=tau_post_trace_exc,
    #         tau_post_trace_inh=tau_post_trace_inh,
    #     )

    # if vectorized_trace:
    #     weights, pre_trace, post_trace = vectorized_trace_func(
    #         check_sleep_interval=check_sleep_interval,
    #         spikes=spikes,
    #         N_x=N_x,
    #         nz_rows=nz_rows,
    #         nz_cols=nz_cols,
    #         delta_w=delta_w,
    #         A_plus=A_plus,
    #         dt=dt,
    #         A_minus=A_minus,
    #         pre_trace=pre_trace,
    #         post_trace=post_trace,
    #         tau_pre_trace_exc=tau_pre_trace_exc,
    #         tau_pre_trace_inh=tau_pre_trace_inh,
    #         tau_post_trace_exc=tau_post_trace_exc,
    #         tau_post_trace_inh=tau_post_trace_inh,
    #         N_inh=N_inh,
    #     )

    # add noise to weights if desired
    # if noisy_weights:
    #     delta_weight_noise = np.random.normal(
    #         loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
    #     )
    #     weights += delta_weight_noise

    # Update weights
    # print(
    #     f"\rexc weight change:{np.round(np.mean(delta_weights_exc),6)}, inh weight change:{np.round(np.mean(delta_weights_inh),6)}",
    #     end="",
    # )

    # weights = clip_weights(
    #     weights=weights,
    #     nz_cols_exc=nz_cols_exc,
    #     nz_cols_inh=nz_cols_inh,
    #     nz_rows_exc=nz_rows_exc,
    #     nz_rows_inh=nz_rows_inh,
    #     min_weight_exc=min_weight_exc,
    #     max_weight_inh=max_weight_inh,
    # )

    return weights, sleep_now_inh, sleep_now_exc


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

    I_in = np.dot(weights.T, spikes)  # this part slows down computation significantly
    mp += (
        gaussian_noise
        + (-(mp - resting_potential) + membrane_resistance * I_in) / tau_m * dt
    )
    return mp


@njit
def binary_search(a, x):
    lo = 0
    hi = len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == x:
            return True
        elif a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


@njit
def filter_array(arr, sorted_filter):
    # First, count how many elements are allowed:
    count = 0
    for x in arr:
        if binary_search(sorted_filter, x):
            count += 1
    # Allocate output array:
    out = np.empty(count, arr.dtype)
    j = 0
    for x in arr:
        if binary_search(sorted_filter, x):
            out[j] = x
            j += 1
    return out


# @njit(parallel=True)
# @profile
def update_spikes(
    st,
    pn,
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
    weights_exc,
    weights_inh,
    spiking_nzw_exc,
    spiking_nzw_inh,
):
    # update spikes array
    mp = np.clip(mp, a_min=min_mp, a_max=max_mp)
    spikes[st:pn][mp > spike_threshold] = 1

    # Add Solve's noisy membrane potential
    if noisy_threshold:
        delta_potential = spike_threshold - mp
        p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
            1 + np.exp(spike_slope * delta_potential + spike_intercept)
        )
        additional_spikes = np.random.binomial(n=1, p=p_fire)
        spikes[st:pn] = spikes[st:pn] | additional_spikes

    # add spike adaption
    if spike_adaption:
        spike_threshold += (spike_threshold_default - spike_threshold) / tau_adaption
        delta_adapt_dt = delta_adaption * dt
        delta_spike_threshold = spike_threshold[spikes[st:pn] == 1] * delta_adapt_dt
        spike_threshold[spikes[st:pn] == 1] -= delta_spike_threshold
        # print(f"\r{np.mean(spike_threshold)}", end="")

    spike_indices = np.where(spikes[:pn] == 1)[0]
    mp[spikes[st:pn] == 1] = reset_potential
    spike_times[spike_indices] = 0
    spike_times[spikes == 0] += dt

    # spiking_nzw_exc = [filter_array(arr, spike_indices) for arr in weights_exc]
    # spiking_nzw_inh = [filter_array(arr, spike_indices) for arr in weights_inh]

    return (
        mp,
        spikes,
        spike_times,
        spike_threshold,
        spiking_nzw_exc,
        spiking_nzw_inh,
    )


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
    unsupervised,
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
    N_classes,
    supervised,
    interval,
    N_x,
    T,
    beta,
    mean_noise,
    var_noise,
    num_exc,
    num_inh,
):
    weight_mask = weights != 0

    if supervised or unsupervised:
        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih + N_classes  # predicted positive
        pn = pp + N_classes  # predicted negative
        tp = pn + N_classes  # true positive
        tn = tp + N_classes  # true negative
        fp = tn + N_classes  # false positive
        fn = fp + N_classes  # false negative
    else:
        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih  # predicted positive
        pn = ih  # predicted negative
        tp = ih  # true positive
        tn = ih  # true negative
        fp = ih  # false positive
        fn = ih  # false negative

    exc_interval = np.arange(st, ex)
    inh_interval = np.arange(ex, ih)
    idx_exc = np.random.choice(exc_interval, size=num_exc, replace=False)
    idx_inh = np.random.choice(inh_interval, size=num_inh, replace=False)

    # create weights_plotting_array
    weights_4_plotting_exc = np.zeros((T // interval, num_exc, pn - st))
    weights_4_plotting_inh = np.zeros((T // interval, num_inh, ex - st))
    weights_4_plotting_exc[0] = weights[idx_exc, st:pn]
    weights_4_plotting_inh[0] = weights[idx_inh, st:ex]
    pre_trace_4_plot = np.zeros((T // interval, pn))
    post_trace_4_plot = np.zeros((T // interval, pn - st))

    # create spike threshold array
    spike_threshold = np.full(
        shape=(T, pn - st), fill_value=spike_threshold_default, dtype=float
    )

    # define which weights counts towards total sum of weights
    sum_weights_exc = np.sum(np.abs(weights[:ex, st:ih]))
    sum_weights_inh = np.sum(np.abs(weights[ex:ih, st:ex]))

    baseline_sum_exc = sum_weights_exc * beta
    baseline_sum_inh = sum_weights_inh * beta
    max_sum_exc = sum_weights_exc * alpha
    max_sum_inh = sum_weights_inh * alpha

    nz_rows_inh1, nz_cols_inh1 = np.nonzero(weights[ex:ih, st:ex])
    nz_rows_inh2, nz_cols_inh2 = np.nonzero(weights[pp:pn, st:ex])
    nz_rows_inh = np.concatenate((nz_rows_inh1 + ex, nz_rows_inh2 + pp))
    nz_cols_inh = np.concatenate((nz_cols_inh1 + st, nz_cols_inh2 + st))
    nz_rows_exc1, nz_cols_exc1 = np.nonzero(weights[:ex, st:pn])
    nz_rows_exc2, nz_cols_exc2 = np.nonzero(weights[ih:pp, st:ex])
    nz_rows_exc = np.concatenate((nz_rows_exc1, nz_rows_exc2 + ih))
    nz_cols_exc = np.concatenate((nz_cols_exc1 + st, nz_cols_exc2 + st))
    sleep_now_inh = False
    sleep_now_exc = False

    # Suppose weights is your initial 2D numpy array of weights.
    # Here, we assume that the columns correspond to post-neurons.
    if train_weights:
        desc = "Training network:"
    else:
        desc = "Testing network:"

    # Compute for neurons N_x to N_post-1
    """
    Look over the part below. It is not quite right:
    """

    # post exc
    weights_exc = []
    for post_id in range(st, pn):
        idx1 = np.nonzero(weights[:ex, post_id])[0]
        idx2 = np.nonzero(weights[ih:pp, post_id])[0] + ih
        weights_exc.append(np.concatenate((idx1, idx2)))

    # post inh
    weights_inh = []
    for post_id in range(st, ex):
        idx1 = np.nonzero(weights[ex:ih, post_id])[0] + ex
        idx2 = np.nonzero(weights[pp:pn, post_id])[0] + pp
        weights_inh.append(np.concatenate((idx1, idx2)))

    spiking_nzw_exc = weights_exc.copy()
    spiking_nzw_inh = weights_inh.copy()

    for t in tqdm(range(1, T), desc=desc):
        # print(f"\r{desc} {t}/{T}", end="")
        # update membrane potential
        mp[t] = update_membrane_potential(
            mp=mp[t - 1],
            weights=weights[:, st:pn],
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
            spiking_nzw_exc,
            spiking_nzw_inh,
        ) = update_spikes(
            st=st,
            pn=pn,
            mp=mp[t],
            dt=dt,
            spikes=spikes[t],
            spike_times=spike_times,
            weights_exc=weights_exc,
            weights_inh=weights_inh,
            spiking_nzw_exc=spiking_nzw_exc,
            spiking_nzw_inh=spiking_nzw_inh,
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
            weights[:pn, :pn], sleep_now_inh, sleep_now_exc = update_weights(
                weights=weights[:pn, :pn],
                spike_times=spike_times,  # maybe need differentiate between exc and inh
                min_weight_exc=min_weight_exc,
                max_weight_inh=max_weight_inh,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                check_sleep_interval=check_sleep_interval,
                spikes=spikes[t],  # maybe wasteful
                sleep=sleep,  # maybe wasteful
                sleep_now_inh=sleep_now_inh,
                sleep_now_exc=sleep_now_exc,
                nz_cols_exc=nz_cols_exc,  # maybe wasteful
                nz_cols_inh=nz_cols_inh,  # maybe wasteful
                nz_rows_exc=nz_rows_exc,  # maybe wasteful
                nz_rows_inh=nz_rows_inh,  # maybe wasteful
                t=t,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                spiking_nzw_exc=spiking_nzw_exc,
                spiking_nzw_inh=spiking_nzw_inh,
            )

        # save weights for plotting
        if t % interval == 0 and t != T:
            weights_4_plotting_exc[t // interval] = weights[idx_exc, st:pn]
            weights_4_plotting_inh[t // interval] = weights[idx_inh, st:ex]
            pre_trace_4_plot[t // interval] = pre_trace
            post_trace_4_plot[t // interval] = post_trace

        # remove training data during sleep
        if sleep:
            if (sleep_now_exc or sleep_now_inh) and t < T - 1:
                spikes[t + 1, :N_x] = 0
                spike_labels[t] = -2

    return (
        weights,
        spikes,
        pre_trace,
        post_trace,
        mp,
        weights_4_plotting_exc,
        weights_4_plotting_inh,
        pre_trace_4_plot,
        post_trace_4_plot,
        spike_threshold,
        weight_mask,
        max_sum_inh,
        max_sum_exc,
    )
