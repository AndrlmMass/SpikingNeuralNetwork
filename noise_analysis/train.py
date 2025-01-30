from numba import njit
from tqdm import tqdm
import pickle as pkl
import numpy as np
import os


@njit
def sleep_func(
    weights,  # shape = (N_pre, N_post)
    max_sum_weights,  # rename to avoid confusion w/ Python 'sleep'
    sleep_now,
    N_inh,
    w_target_exc,
    w_target_inh,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    baseline_weight_sum,
):
    """
    Numba-nopython-compatible version that:
     1) Computes sum of |weights|.
     2) If sum > max_sum_weights => sleep_now = True.
     3) If sleep_flag and sleep_now => apply decay to excitatory (rows [0..end_exc))
        and inhibitory (rows [end_exc..N_pre)) synapses.
     4) If the new sum of |weights| <= baseline_weight_sum => sleep_now = False.
    """

    # ------------------------------------------------
    # 1) Sum of absolute weights
    # ------------------------------------------------
    sum_weights = 0.0
    N_pre, N_post = weights.shape
    for i in range(N_pre):
        for j in range(N_post):
            sum_weights += abs(weights[i, j])

    # ------------------------------------------------
    # 2) Check if we exceed max_sum_weights
    # ------------------------------------------------
    if sum_weights > max_sum_weights:
        sleep_now = True

    # ------------------------------------------------
    # 3) If sleeping is active, apply decay
    # ------------------------------------------------
    if sleep_now:
        # Number of excitatory rows:
        end_exc = N_pre - N_inh

        # --- Excitatory portion: rows [0..end_exc) ---
        for i in range(end_exc):
            for j in range(N_post):
                w_ij = weights[i, j]
                if w_ij != 0.0:
                    # Decay = w_target_exc * (w_ij / w_target_exc)^weight_decay_rate_exc
                    decay_exc = w_target_exc * (
                        (w_ij / w_target_exc) ** weight_decay_rate_exc
                    )
                    weights[i, j] = w_ij - decay_exc

        # --- Inhibitory portion: rows [end_exc..N_pre) ---
        for i in range(end_exc, N_pre):
            for j in range(N_post):
                w_ij = weights[i, j]
                if w_ij != 0.0:
                    # Decay = w_target_inh * (w_ij / w_target_inh)^weight_decay_rate_inh
                    decay_inh = w_target_inh * (
                        (w_ij / w_target_inh) ** weight_decay_rate_inh
                    )
                    weights[i, j] = w_ij + decay_inh

        # Re-check sum of absolute weights
        sum_weights2 = 0.0
        for i in range(N_pre):
            for j in range(N_post):
                sum_weights2 += abs(weights[i, j])

        # If weights decayed below baseline => stop sleeping
        if sum_weights2 <= baseline_weight_sum:
            sleep_now = False

    return weights, sleep_now


@njit
def trace_STDP(
    spikes: np.ndarray,
    weights: np.ndarray,
    delta_w: np.ndarray,
    pre_trace: np.ndarray,
    post_trace: np.ndarray,
    dt: float,
    N_x: int,
    A_plus: float,
    A_minus: float,
    N_inh: int,
    tau_pre_trace_exc: float,
    tau_pre_trace_inh: float,
    tau_post_trace_exc: float,
    tau_post_trace_inh: float,
):
    """
    Optimized Numba-friendly STDP update.
    """
    # Precompute constants
    A_plus_dt = A_plus * dt
    A_minus_dt = A_minus * dt
    exp_pre_exc = np.exp(-dt / tau_pre_trace_exc)
    exp_pre_inh = np.exp(-dt / tau_pre_trace_inh)
    exp_post_exc = np.exp(-dt / tau_post_trace_exc)
    exp_post_inh = np.exp(-dt / tau_post_trace_inh)

    # Identify spike indices
    spike_idx = []
    for i in range(spikes.size):
        if spikes[i] == 1:
            spike_idx.append(i)
    spike_count = len(spike_idx)

    # Separate post and pre spikes
    post_spikes = []
    pre_spikes = []
    for idx in range(spike_count):
        i = spike_idx[idx]
        if i >= N_x:
            post_spikes.append(i)
        pre_spikes.append(i)

    # Weight updates
    for i_post in post_spikes:
        post_tr = post_trace[i_post - N_x]
        for j in pre_spikes:
            # Potentiation
            delta_w[j, i_post] += A_plus_dt * pre_trace[j]

            # Depression
            if weights[j, i_post] != 0.0:
                delta_w[j, i_post] -= A_minus_dt * post_tr

    # Flip sign for inhibitory weights
    start_inh = weights.shape[0] - N_inh
    for i in range(start_inh, weights.shape[0]):
        for j in range(weights.shape[1]):
            delta_w[i, j] = -delta_w[i, j]

    # Update weights
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] += delta_w[i, j]

    # Decay the eligibility traces
    for i in range(pre_trace.size):
        if i < pre_trace.size - N_inh:
            pre_trace[i] *= exp_pre_exc
        else:
            pre_trace[i] *= exp_pre_inh

    for i in range(post_trace.size):
        if i < post_trace.size - N_inh:
            post_trace[i] *= exp_post_exc
        else:
            post_trace[i] *= exp_post_inh

    # Increase pre-trace for all spiking neurons
    for idx in spike_idx:
        pre_trace[idx] += dt

    # Increase post-trace for spiking post-neurons
    for i_post in post_spikes:
        post_spike_idx = i_post - N_x
        post_trace[post_spike_idx] += dt

    return post_trace, pre_trace, weights


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
    max_sum_weights,
    clip_exc_weights,
    clip_inh_weights,
    vectorized_trace,
    baseline_weight_sum,
    check_sleep_interval,
    spikes,
    N_inh,
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
        if t % check_sleep_interval == 0:
            weights, sleep_now = sleep_func(
                weights,
                max_sum_weights,
                sleep_now,
                N_inh,
                w_target_exc,
                w_target_inh,
                weight_decay_rate_exc,
                weight_decay_rate_inh,
                baseline_weight_sum,
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

    if trace_update:
        post_trace, pre_trace, weights = trace_STDP(
            spikes=spikes,
            weights=weights,
            delta_w=delta_w,
            pre_trace=pre_trace,
            post_trace=post_trace,
            dt=dt,
            N_x=N_x,
            A_plus=A_plus,
            A_minus=A_minus,
            N_inh=N_inh,
            tau_pre_trace_exc=tau_pre_trace_exc,
            tau_pre_trace_inh=tau_pre_trace_inh,
            tau_post_trace_exc=tau_post_trace_exc,
            tau_post_trace_inh=tau_post_trace_inh,
        )

    if vectorized_trace:
        """
        if used in future, select which posts and pre neurons
        spiked to significantly improve inference time
        """

        # Suppose:
        #   weights.shape = (N_pre, N_post)
        #   spikes is a 1D array with 1's at indices of spiking neurons
        #   pre_spikes and post_spikes are arrays of indices for spiking pre/post neurons
        #   pre_trace and post_trace are the STDP traces for pre/post neurons
        #   A_plus, A_minus, dt are scalars
        #   N_x is the boundary between pre- and post-neuron indices in 'spikes
        spike_idx = np.where(spikes == 1)[0]
        pre_spikes = spike_idx
        post_spikes = spike_idx[spike_idx > N_x]

        #  - nz_rows: all pre-neuron indices with nonzero weights
        #  - nz_cols: the corresponding post-neuron indices

        # 2. Filter for post-synapses that actually spiked.
        #    We want columns (nz_cols) that appear in 'post_spikes'.
        post_mask = np.in1d(nz_cols, post_spikes)
        post_rows = nz_rows[post_mask]
        post_cols = nz_cols[post_mask]

        # 3. LTP step:
        #    For every (pre, post) that is nonzero and the post spiked:
        #      delta_w[pre, post] += A_plus * pre_trace[pre] * dt
        delta_w[post_rows, post_cols] += A_plus * pre_trace[post_rows] * dt

        # 4. LTD step:
        #    Among those same (pre, post) pairs, we only subtract if the pre also spiked:
        #    i.e. if pre in pre_spikes. Then we subtract A_minus * post_trace[post-N_x].
        pre_mask = np.in1d(post_rows, pre_spikes)
        ltd_rows = post_rows[pre_mask]
        ltd_cols = post_cols[pre_mask]

        #    Because these post indices are in [N_x..], we index post_trace properly:
        delta_w[ltd_rows, ltd_cols] -= A_minus * post_trace[ltd_cols - N_x] * dt

        # delta_w now holds your STDP weight updates in a fully vectorized way.

        # Flip sign of inhibitory weights change
        delta_w[-N_inh:] *= -1
        weights += delta_w

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

        # increase trace if neuron spiked
        pre_trace += spikes * dt
        post_trace += spikes[N_x:] * dt

    # add noise to weights if desired
    # if noisy_weights:
    # delta_weight_noise = np.random.normal(
    # loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
    # )
    # else:
    # delta_weight_noise = np.zeros(weights.shape)

    # Update weights
    # print(
    #     f"\rexc weight change:{np.round(np.mean(delta_weights_exc),6)}, inh weight change:{np.round(np.mean(delta_weights_inh),6)}",
    #     end="",
    # )

    # if clip_exc_weights:
    #     weights[:-N_inh] = np.clip(
    #         weights[:-N_inh], a_min=min_weight_exc, a_max=max_weight_exc
    #     )

    # if clip_inh_weights:
    #     weights[-N_inh:] = np.clip(
    #         weights[-N_inh:], a_min=min_weight_inh, a_max=max_weight_inh
    #     )

    # weights += delta_weight_noise
    # weights[non_weight_mask] = 0

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
    weight_mask = weights != 0
    non_weight_mask = weights == 0

    baseline_weight_sum = np.sum(np.abs(weights))
    max_sum_weights = baseline_weight_sum * alpha
    delta_w = np.zeros(shape=weights.shape)

    nz_rows, nz_cols = np.nonzero(weights)
    sleep_now = False

    print("Training network...")
    for t in tqdm(range(1, T)):

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
            spikes=spikes[t - 1],
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
                max_sum_weights=max_sum_weights,
                non_weight_mask=non_weight_mask,
                baseline_weight_sum=baseline_weight_sum,
                check_sleep_interval=check_sleep_interval,
                sleep=sleep,
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
        if t % interval == 0:
            weights_4_plotting[t // interval] = weights[N_x:]

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
        spike_threshold,
    )
