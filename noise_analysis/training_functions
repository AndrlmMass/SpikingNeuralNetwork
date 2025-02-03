import numpy as np
from numba import njit


@njit
def sleep_func(
    weights,  # shape = (N_pre, N_post)
    max_sum_weights,
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
    else:
        sleep_now = False
        return weights, sleep_now

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
                    # Calculate the ratio for decay
                    ratio = w_ij / w_target_exc

                    # Compute decay based on the ratio and decay rate
                    weights[i, j] = w_target_exc * (ratio**weight_decay_rate_exc)

        # --- Inhibitory portion: rows [end_exc..N_pre) ---
        for i in range(end_exc, N_pre):
            for j in range(N_post):
                w_ij = weights[i, j]
                if w_ij != 0.0:
                    # Calculate the ratio for decay
                    ratio = abs(w_ij / w_target_inh)

                    # Compute decay based on the ratio and decay rate
                    weights[i, j] = w_target_inh * (ratio**weight_decay_rate_inh)

        # Re-check sum of absolute weights
        sum_weights2 = 0.0
        for i in range(N_pre):
            for j in range(N_post):
                sum_weights2 += abs(weights[i, j])

        # If weights decayed below baseline => stop sleeping
        if sum_weights2 <= baseline_weight_sum:
            sleep_now = False

    return weights, sleep_now


def vectorized_trace_func(
    spikes,
    N_x,
    nz_rows,
    nz_cols,
    delta_w,
    A_plus,
    dt,
    A_minus,
    pre_trace,
    post_trace,
    tau_pre_trace_exc,
    tau_pre_trace_inh,
    tau_post_trace_exc,
    tau_post_trace_inh,
    N_inh,
):
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

    return weights, pre_trace, post_trace


@njit
def trace_STDP(
    spikes: np.ndarray,
    weights: np.ndarray,
    pre_trace: np.ndarray,
    post_trace: np.ndarray,
    learning_rate_exc: float,
    learning_rate_inh: float,
    dt: float,
    N_x: int,
    A_plus: float,
    A_minus: float,
    N_inh: int,
    N_exc: int,
    N: int,
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
    pre_spikes_exc = []
    pre_spikes_inh = []
    N_inh_idx = N - N_inh
    for idx in range(spike_count):
        i = spike_idx[idx]
        if i >= N_x:
            post_spikes.append(i)
        if i <= N_inh_idx:
            pre_spikes_exc.append(i)
        else:
            pre_spikes_inh.append(i)

    # Weight updates
    for i_post in post_spikes:
        post_tr = post_trace[i_post - N_x]
        for j in pre_spikes_exc:
            if weights[j, i_post] != 0.0:
                # potentiation
                weights[j, i_post] += A_plus_dt * pre_trace[j] * learning_rate_exc
                # depression
                weights[j, i_post] -= A_minus_dt * post_tr * learning_rate_exc

        for j in pre_spikes_inh:
            if weights[j, i_post] != 0.0:
                # potentiation
                weights[j, i_post] -= A_plus_dt * pre_trace[j] * learning_rate_inh
                # depression
                weights[j, i_post] += A_minus_dt * post_tr * learning_rate_inh

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


def spike_timing(
    spike_times,
    spike_idx,
    tau_LTP,
    tau_LTD,
    learning_rate_exc,
    learning_rate_inh,
    N_inh,
    weights,
):
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
    weights[:-N_inh] += (
        learning_rate_exc * stdp_update[:-N_inh]
    )  # For excitatory connections
    weights[-N_inh:] -= (
        learning_rate_inh * stdp_update[-N_inh:]
    )  # For inhibitory connections

    return weights
