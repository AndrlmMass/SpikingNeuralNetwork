import numpy as np
from numba import njit


@njit
def sleep_func(
    weights,  # shape = (N_pre, N_post)
    max_sum_exc,
    max_sum_inh,
    sleep_now_inh,
    sleep_now_exc,
    N_inh,
    N_exc,
    N_x,
    w_target_exc,
    w_target_inh,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    baseline_sum_exc,
    baseline_sum_inh,
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
    sum_weights_exc = 0.0
    sum_weights_inh = 0.0
    N_pre, N_post = weights.shape
    for i in range(N_pre):
        for j in range(N_post):
            delta_weights = abs(weights[i, j])
            if delta_weights != None:
                if j >= N_exc + N_x:
                    sum_weights_inh += delta_weights
                else:
                    sum_weights_exc += delta_weights

    # ------------------------------------------------#
    # 2) Check if we exceed max_sum_weights           #
    # ------------------------------------------------#
    if sum_weights_exc > max_sum_exc:
        sleep_now_exc = True
        print("sleepy exc!")
    if sum_weights_inh > max_sum_inh:
        sleep_now_inh = True
        print("sleepy inh!")
    if not sleep_now_inh and not sleep_now_exc:
        return weights, sleep_now_inh, sleep_now_exc

    # ------------------------------------------------
    # 3) If sleeping is active, apply decay
    # ------------------------------------------------
    if sleep_now_exc:
        # Number of excitatory rows:
        end_exc = N_pre - N_inh

        # --- Excitatory portion: rows [0..end_exc) ---
        for i in range(end_exc):
            for j in range(N_x, N_post):
                w_ij = weights[i, j]
                if w_ij != 0.0:
                    # Calculate the ratio for decay
                    ratio = w_ij / w_target_exc

                    # Compute decay based on the ratio and decay rate
                    weights[i, j] = w_target_exc * (ratio**weight_decay_rate_exc)

        # calculte the new sum of weights
        sum_weights_exc2 = 0.0
        N_pre, N_post = weights.shape
        for i in range(end_exc):
            for j in range(N_x, N_post):
                delta_weights = abs(weights[i, j])
                if delta_weights != None:
                    sum_weights_exc2 += delta_weights

        # If weights decayed below baseline => stop sleeping
        if sum_weights_exc2 <= baseline_sum_exc:
            sleep_now_exc = False

    if sleep_now_inh:
        # Number of excitatory rows:
        end_inh = N_pre - N_inh
        end_inh_post = N_post - N_inh

        # --- Inhibitory portion: rows [end_exc..N_pre) ---
        for i in range(end_inh, N_pre):
            for j in range(N_x, end_inh_post):
                w_ij = weights[i, j]
                if w_ij != 0.0:
                    # Calculate the ratio for decay
                    ratio = abs(w_ij / w_target_inh)

                    # Compute decay based on the ratio and decay rate
                    weights[i, j] = w_target_inh * (ratio**weight_decay_rate_inh)

        sum_weights_inh2 = 0.0
        N_pre, N_post = weights.shape
        for i in range(end_inh, N_pre):
            for j in range(N_x, end_inh_post):
                delta_weights = abs(weights[i, j])
                if delta_weights != None:
                    sum_weights_inh2 += delta_weights

        # If weights decayed below baseline => stop sleeping
        if sum_weights_inh2 <= baseline_sum_inh:
            sleep_now_inh = False

    return weights, sleep_now_inh, sleep_now_exc


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
    nonzero_pre_idx,  # This is a typed List of arrays
):
    # Precompute some constants
    A_plus_dt = A_plus * dt
    A_minus_dt = A_minus * dt
    exp_pre_exc = np.exp(-dt / tau_pre_trace_exc)
    exp_pre_inh = np.exp(-dt / tau_pre_trace_inh)
    exp_post_exc = np.exp(-dt / tau_post_trace_exc)
    exp_post_inh = np.exp(-dt / tau_post_trace_inh)

    # Determine which neurons spiked
    # (This loop is similar to your original code.)
    spike_idx = []
    for i in range(spikes.size):
        if spikes[i] == 1:
            spike_idx.append(i)

    # Separate post-spikes (i >= N_x)
    post_spikes = []
    for idx in spike_idx:
        if idx >= N_x:
            post_spikes.append(idx)

    # For each post–neuron that spiked, update only its incoming weights.
    # Here we assume that nonzero_pre_idx[post_col] corresponds to the post–neuron
    # with index post = post_col + N_x (since your weights are in columns N_x: ).
    for i_post in post_spikes:
        # Determine the column index in the weight matrix and in the connectivity list
        post_col = i_post - N_x
        pre_indices = nonzero_pre_idx[post_col]
        # Now loop over only those pre–neurons that have a connection to i_post
        for j in pre_indices:
            # Depending on whether pre neuron j is excitatory or inhibitory,
            # update the weight accordingly. (Your original code uses an index check.)
            if j < (N - N_inh):  # excitatory
                weights[j, i_post] += A_plus_dt * pre_trace[j] * learning_rate_exc
                weights[j, i_post] -= (
                    A_minus_dt * post_trace[post_col] * learning_rate_exc
                )
            else:  # inhibitory
                weights[j, i_post] -= A_plus_dt * pre_trace[j] * learning_rate_inh
                weights[j, i_post] += (
                    A_minus_dt * post_trace[post_col] * learning_rate_inh
                )

    # Now decay the eligibility traces as before
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

    # Increase traces for spiking neurons.
    for idx in spike_idx:
        pre_trace[idx] += dt
    for i_post in post_spikes:
        post_trace[i_post - N_x] += dt

    return post_trace, pre_trace, weights


# @njit
def spike_timing(
    spike_times,
    tau_LTP,
    tau_LTD,
    learning_rate_exc,
    learning_rate_inh,
    N_inh,
    weights,
    N_x,
):

    pre_spikes = []
    for i in spike_times:
        if i == 0:
            pre_spikes.append(i)

    # Separate post-spikes (i >= N_x)
    post_spikes = []
    for i in spike_times:
        if i >= N_x:
            if i == 0:
                post_spikes.append(i)

    # Compute the update using explicit loops.
    exc_pre_spikes = pre_spikes[:-N_inh]
    inh_pre_spikes = pre_spikes[-N_inh:]
    for j_ in range(len(exc_pre_spikes)):
        for i_ in range(len(post_spikes)):
            j = pre_spikes[j_]
            i = post_spikes[i_]
            dt = spike_times[i] - spike_times[j]
            if dt >= 0:
                weights[j, i] += np.exp(-dt / tau_LTP) * learning_rate_exc
            else:
                weights[j, i] -= np.exp(dt / tau_LTD) * learning_rate_inh

    for j_ in range(len(inh_pre_spikes)):
        for i_ in range(len(post_spikes)):
            j = pre_spikes[j_]
            i = post_spikes[i_]
            dt = spike_times[i] - spike_times[j]
            if dt >= 0:
                weights[j, i] -= np.exp(-dt / tau_LTP) * learning_rate_exc
            else:
                weights[j, i] += np.exp(dt / tau_LTD) * learning_rate_inh

    return weights
