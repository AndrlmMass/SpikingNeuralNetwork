import math
import numpy as np
from numba import njit, prange


@njit
def sleep_func(
    weights,  # shape = (N_pre, N_post)
    max_sum,
    max_sum_exc,
    max_sum_inh,
    sleep_now_inh,
    sleep_now_exc,
    w_target_exc,
    w_target_inh,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    baseline_sum_exc,
    baseline_sum_inh,
    sleep_synchronized,
    nz_rows,
    nz_cols,
    baseline_sum,
    nz_rows_exc,
    nz_rows_inh,
    nz_cols_exc,
    nz_cols_inh,
):
    """
    Optimized, vectorized version:
      1) Computes sum of |weights| using slices.
      2) Checks if the sums exceed max values and sets sleep flags.
      3) If sleeping is active, applies decay in a vectorized way.
      4) Recomputes the sum to stop sleeping if below baseline.
    """

    # Instead of nested loops, use slicing for excitatory/inhibitory sums.
    # According to your code, columns [0, N_exc+N_x) are excitatory,
    # and columns [N_exc+N_x, N_post) are inhibitory.

    if not sleep_now_inh or not sleep_now_exc:
        if sleep_synchronized:
            sum_weights = 0
            for i in range(nz_rows.size):
                sum_weights += weights[nz_rows[i], nz_cols[i]]
            # print("1", sum_weights, max_sum)
            if sum_weights > max_sum:
                sleep_now_exc = True
                sleep_now_inh = True

        else:
            sum_weights_exc = 0
            for i in range(nz_rows_exc.size):
                sum_weights_exc += weights[nz_rows_exc[i], nz_cols_exc[i]]
            if sum_weights_exc > max_sum_exc:
                sleep_now_exc = True

            sum_weights_inh = 0
            for i in range(nz_rows_inh.size):
                sum_weights_inh += np.abs(weights[nz_rows_inh[i], nz_cols_inh[i]])
            if sum_weights_inh > max_sum_inh:
                sleep_now_inh = True

    # --- Decay excitatory weights --
    if sleep_now_exc:
        # Apply decay to columns [N_x, N_post] (excitatory weights)
        for i in range(nz_rows_exc.size):
            weights[nz_rows_exc[i], nz_cols_exc[i]] = w_target_exc * (
                (weights[nz_rows_exc[i], nz_cols_exc[i]] / w_target_exc)
                ** weight_decay_rate_exc
            )

        if not sleep_synchronized:
            # Recompute the excitatory sum (only for the decayed submatrix)
            sum_weights_exc2 = 0
            for i in range(nz_rows_exc.size):
                sum_weights_exc2 += np.abs(weights[nz_rows_exc[i], nz_cols_exc[i]])
            if sum_weights_exc2 <= baseline_sum_exc:
                sleep_now_exc = False

    # --- Decay inhibitory weights ---
    if sleep_now_inh:
        # print("decaying inh")
        # Apply decay to columns [N_x, N_post] (excitatory weights)
        for i in range(nz_rows_inh.size):
            weights[nz_rows_inh[i], nz_cols_inh[i]] = w_target_inh * (
                (weights[nz_rows_inh[i], nz_cols_inh[i]] / w_target_inh)
                ** weight_decay_rate_inh
            )
        if not sleep_synchronized:
            # Recompute the excitatory sum (only for the decayed submatrix)
            sum_weights_inh2 = 0
            for i in range(nz_rows_inh.size):
                sum_weights_inh2 += np.abs(weights[nz_rows_inh[i], nz_cols_inh[i]])
            if sum_weights_inh2 <= baseline_sum_inh:
                sleep_now_inh = False

    if sleep_now_inh and sleep_now_exc and sleep_synchronized:
        # Recompute the excitatory sum (only for the decayed submatrix)
        sum_weights2 = 0
        for i in range(nz_rows.size):
            sum_weights2 += np.abs(weights[nz_rows[i], nz_cols[i]])
        # print("2", sum_weights2, baseline_sum)
        if sum_weights2 <= baseline_sum:
            sleep_now_inh = False
            sleep_now_exc = False

    if np.isnan(weights).any():
        raise SystemError("Matrix contains NaN values.")

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
    post_spikes = []
    for i in range(spikes.size):
        if spikes[i] == 1:
            if i >= N_x:
                post_spikes.append(i)
            spike_idx.append(i)

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


@njit(parallel=True)
def spike_timing(
    spike_times,  # Array of spike times
    tau_LTP,
    tau_LTD,
    learning_rate_exc,
    learning_rate_inh,
    N_inh,
    weights,  # Weight matrix (pre x post)
    N_x,  # Starting index for postsynaptic neurons
    spikes,  # Binary spike indicator array
    nonzero_pre_idx,  # Typed list: for each post neuron, an array of nonzero pre indices
):
    n_neurons = spike_times.shape[0]

    # Loop over postsynaptic neurons, parallelized.
    for i in prange(N_x, n_neurons):
        t_post = spike_times[i]
        # Retrieve the pre-synaptic indices that have a nonzero connection to neuron i.
        # Note: We assume the nonzero_pre_idx list is indexed relative to the postsynaptic
        # neurons starting at N_x (i.e., index 0 in the list corresponds to neuron N_x)
        pre_indices = nonzero_pre_idx[i - N_x]

        # Iterate only over connections that exist.
        for j in pre_indices:
            # Skip if the presynaptic neuron did not spike
            if spikes[j] == 0 and spikes[i] == 0:
                continue

            t_pre = spike_times[j]
            dt = t_post - t_pre

            # Determine if the connection is excitatory or inhibitory.
            if j < (n_neurons - N_inh):  # excitatory pre–synaptic neuron
                if dt >= 0:
                    weights[j, i] += math.exp(-dt / tau_LTP) * learning_rate_exc
                else:
                    weights[j, i] -= math.exp(dt / tau_LTD) * learning_rate_exc
            else:  # inhibitory pre–synaptic neuron
                if dt >= 0:
                    weights[j, i] -= math.exp(-dt / tau_LTP) * learning_rate_inh
                else:
                    weights[j, i] += math.exp(dt / tau_LTD) * learning_rate_inh

    return weights
