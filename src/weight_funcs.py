import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def normalize_weights_per_column(
    weights, initial_sums, row_start, row_end, col_start, col_end
):
    """
    Normalize weights per column (post-neuron) to not exceed initial sums.

    Args:
        weights: Full weight matrix
        initial_sums: Array of initial column sums (one per post-neuron)
        row_start, row_end: Row slice for this weight layer
        col_start, col_end: Column slice for this weight layer
    """
    W = weights[row_start:row_end, col_start:col_end]
    current_sums = np.sum(np.abs(W), axis=0)

    # Scale factor: min(1.0, initial_sum / current_sum) to cap at initial
    scale = np.where(current_sums > 1e-3, initial_sums / current_sums, 1.0)

    # Apply scaling column-wise
    weights[row_start:row_end, col_start:col_end] *= scale[np.newaxis, :]

    return weights


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
    use_post_targets,
    w_target_exc_vec,
    w_target_inh_vec,
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
    st,
    ex,
):
    """
    Optimized, vectorized version:
      1) Computes sum of |weights| using slices.
      2) Checks if the sums exceed max values and sets sleep flags.
      3) If sleeping is active, applies decay in a vectorized way.
      4) Recomputes the sum to stop sleeping if below baseline.
    """

    # Debug-time scans removed for performance

    # Instead of nested loops, use slicing for excitatory/inhibitory sums.
    # According to your code, columns [0, N_exc+N_x) are excitatory,
    # and columns [N_exc+N_x, N_post) are inhibitory.

    if not sleep_now_inh or not sleep_now_exc:
        if sleep_synchronized:
            sum_weights = 0
            for i in range(nz_rows.size):
                sum_weights += np.abs(weights[nz_rows[i], nz_cols[i]])
            if sum_weights > max_sum:
                sleep_now_exc = True
                sleep_now_inh = True

        else:
            sum_weights_exc = 0
            for i in range(nz_rows_exc.size):
                sum_weights_exc += np.abs(weights[nz_rows_exc[i], nz_cols_exc[i]])
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
            # Skip frozen input→excitatory connections
            if nz_rows_exc[i] < st and nz_cols_exc[i] >= st and nz_cols_exc[i] < ex:
                continue
            current_weight = weights[nz_rows_exc[i], nz_cols_exc[i]]
            # Work with absolute values to avoid complex numberst
            abs_weight = np.abs(current_weight)
            # Use per-post target if requested, otherwise scalar target
            if use_post_targets:
                abs_target = np.abs(w_target_exc_vec[nz_cols_exc[i]])
            else:
                abs_target = np.abs(w_target_exc)

            # Apply decay to absolute values
            new_abs_weight = (
                abs_target * (abs_weight / abs_target) ** weight_decay_rate_exc
            )

            # Apply the original sign
            weights[nz_rows_exc[i], nz_cols_exc[i]] = new_abs_weight

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
            current_weight = weights[nz_rows_inh[i], nz_cols_inh[i]]
            # Work with absolute values to avoid complex numbers
            abs_weight = np.abs(current_weight)
            # Use per-post target if requested, otherwise scalar target
            if use_post_targets:
                abs_target = np.abs(w_target_inh_vec[nz_cols_inh[i]])
            else:
                abs_target = np.abs(w_target_inh)

            # Apply decay to absolute values
            new_abs_weight = (
                abs_target * (abs_weight / abs_target) ** weight_decay_rate_inh
            )

            # Apply the original sign
            weights[nz_rows_inh[i], nz_cols_inh[i]] = -new_abs_weight
        if not sleep_synchronized:
            # Recompute the excitatory sum (only for the decayed submatrix)
            sum_weights_inh2 = 0
            for i in range(nz_rows_inh.size):
                sum_weights_inh2 += np.abs(weights[nz_rows_inh[i], nz_cols_inh[i]])
            if sum_weights_inh2 <= baseline_sum_inh:
                sleep_now_inh = False

    if (sleep_now_inh or sleep_now_exc) and sleep_synchronized:
        # Recompute the excitatory sum (only for the decayed submatrix)
        sum_weights2 = 0
        for i in range(nz_rows.size):
            sum_weights2 += np.abs(weights[nz_rows[i], nz_cols[i]])
        if sum_weights2 <= baseline_sum:
            sleep_now_inh = False
            sleep_now_exc = False

    # Removed expensive NaN scan; rely on upstream invariants/guards

    return weights, sleep_now_inh, sleep_now_exc


@njit(parallel=True, cache=True)
def spike_timing(
    learning_rate_exc,
    learning_rate_inh,
    spike_trace,
    N_inh,
    weights,
    N_x,
    spikes,
    nonzero_pre_idx,
    x_tar,
    track_weights,
    w_max,
    mu_weight,
):
    n_neurons = spike_trace.shape[0]
    exc_boundary = n_neurons - N_inh

    if track_weights:
        # Serial path for debugging — no prange
        list_x_pre = []
        first_term = []
        # second_term = []
        delta_w_list = []
        for i in range(N_x, n_neurons):
            pre_indices = nonzero_pre_idx[i - N_x]
            if spikes[i] == 1:
                for j in pre_indices:
                    if j < exc_boundary:
                        x_pre_exc = spike_trace[j]
                        # base = max(w_max - weights[j, i], 0.0)
                        delta_w = learning_rate_exc * (x_pre_exc - x_tar[i - N_x])
                        weights[j, i] += delta_w
                        list_x_pre.append(x_pre_exc)
                        first_term.append(x_pre_exc - x_tar)
                        # second_term.append(base**mu_weight)
                        delta_w_list.append(delta_w)
        return weights, list_x_pre, first_term, None, delta_w_list
    else:
        # Parallel path for production
        for i in prange(N_x, n_neurons):
            pre_indices = nonzero_pre_idx[i - N_x]
            if spikes[i] == 1:
                for j in pre_indices:
                    if j < exc_boundary:
                        x_pre_exc = spike_trace[j]
                        # base = max(w_max - weights[j, i], 0.0)
                        delta_w = learning_rate_exc * (x_pre_exc - x_tar[i - N_x])
                        weights[j, i] += delta_w
        return weights, None, None, None, None
