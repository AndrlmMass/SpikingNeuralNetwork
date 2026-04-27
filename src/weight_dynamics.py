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
    # print("before normalization", np.std(weights[row_start:row_end, col_start:col_end]))
    W = weights[row_start:row_end, col_start:col_end]
    current_sums = np.sum(np.abs(W), axis=0)

    # Scale factor: min(1.0, initial_sum / current_sum) to cap at initial
    scale = np.where(current_sums > 1e-8, initial_sums / current_sums, 1.0)

    # Apply scaling column-wise
    weights[row_start:row_end, col_start:col_end] *= scale[np.newaxis, :]

    # print("after normalization", np.std(weights[row_start:row_end, col_start:col_end]))
    return weights


@njit(cache=True, parallel=True)
def go_sleep(
    weights,
    w_target,
    use_post_targets,
    use_layer_targets,
    use_static_targets,
    weight_decay_rate,
    nz_rows,
    nz_cols,
):
    """
    Compiles normalization function as
      1) Static targets
      2) Layer-weight sum targets
      3) Targets per post-synapse sum
    These are run for 1/x, where x is the predetermined sleep duration
    """

    """The static method has the same target throughout training"""
    if use_static_targets:
        # Apply decay to columns [N_x, N_post] (excitatory weights)
        for i in range(nz_rows.size):
            w = weights[nz_rows[i], nz_cols[i]]

            # Apply decay to absolute values
            new_abs_weight = w * (w_target / w) ** weight_decay_rate

            # Apply the original sign
            weights[nz_rows[i], nz_cols[i]] = new_abs_weight

        return weights

    elif use_layer_targets:
        pass

    elif use_post_targets:
        pass

    else:
        raise ValueError(
            "use_post_targets, use_layer_targets and use_static_targets cannot all be False or True. Only one of these boolean variables can be true at runtime."
        )


@njit(parallel=True, cache=True)
def trace_STDP(
    learning_rate_exc,
    spike_trace,
    weights,
    N_x,
    spikes,
    nonzero_pre_idx,
    x_tar_se,
    x_tar_ex,
    track_weights,
    w_max,
    mu_weight,
):
    n_neurons = spike_trace.shape[0]

    if track_weights:
        # Serial path for debugging — no prange
        list_x_pre = 0
        first_term = 0
        delta_w_sum = 0
        count = 0
        for i in range(N_x, n_neurons):
            if spikes[i] == 1:
                pre_indices = nonzero_pre_idx[i - N_x]
                for j in pre_indices:
                    if j < N_x:
                        first_trm = spike_trace[j] - x_tar_se
                    else:
                        first_trm = spike_trace[j] - x_tar_ex

                    second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight

                    delta_weight = learning_rate_exc * first_trm * second_trm

                    weights[j, i] += delta_weight
                    list_x_pre += spike_trace[j]
                    first_term += first_trm
                    delta_w_sum += delta_weight
                    count += 1
        return (
            weights,
            list_x_pre / (count + 1e-5),
            first_term / (count + 1e-5),
            delta_w_sum / (count + 1e-5),
        )
    else:
        # Parallel path for production
        for i in prange(N_x, n_neurons):
            pre_indices = nonzero_pre_idx[i - N_x]
            for j in pre_indices:
                if j < N_x:
                    first_trm = spike_trace[j] - x_tar_se
                else:
                    first_trm = spike_trace[j] - x_tar_ex

                second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight

                delta_weight = learning_rate_exc * first_trm * second_trm
                weights[j, i] += delta_weight

        return weights, 0, 0, 0
