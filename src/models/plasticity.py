import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
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
    spike_times,  # Array of spike times
    A_plus, 
    A_minus,
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
                    weights[j, i] += math.exp(-dt / tau_LTP) * learning_rate_exc*A_plus
                else:
                    weights[j, i] -= math.exp(dt / tau_LTD) * learning_rate_exc*A_minus
            else:  # inhibitory pre–synaptic neuron
                if dt >= 0:
                    weights[j, i] -= math.exp(-dt / tau_LTP) * learning_rate_inh*A_plus
                else:
                    weights[j, i] += math.exp(dt / tau_LTD) * learning_rate_inh*A_minus

    return weights
