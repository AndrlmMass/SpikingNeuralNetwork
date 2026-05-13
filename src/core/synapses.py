import numpy as np
from numba import njit
from dataclasses import dataclass


@njit(cache=True)
def clip_weights(
    weights,
    nz_cols,
    nz_rows,
    min_weight,
    max_weight,
):
    for i_ in range(nz_rows.shape[0]):
        i, j = nz_rows[i_], nz_cols[i_]
        if weights[i, j] < min_weight:
            weights[i, j] = min_weight
        elif weights[i, j] > max_weight:
            weights[i, j] = max_weight
    return weights


@njit(parallel=False, cache=True)
def trace_STDP(
    learning_rate,
    spike_trace,
    weights,
    N_x,
    spikes,
    nonzero_pre_idx,
    x_tar_se,
    x_tar_ee,
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
                    if j == -1:
                        continue  # Skip padding
                    if j < N_x:
                        first_trm = spike_trace[j] - x_tar_se
                    else:
                        first_trm = spike_trace[j] - x_tar_ee

                    second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight

                    delta_weight = learning_rate * first_trm * second_trm

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
        for i in range(N_x, n_neurons):
            pre_indices = nonzero_pre_idx[i - N_x]
            for j in pre_indices:
                if j == -1:
                    continue  # Skip padding
                if j < N_x:
                    first_trm = spike_trace[j] - x_tar_se
                else:
                    first_trm = spike_trace[j] - x_tar_ee

                second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight

                delta_weight = learning_rate * first_trm * second_trm
                weights[j, i] += delta_weight
        return weights, 0, 0, 0


@dataclass
class Learner:
    learning_rate: float
    N_x: int
    nonzero_pre_idx: list
    w_max: float
    mu_weight: float

    def __post_init__(self):
        # In Learner.__post_init__, pad to fixed width:
        max_len = max(len(x) for x in self.nonzero_pre_idx)
        self.pre_idx_arr = np.full(
            (len(self.nonzero_pre_idx), max_len), -1, dtype=np.int64
        )
        for i, idx in enumerate(self.nonzero_pre_idx):
            self.pre_idx_arr[i, : len(idx)] = idx
        del self.nonzero_pre_idx  # remove list entirely

    def step(self, weights, spikes, spike_trace, x_tar_se, x_tar_ee, track_weights):
        return trace_STDP(
            self.learning_rate,
            spike_trace,
            weights,
            self.N_x,
            spikes,
            self.pre_idx_arr,
            x_tar_se,
            x_tar_ee,
            track_weights,
            self.w_max,
            self.mu_weight,
        )


@dataclass
class Clipper:
    nz_cols: list
    nz_rows: list
    min_weight_exc: float
    max_weight_exc: float

    def step(self, weights):
        return clip_weights(
            weights,
            self.nz_cols,
            self.nz_rows,
            self.min_weight_exc,
            self.max_weight_exc,
        )
