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
    '''
    Clips excitatory nonzero weights to min/max
    '''
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
    '''
    Hebbian weight update function. Using trace-based STDP to increase runtime. 
    '''
    # Fetch number of post-neurons with dynamic weights (input + excitatory layer)
    n_neurons = spike_trace.shape[0]

    # Dedicated leg for weight tracking to differentiated parallelized and non-parallelized runs
    if track_weights:
        # Pre-compute tracking variables
        list_x_pre = 0
        first_term = 0
        delta_w_sum = 0
        ltp_sum = 0.0
        ltd_sum = 0.0
        count = 0
        # Loop over post-neurons (only excitatory, not input neurons as they do not receive weights, just sends)
        for i in range(N_x, n_neurons):
            # Check if post-neuron has spiked
            if spikes[i] == 1:
                # Extract presynaptic indices for neuron i
                pre_indices = nonzero_pre_idx[i - N_x]
                # Loop over each presynaptic neuron index
                for j in pre_indices:
                    # Skip padding sentinel (-1). NOT a spike check: every
                    # structural pre of the spiking post is updated via its trace,
                    # so stale pre (trace < x_tar) are depressed (LTD).
                    if j == -1:
                        continue
                    # Compute difference between trace and target for stimulation (input) to excitatory (SE)
                    if j < N_x:
                        first_trm = spike_trace[j] - x_tar_se
                    # Same, but for excitatory to excitatory (EE)
                    else:
                        first_trm = spike_trace[j] - x_tar_ee
                    # Compute distance to max weight, scaled by mu
                    second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight
                    # Compute delta weight
                    delta_weight = learning_rate * first_trm * second_trm
                    # Update weights
                    weights[j, i] += delta_weight
                    # Update stats tracking terms
                    list_x_pre += spike_trace[j]
                    first_term += first_trm
                    delta_w_sum += delta_weight
                    # split the signed update into potentiation (LTP) vs
                    # depression (LTD) magnitudes to check the static-x_tar balance
                    if delta_weight >= 0.0:
                        ltp_sum += delta_weight
                    else:
                        ltd_sum += -delta_weight
                    count += 1
        return (
            weights,
            list_x_pre / (count + 1e-5),
            first_term / (count + 1e-5),
            delta_w_sum / (count + 1e-5),
            ltp_sum / (count + 1e-5),
            ltd_sum / (count + 1e-5),
        )
    else:
        # Loop over post-neurons (only excitatory, not input neurons as they do not receive weights, just sends)
        for i in range(N_x, n_neurons):
            # Check if post-neuron has spiked
            if spikes[i] == 1:
                # Extract presynaptic indices for neuron i
                pre_indices = nonzero_pre_idx[i - N_x]
                # Loop over each presynaptic neuron index
                for j in pre_indices:
                    # Skip padding sentinel (-1). NOT a spike check: every
                    # structural pre of the spiking post is updated via its trace,
                    # so stale pre (trace < x_tar) are depressed (LTD).
                    if j == -1:
                        continue
                    # Compute difference between trace and target for stimulation (input) to excitatory (SE)
                    if j < N_x:
                        first_trm = spike_trace[j] - x_tar_se
                    # Same, but for excitatory to excitatory (EE)
                    else:
                        first_trm = spike_trace[j] - x_tar_ee
                    # Compute distance to max weight, scaled by mu
                    second_trm = max(w_max - weights[j, i], 0.0) ** mu_weight
                    # Compute delta weight
                    delta_weight = learning_rate * first_trm * second_trm
                    # Update weights
                    weights[j, i] += delta_weight
        return weights, 0, 0, 0, 0, 0


@njit(parallel=False, cache=True)
def triplet_STDP(
    learning_rate,
    spike_trace,
    r2,
    o2,
    weights,
    N_x,
    spikes,
    nonzero_pre_idx,
    A2_plus,
    A3_plus,
    A2_minus,
    A3_minus,
    w_max,
    w_min,
    mu_weight,
):
    """Pfister & Gerstner (2006) triplet STDP with asymmetric soft weight bounds.

    Triggered at postsynaptic spikes (approximation of the full rule).
    LTP uses fast pre trace r1 and slow post trace o2.
    LTD uses fast post trace o1 and slow pre trace r2.
    Soft bounds: (w_max - w)^mu for LTP, (w - w_min)^mu for LTD.

    o1 is read as spike_trace[i] — the fast post trace value after one decay
    step since the triggering spike (we use spikes_prev, so the spike was in
    the previous timestep and the trace has decayed by one step).
    """
    n_neurons = spike_trace.shape[0]

    for i in range(N_x, n_neurons):
        if spikes[i] == 1:
            o1_i = spike_trace[i]          # fast post trace (decayed since spike)
            o2_i = o2[i - N_x]            # slow post trace

            pre_indices = nonzero_pre_idx[i - N_x]
            for j in pre_indices:
                if j == -1:
                    continue
                r1_j = spike_trace[j]      # fast pre trace
                r2_j = r2[j]              # slow pre trace

                ltp = r1_j * (A2_plus + A3_plus * o2_i)
                ltd = o1_i * (A2_minus + A3_minus * r2_j)

                ltp_bound = max(w_max - weights[j, i], 0.0) ** mu_weight
                ltd_bound = max(weights[j, i] - w_min, 0.0) ** mu_weight

                weights[j, i] += learning_rate * (ltp * ltp_bound - ltd * ltd_bound)

    return weights


@dataclass
class Learner:
    learning_rate: float
    N_x: int
    nonzero_pre_idx: list
    w_max: float
    mu_weight: float

    '''
    Initiates nonzero pre-synaptic indices before training.
    Then performs step-function to update weights.  
    '''

    def __post_init__(self):
        max_len = max(len(x) for x in self.nonzero_pre_idx)
        self.pre_idx_arr = np.full(
            (len(self.nonzero_pre_idx), max_len), -1, dtype=np.int64
        )
        for i, idx in enumerate(self.nonzero_pre_idx):
            self.pre_idx_arr[i, : len(idx)] = idx
        del self.nonzero_pre_idx

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
class TripletLearner:
    learning_rate: float
    N_x: int
    nonzero_pre_idx: list
    w_max: float
    w_min: float
    mu_weight: float
    A2_plus: float
    A3_plus: float
    A2_minus: float
    A3_minus: float

    def __post_init__(self):
        max_len = max(len(x) for x in self.nonzero_pre_idx)
        self.pre_idx_arr = np.full(
            (len(self.nonzero_pre_idx), max_len), -1, dtype=np.int64
        )
        for i, idx in enumerate(self.nonzero_pre_idx):
            self.pre_idx_arr[i, : len(idx)] = idx
        del self.nonzero_pre_idx

    def step(self, weights, spikes, spike_trace, r2, o2):
        return triplet_STDP(
            self.learning_rate,
            spike_trace,
            r2,
            o2,
            weights,
            self.N_x,
            spikes,
            self.pre_idx_arr,
            self.A2_plus,
            self.A3_plus,
            self.A2_minus,
            self.A3_minus,
            self.w_max,
            self.w_min,
            self.mu_weight,
        )


@dataclass
class Clipper:
    nz_cols: list
    nz_rows: list
    min_weight_exc: float
    max_weight_exc: float

    '''
    Object retains static clipping parameters and calls function when step is initiated.
    '''

    def step(self, weights):
        return clip_weights(
            weights,
            self.nz_cols,
            self.nz_rows,
            self.min_weight_exc,
            self.max_weight_exc,
        )
