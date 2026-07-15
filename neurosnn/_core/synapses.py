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
            #if spikes[i] == 1:
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
            #if spikes[i] == 1:
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


@njit(parallel=False, cache=True)
def vogels_iSTDP(
    learning_rate, spike_trace, inh_trace, weights,
    N_x, N_exc, N_inh, spikes, rho_0, w_min, w_max, mu_weight,
):
    """Vogels et al. 2011 iSTDP for W_ie (I→E synapses, negative weights).

    W_ie lives at weights[N_x+N_exc:N_x+N_exc+N_inh, N_x:N_x+N_exc].
    Weights are negative; LTP = more negative = stronger inhibition.

    On E spike j:  Δw[i,j] -= lr * x_pre[i]            (Hebbian)
    On I spike i:  Δw[i,j] -= lr * (x_post[j] - 2*ρ₀)  (homeostatic target)

    Soft bounds: LTP uses (w - w_min)^μ, LTD uses (w_max - w)^μ.
    """
    ex = N_x + N_exc

    # On E post-synaptic spike: Hebbian term
    for j in range(N_exc):
        if spikes[N_x + j] == 1:
            for i in range(N_inh):
                x_pre_i = inh_trace[i]
                dw = -learning_rate * x_pre_i
                w = weights[ex + i, N_x + j]
                if dw <= 0.0:  # LTP (more inhibitory)
                    bound = max(w - w_min, 0.0) ** mu_weight
                else:           # LTD (less inhibitory)
                    bound = max(w_max - w, 0.0) ** mu_weight
                weights[ex + i, N_x + j] = w + dw * bound

    # On I pre-synaptic spike: homeostatic term
    for i in range(N_inh):
        if spikes[ex + i] == 1:
            for j in range(N_exc):
                x_post_j = spike_trace[N_x + j]
                dw = -learning_rate * (x_post_j - 2.0 * rho_0)
                w = weights[ex + i, N_x + j]
                if dw <= 0.0:
                    bound = max(w - w_min, 0.0) ** mu_weight
                else:
                    bound = max(w_max - w, 0.0) ** mu_weight
                weights[ex + i, N_x + j] = w + dw * bound

    return weights


@njit(parallel=False, cache=True)
def reward_STDP(
    learning_rate,
    spike_count,
    reward_post,
    weights,
    N_x,
    nonzero_pre_idx,
    w_max,
    w_min,
    mu_weight,
):
    """Reward-modulated STDP with a count-product eligibility (V1).

    Applied ONCE per sample (at the sample boundary), not per timestep.

    Eligibility is the symmetric rate/correlation trace
        e_ij = spike_count[j] * spike_count[i]      (#pre spikes x #post spikes, >= 0)
    accumulated over the sample. The supervised direction is carried by a
    per-post-neuron reward (already baseline-subtracted):
        Δw_ij = learning_rate * reward_post[i] * e_ij * bound
    reward_post[i] > 0 for target-class neurons (potentiate their active inputs),
    < 0 for non-target (depress). Since e_ij = 0 whenever post i is silent, no
    explicit "did i fire" check is needed. Soft bounds match the trace/triplet
    kernels: (w_max - w)^mu for potentiation, (w - w_min)^mu for depression.

    spike_count is length n_neurons (= N_x + N_exc); reward_post is length N_exc.
    """
    n_neurons = spike_count.shape[0]
    for i in range(N_x, n_neurons):
        r_i = reward_post[i - N_x]
        pc_i = spike_count[i]
        # silent post (pc_i == 0) or zero reward -> eligibility/update is zero
        if pc_i == 0.0 or r_i == 0.0:
            continue
        pre_indices = nonzero_pre_idx[i - N_x]
        for j in pre_indices:
            # skip if padding pre-index
            if j == -1:
                continue
            e = spike_count[j] * pc_i
            if e == 0.0:
                continue
            dw = learning_rate * r_i * e
            w = weights[j, i]
            if dw >= 0.0:  # potentiation
                bound = max(w_max - w, 0.0) ** mu_weight
            else:           # depression
                bound = max(w - w_min, 0.0) ** mu_weight
            weights[j, i] = w + dw * bound
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
class iLearner:
    learning_rate: float
    N_x: int
    N_exc: int
    N_inh: int
    rho_0: float
    w_min: float
    w_max: float
    mu_weight: float

    def step(self, weights, spikes, spike_trace, inh_trace):
        return vogels_iSTDP(
            self.learning_rate, spike_trace, inh_trace, weights,
            self.N_x, self.N_exc, self.N_inh, spikes,
            self.rho_0, self.w_min, self.w_max, self.mu_weight,
        )


@dataclass
class RewardLearner:
    """Reward-modulated STDP learner (V1): only feedforward SE weights.

    Accumulate per-neuron spike counts over a sample via `accumulate`, then call
    `step(weights, target_label)` at the sample boundary to apply the reward
    update and reset the counts.

    reward_post[i] = R_i - baseline, with R_i = +1 if neuron_class[i] == target
    else -1. The baseline is an EMA of the mean teacher signal; it centers the
    (mostly-negative, since 1 target vs C-1 non-target classes) reward so the net
    update is ~zero-mean (kills DC drift, reduces variance). For fixed class
    balance it converges to a constant; it becomes a true performance baseline if
    R_i is later made performance-dependent.
    """

    learning_rate: float
    N_x: int
    N_exc: int
    nonzero_pre_idx: list
    neuron_class: np.ndarray  # (N_exc,) fixed class label per exc neuron
    w_max: float
    w_min: float
    mu_weight: float
    baseline_decay: float = 0.01

    def __post_init__(self):
        max_len = max(len(x) for x in self.nonzero_pre_idx)
        self.pre_idx_arr = np.full(
            (len(self.nonzero_pre_idx), max_len), -1, dtype=np.int64
        )
        for i, idx in enumerate(self.nonzero_pre_idx):
            self.pre_idx_arr[i, : len(idx)] = idx
        del self.nonzero_pre_idx
        self.neuron_class = np.asarray(self.neuron_class)
        self.spike_count = np.zeros(self.N_x + self.N_exc, dtype=np.float64)
        # Initialize the baseline to the class-balance mean of R_i so updates are
        # zero-mean from step 0 (1 target class at +1, C-1 non-target at -1 ->
        # mean = (2 - C) / C). Avoids early net-depression drift before the EMA
        # would otherwise converge. For C=2 this is 0.0 (unit-test compatible).
        n_classes = len(np.unique(self.neuron_class))
        self.n_classes = n_classes
        self.baseline = (2.0 - n_classes) / n_classes
        # online efficacy tracking (window since last pop_online_stats)
        self._n = 0
        self._correct = 0

    def pop_online_stats(self):
        """Online training accuracy (net's own pooled prediction vs the reward
        target) over samples since the last call, plus the current baseline.
        Reset the window. NaN acc if no samples fired."""
        acc = (self._correct / self._n) if self._n else float("nan")
        self._n = 0
        self._correct = 0
        return dict(online_acc=acc, baseline=float(self.baseline))

    def accumulate(self, spikes):
        """Add a timestep's (or a (T, n_neurons) block's) spikes to the counts.

        `spikes` may include the inhibitory tail (length N_x+N_exc+N_inh); only
        the input+exc entries (length N_x+N_exc) are counted.
        """
        s = np.asarray(spikes)
        add = s.sum(axis=0) if s.ndim == 2 else s
        n = self.spike_count.shape[0]
        self.spike_count += add[:n]

    def reset_counts(self):
        self.spike_count[:] = 0.0

    def step(self, weights, target_label):
        # online prediction from this sample's pooled exc counts (before reset):
        # the net's own decision = argmax over per-class summed exc rates.
        exc = self.spike_count[self.N_x:]
        if exc.sum() > 0:
            scores = np.array([exc[self.neuron_class == c].sum() for c in range(self.n_classes)])
            self._n += 1
            self._correct += int(scores.argmax() == target_label)
        R = np.where(self.neuron_class == target_label, 1.0, -1.0)
        reward_post = R - self.baseline
        weights = reward_STDP(
            self.learning_rate,
            self.spike_count,
            reward_post,
            weights,
            self.N_x,
            self.pre_idx_arr,
            self.w_max,
            self.w_min,
            self.mu_weight,
        )
        self.baseline += self.baseline_decay * (float(R.mean()) - self.baseline)
        self.reset_counts()
        return weights


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
