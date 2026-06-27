from numba import njit
from dataclasses import dataclass
import numpy as np


@njit(cache=True)
def seed_numba_rng(seed: int):
    """Seed Numba's internal per-thread RNG.

    Numba maintains its own RNG state separate from NumPy's global state.
    Call this once before training begins to make membrane noise reproducible.
    """
    np.random.seed(seed)


@njit(cache=True)
def update_membrane_potential(
    mp,
    mp_new,
    weights_exc,
    weights_inh,
    spikes,
    noisy_potential_now,
    resting_potential,
    membrane_resistance_exc,
    membrane_resistance_inh,
    dt,
    st,
    ex,
    track_stats,
    N_exc,
    N_inh,
    I_syn_exc,
    I_syn_inh,
    tau_syn_exc,
    tau_syn_inh,
    tau_m_exc,
    tau_m_inh,
    mean_noise,
    var_noise,
):
    '''
    Function takes current spiking activity and updates membrane potentials for all non-input neurons. 
    Returns intermediate change-values if track_stats is true
    
    '''


    if track_stats:
        # allocate arrays for stat tracking
        delta_mp_ex = np.zeros(N_exc)
        delta_mp_ih = np.zeros(N_inh)
        delta_I_syn_ex = np.zeros(N_exc)
        delta_I_syn_ih = np.zeros(N_inh)
    else:
        # allocate empty arrays to ensure stable return variables (JIT requirement) 
        delta_mp_ex = np.empty(0)
        delta_mp_ih = np.empty(0)
        delta_I_syn_ex = np.empty(0)
        delta_I_syn_ih = np.empty(0)

    # fetch indices for spiking neurons
    nonzero_all = np.where(spikes != 0)[0]
    # fetch indices for excitatory layer spiking neurons (hidden recurrent layer)
    nonzero_exc_src = np.where(spikes[st:ex] != 0)[0]

    # --- Excitatory population ---

    # Loop over each excitatory post-neurons to update membrane potential
    for i in range(N_exc):
        # compute membrane current drive
        drive = 0.0
        # loop over nonzero presynaptic spiking neurons (includes all three layers) per post-neuron
        for j in nonzero_all:
            drive += weights_exc[i, j]
        # Compute and update total change in input current
        d_I = (-I_syn_exc[i] + drive) * dt / tau_syn_exc
        I_syn_exc[i] += d_I
        # Compute change in membrane potential (LIF formula)
        d_mp = (
            (-(mp[i] - resting_potential) + membrane_resistance_exc * I_syn_exc[i])
            / tau_m_exc
            * dt
        )
        # Update membrane potential
        mp_new[i] = mp[i] + d_mp
        # Apply Gaussian noise to membrane potential if network is in sleep-mode 
        if noisy_potential_now:
            mp_new[i] += np.random.normal(mean_noise, var_noise)
        # Update delta-membrane change and delta-input current trackers for run stats
        if track_stats:
            delta_mp_ex[i] = d_mp
            delta_I_syn_ex[i] = d_I

    # --- Inhibitory population ---

    # Loop over each inhibitory post-neuron to update membrane potential
    for i in range(N_inh):
        # Create inhibitory adjusted index
        ih_id = i + N_exc
        # Compute current membrane drive
        drive = 0.0
        for j in nonzero_exc_src:
            drive += weights_inh[i, j]
        # Apply decaying and time constants to drive
        d_I = (-I_syn_inh[i] + drive) * dt / tau_syn_inh
        # Update total membrane current
        I_syn_inh[i] += d_I
        # Compute change in membrane potential
        d_mp = (
            (-(mp[ih_id] - resting_potential) + membrane_resistance_inh * I_syn_inh[i])
            / tau_m_inh
            * dt
        )
        # Update total membrane potential
        mp_new[ih_id] = mp[ih_id] + d_mp
        # Apply Gaussian noise to membrane potential
        if noisy_potential_now:
            mp_new[ih_id] += np.random.normal(mean_noise, var_noise)
        # Track change in current and membrane potential
        if track_stats:
            delta_mp_ih[i] = d_mp
            delta_I_syn_ih[i] = d_I

    return (
        mp_new,
        I_syn_exc,
        I_syn_inh,
        delta_mp_ex,
        delta_mp_ih,
        delta_I_syn_ex,
        delta_I_syn_ih,
    )


@njit(cache=True)
def update_spikes(
    st,
    ih,
    N_exc,
    mp,
    dt,
    a,
    spike_trace,
    spikes,
    max_mp,
    min_mp,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold,
    spike_threshold_default,
    reset_potential,
    decay,
):
    '''
    Function takes updated membrane potential, estimates which neuronal potentials surpass the spiking threshold, 
    updates spiking arrays and spiking thresholds, then resets the membrane potential.
    '''
    # Compute total non-input neurons
    n_total = ih - st

    # Loop over non-input neurons to clip potentials and compute threshold-surpassing potentials
    for j in range(n_total):
        # Clip potentials to min/max
        if mp[j] < min_mp:
            mp[j] = min_mp
        elif mp[j] > max_mp:
            mp[j] = max_mp
        # Compute potentials above threshold and reset
        if mp[j] > spike_threshold[j]:
            spikes[st + j] = 1
            mp[j] = reset_potential
    # Update spiking threshold
    if spike_adaption:
        # Loop over non-input neurons
        for j in range(n_total):
            # Decay alpha (additive spiking threshold variable)
            a[j] += (-a[j] / tau_adaption) * dt
            # If recent spike, increase alpha by delta_adaption
            if spikes[st + j] == 1:
                a[j] += delta_adaption
            # Add alpha to spiking threshold
            spike_threshold[j] = spike_threshold_default + a[j]
    # Update spike trace (used in trace-STDP later)
    
    # Loop over excitatory neurons (inhibitoroy neurons are static and thus do not require tracking)
    for j in range(N_exc + st):
        idx = j
        # Increase trace if neuron recently spiked
        if spikes[idx] == 1:
            spike_trace[idx] = spike_trace[idx] * decay + 1.0
        # Else, only apply decay
        else:
            spike_trace[idx] *= decay

    return (
        mp,
        spikes,
        spike_threshold,
        a,
        spike_trace,
    )

@njit(cache=True)
def update_slow_traces(spikes, r2, o2, decay_r2, decay_o2, N_x, N_exc):
    """Decay and update slow pre (r2) and post (o2) traces for triplet STDP.

    r2 tracks slow pre-synaptic bursting (input + excitatory neurons).
    o2 tracks slow post-synaptic bursting (excitatory neurons only).
    Both decay exponentially and jump by +1 on each spike.
    """
    for j in range(N_x + N_exc):
        r2[j] *= decay_r2
        if spikes[j] == 1:
            r2[j] += 1.0
    for j in range(N_exc):
        o2[j] *= decay_o2
        if spikes[N_x + j] == 1:
            o2[j] += 1.0
    return r2, o2


def update_x_tar(spike_trace, N_x, mode="mean", pct_se=60.0, pct_ee=30.0):
    '''
    Update trace target. If presynaptic neuron is above target, the weight is strenghtened, and below it is weakened.
    See trace-STDP function for more details.

    mode:
        "mean"       - population-mean trace per layer (original behaviour).
        "percentile" - Kth percentile over the *active* (nonzero-trace) sub-population
                       per layer. Robust to the heavy zero-inflation of both layers
                       (input ~80% background, exc median trace ~0): a percentile taken
                       over all neurons would collapse to ~0. A higher SE percentile
                       depresses weak/surround pixels (sharpens RFs); a lower EE
                       percentile spares moderately-active neurons (anti-domination).
    '''
    if mode == "mean":
        x_tar_se = np.mean(spike_trace[:N_x], axis=0)
        x_tar_ee = np.mean(spike_trace[N_x:], axis=0)
        return x_tar_se, x_tar_ee
    # percentile over active (nonzero) traces; empty active set -> 0.0
    se_tr = spike_trace[:N_x]
    ee_tr = spike_trace[N_x:]
    se_act = se_tr[se_tr > 0]
    ee_act = ee_tr[ee_tr > 0]
    x_tar_se = np.percentile(se_act, pct_se) if se_act.size else 0.0
    x_tar_ee = np.percentile(ee_act, pct_ee) if ee_act.size else 0.0
    return x_tar_se, x_tar_ee

@dataclass
class NeuronState:
    st: int
    ih: int
    N_exc: int
    dt: float | int
    max_mp: float | int
    min_mp: float | int
    spike_adaption: bool
    tau_adaption: float
    delta_adaption: float | int
    spike_threshold_default: (
        np.ndarray
    )
    reset_potential: float | int
    tau_trace: float

    '''
    NeuronState object maintains constant variables related to "update_spikes" function
    '''

    def __post_init__(self):
        self.decay = np.exp(-self.dt / self.tau_trace)

    def step(self, mp, a, spike_trace, spikes, spike_threshold):
        return update_spikes(
            self.st,
            self.ih,
            self.N_exc,
            mp,
            self.dt,
            a,
            spike_trace,
            spikes,
            self.max_mp,
            self.min_mp,
            self.spike_adaption,
            self.tau_adaption,
            self.delta_adaption,
            spike_threshold,
            self.spike_threshold_default,
            self.reset_potential,
            self.decay,
        )


@dataclass
class MembranePotential:
    mp_new: np.ndarray
    resting_potential: float
    membrane_resistance_exc: float
    membrane_resistance_inh: float
    dt: float
    st: int
    ex: int
    track_stats: bool
    N_exc: int
    N_inh: int
    tau_syn_exc: float
    tau_syn_inh: float
    tau_m_exc: float
    tau_m_inh: float
    mean_noise: float
    var_noise: float

    '''
    MembranePotential object retains constant parameters for membrane update calls
    '''

    def step(
        self,
        mp,
        weights_exc,
        weights_inh,
        spikes,
        I_syn_exc,
        I_syn_inh,
        noisy_potential_now,
    ):
        return update_membrane_potential(
            mp,
            self.mp_new,
            weights_exc,
            weights_inh,
            spikes,
            noisy_potential_now,
            self.resting_potential,
            self.membrane_resistance_exc,
            self.membrane_resistance_inh,
            self.dt,
            self.st,
            self.ex,
            self.track_stats,
            self.N_exc,
            self.N_inh,
            I_syn_exc,
            I_syn_inh,
            self.tau_syn_exc,
            self.tau_syn_inh,
            self.tau_m_exc,
            self.tau_m_inh,
            self.mean_noise,
            self.var_noise,
        )
