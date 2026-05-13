from numba import njit
from dataclasses import dataclass
import numpy as np


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
    if track_stats:
        delta_mp_ex = np.zeros(N_exc)
        delta_mp_ih = np.zeros(N_inh)
        delta_I_syn_ex = np.zeros(N_exc)
        delta_I_syn_ih = np.zeros(N_inh)
    else:
        delta_mp_ex = np.empty(0)
        delta_mp_ih = np.empty(0)
        delta_I_syn_ex = np.empty(0)
        delta_I_syn_ih = np.empty(0)

    # --- Sparse presynaptic indices (computed once, shared across prange) ---
    nonzero_all = np.where(spikes != 0)[0]  # all spiking → exc targets
    nonzero_exc_src = np.where(spikes[st:ex] != 0)[0]  # exc spiking → inh targets
    # if nonzero_exc_src.size != 0:
    #     print(f"here is the list {nonzero_exc_src}")

    # np.copyto(weights_exc, weights[:, self.st : self.ex].T)
    # np.copyto(weights_inh, weights[:, self.ex : self.ih].T)

    # --- Excitatory population ---
    for i in range(N_exc):
        drive = 0.0
        for j in nonzero_all:  # only active pre-synaptic neurons
            drive += weights_exc[i, j]

        d_I = (-I_syn_exc[i] + drive) * dt / tau_syn_exc
        I_syn_exc[i] += d_I

        d_mp = (
            (-(mp[i] - resting_potential) + membrane_resistance_exc * I_syn_exc[i])
            / tau_m_exc
            * dt
        )
        mp_new[i] = mp[i] + d_mp
        if noisy_potential_now:
            mp_new[i] += np.random.normal(mean_noise, var_noise)
        if track_stats:
            delta_mp_ex[i] = d_mp
            delta_I_syn_ex[i] = d_I

    # --- Inhibitory population ---
    for i in range(N_inh):
        ih_id = i + N_exc
        drive = 0.0
        for j in nonzero_exc_src:
            drive += weights_inh[i, j]
            # if drive < 0.0:
            #     print(f"drive is negative: {drive}")

        d_I = (-I_syn_inh[i] + drive) * dt / tau_syn_inh
        I_syn_inh[i] += d_I

        d_mp = (
            (-(mp[ih_id] - resting_potential) + membrane_resistance_inh * I_syn_inh[i])
            / tau_m_inh
            * dt
        )
        mp_new[ih_id] = mp[ih_id] + d_mp
        if noisy_potential_now:
            mp_new[ih_id] += np.random.normal(mean_noise, var_noise)
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
    n_total = ih - st

    for j in range(n_total):
        if mp[j] < min_mp:
            mp[j] = min_mp
        elif mp[j] > max_mp:
            mp[j] = max_mp
        if mp[j] > spike_threshold[j]:
            spikes[st + j] = 1
            mp[j] = reset_potential

    if spike_adaption:
        for j in range(n_total):
            a[j] += (-a[j] / tau_adaption) * dt
            if spikes[st + j] == 1:
                a[j] += delta_adaption
            spike_threshold[j] = spike_threshold_default + a[j]

    for j in range(N_exc + st):
        idx = j
        if spikes[idx] == 1:
            spike_trace[idx] = spike_trace[idx] * decay + 1.0
        else:
            spike_trace[idx] *= decay

    return (
        mp,
        spikes,
        spike_threshold,
        a,
        spike_trace,
    )


def update_x_tar(spike_trace, N_x):
    x_tar_se = np.mean(spike_trace[:N_x], axis=0)
    x_tar_ex = np.mean(spike_trace[N_x:], axis=0)
    return x_tar_se, x_tar_ex


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
    )  # should we consider changing this to a float instead?
    reset_potential: float | int
    tau_trace: float

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
