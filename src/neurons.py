from numba import njit, prange
from dataclasses import dataclass
import numpy as np


@njit(parallel=True, cache=True)
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
    nonzero_exc_src = np.where(spikes[st:ex] != 0)[0] + st  # exc spiking → inh targets

    # --- Excitatory population ---
    for i in prange(N_exc):
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
    for i in prange(N_inh):
        ih_id = i + N_exc
        drive = 0.0
        for j in nonzero_exc_src:  # only spiking exc neurons
            drive += weights_inh[i, j]

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


@njit(parallel=True, cache=True)
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
    tau_trace,
):
    decay = np.exp(-dt / tau_trace)
    n_total = ih - st

    for j in prange(n_total):
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

    for j in prange(N_exc + st):
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

    def step(self, mp, a, spike_trace, spikes, spike_threshold):

        return update_spikes(
            st=self.st,
            ih=self.ih,
            N_exc=self.N_exc,
            mp=mp,
            dt=self.dt,
            a=a,
            spike_trace=spike_trace,
            spikes=spikes,
            max_mp=self.max_mp,
            min_mp=self.min_mp,
            spike_adaption=self.spike_adaption,
            tau_adaption=self.tau_adaption,
            delta_adaption=self.delta_adaption,
            spike_threshold=spike_threshold,
            spike_threshold_default=self.spike_threshold_default,
            reset_potential=self.reset_potential,
            tau_trace=self.tau_trace,
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
            mp=mp,
            weights_exc=weights_exc,
            weights_inh=weights_inh,
            spikes=spikes,
            I_syn_exc=I_syn_exc,
            I_syn_inh=I_syn_inh,
            mp_new=self.mp_new,
            noisy_potential_now=noisy_potential_now,
            resting_potential=self.resting_potential,
            membrane_resistance_exc=self.membrane_resistance_exc,
            membrane_resistance_inh=self.membrane_resistance_inh,
            dt=self.dt,
            st=self.st,
            ex=self.ex,
            track_stats=self.track_stats,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            tau_syn_exc=self.tau_syn_exc,
            tau_syn_inh=self.tau_syn_inh,
            tau_m_exc=self.tau_m_exc,
            tau_m_inh=self.tau_m_inh,
            mean_noise=self.mean_noise,
            var_noise=self.var_noise,
        )
