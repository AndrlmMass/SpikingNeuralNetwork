from dataclasses import dataclass
import numpy as np


@dataclass
class Network:
    N: int
    N_exc: int
    N_inh: int
    st: int
    resting_membrane: float
    spike_threshold_default: float
    time: int
    data: np.ndarray

    def __post_init__(self):
        if self.time == 0:
            raise ValueError("total_time_test must be > 0 to create test arrays")

    def build(self):
        # create membrane potential
        membrane_potential = np.full(self.N - self.st, self.resting_membrane)

        # create spikes array
        spikes = np.zeros((self.time, self.N), dtype=np.int8)
        spikes[:, : self.st] = self.data

        # create spike traces for each neuron
        spike_trace = np.zeros(self.N - self.N_inh, dtype=np.float32)

        # create missing arrays
        I_syn_exc = np.zeros(self.N_exc)
        I_syn_inh = np.zeros(self.N_inh)
        a = np.zeros(self.N_exc + self.N_inh)

        # create spike threshold array
        spike_threshold = np.full(
            shape=(self.N - self.st),
            fill_value=self.spike_threshold_default,
            dtype=float,
        )

        # create pre-targets for STDP
        x_tar_se = np.zeros(self.N_x)
        x_tar_ee = np.zeros(self.N_exc)

        return (
            membrane_potential,
            spikes,
            spike_trace,
            I_syn_exc,
            I_syn_inh,
            a,
            spike_threshold,
            x_tar_se,
            x_tar_ee,
        )
