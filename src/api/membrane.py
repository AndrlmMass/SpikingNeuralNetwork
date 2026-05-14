from dataclasses import dataclass, field


@dataclass
class LIF:
    """Leaky integrate-and-fire membrane spec for one E/I population pair.

    All time constants are in ms; potentials in mV.
    """

    # --- time constants ---
    tau_m_exc: float = 30.0
    tau_m_inh: float = 30.0
    tau_syn_exc: float = 30.0
    tau_syn_inh: float = 30.0

    # --- resistance ---
    membrane_resistance_exc: float = 30.0
    membrane_resistance_inh: float = 30.0

    # --- potential bounds ---
    resting_potential: float = -70.0
    reset_potential: float = -80.0
    spike_threshold: float = -55.0
    min_mp: float = -100.0
    max_mp: float = 40.0

    # --- noise ---
    mean_noise: float = 0.0
    var_noise: float = 1.0

    # --- spike-frequency adaptation ---
    spike_adaptation: bool = True
    tau_adaptation: float = 100.0
    delta_adaptation: float = 1.0
