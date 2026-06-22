from dataclasses import dataclass


@dataclass
class TraceSTDP:
    """Spike-trace STDP with BCM-style soft weight bound.

    The rule potentiates a synapse when the presynaptic trace exceeds the
    population-mean trace at the moment the postsynaptic neuron fires, and
    depresses it otherwise.  A multiplicative soft bound prevents runaway
    growth.

    All time constants are in ms.

    Parameters
    ----------
    learning_rate : float
        Step size for each weight update.
    tau_trace : int
        Decay time constant for the presynaptic spike trace.
    w_max : float
        Soft upper bound on excitatory weights.
    mu_weight : float
        Sharpness of the soft bound — higher values suppress growth more
        aggressively near w_max.
    update_freq : int
        Number of timesteps between weight update calls.
    clip_weights : bool
        Whether to hard-clip weights to [min_exc, max_exc] / [min_inh, max_inh]
        at each update.
    min_weight_exc : float
        Lower hard bound for excitatory weights (only applied if clip_weights).
    max_weight_exc : float
        Upper hard bound for excitatory weights (only applied if clip_weights).
    min_weight_inh : float
        Lower hard bound for inhibitory weights (only applied if clip_weights).
    max_weight_inh : float
        Upper hard bound for inhibitory weights (only applied if clip_weights).
    """

    learning_rate: float = 0.0008
    tau_trace: int = 25
    w_max: float = 10.0
    mu_weight: float = 0.6
    update_freq: int = 100

    x_tar_mode: str = "mean"
    x_tar_pct_se: float = 60.0
    x_tar_pct_ee: float = 30.0

    clip_weights: bool = False
    min_weight_exc: float = 0.01
    max_weight_exc: float = 25.0
    min_weight_inh: float = -25.0
    max_weight_inh: float = -0.01

    def _to_runner_kwargs(self) -> dict:
        return dict(
            learning_rate=self.learning_rate,
            tau_trace=self.tau_trace,
            w_max=self.w_max,
            mu_weight=self.mu_weight,
            x_tar_mode=self.x_tar_mode,
            x_tar_pct_se=self.x_tar_pct_se,
            x_tar_pct_ee=self.x_tar_pct_ee,
            update_weights_freq=self.update_freq,
            clip_weights=self.clip_weights,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
            min_weight_inh=self.min_weight_inh,
            max_weight_inh=self.max_weight_inh,
        )
