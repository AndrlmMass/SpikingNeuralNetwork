from dataclasses import dataclass


@dataclass
class VogelsSTDP:
    """Vogels et al. 2011 inhibitory STDP for I→E synapses.

    Pass as `inh_learner` to model.train() alongside a TraceSTDP or
    TripletSTDP excitatory learner.

    Parameters
    ----------
    learning_rate : float
        Step size for W_ie updates (default: 0.01).
    rho_0 : float
        Target E neuron firing rate in trace units.
        rho_0 = target_Hz * tau_trace_ms / 1000.
        At tau_trace=20ms: rho_0=0.1 ≈ 5 Hz, rho_0=0.04 ≈ 2 Hz.
    mu_weight : float
        Soft-bound exponent (shared with excitatory rule).
    min_weight_inh : float
        Lower hard bound for I→E weights (most inhibitory).
    max_weight_inh : float
        Upper hard bound for I→E weights (least inhibitory).
    """

    learning_rate: float = 0.01
    rho_0: float = 0.1
    mu_weight: float = 0.6
    min_weight_inh: float = -25.0
    max_weight_inh: float = -0.01

    def _to_runner_kwargs(self) -> dict:
        return dict(
            use_vogels=True,
            lr_inh=self.learning_rate,
            rho_0=self.rho_0,
        )


@dataclass
class TripletSTDP:
    """Pfister & Gerstner (2006) triplet STDP learning rule.

    Extends pair STDP with slow pre (r2) and post (o2) traces so that
    bursting history modulates both LTP and LTD:

        LTP = A2+ * r1 + A3+ * r1 * o2    (at post-synaptic spike)
        LTD = A2- * o1 + A3- * o1 * r2    (at post-synaptic spike, approx.)

    Default amplitudes approximate the visual-cortex fit from Pfister &
    Gerstner (2006), Table 2.

    Parameters
    ----------
    tau_trace : int
        Shared fast trace time constant for r1 / o1 (ms).
    tau_x : float
        Slow presynaptic trace time constant for r2 (ms).
    tau_y : float
        Slow postsynaptic trace time constant for o2 (ms).
    A2_plus, A3_plus : float
        Pair and triplet LTP amplitudes.
    A2_minus, A3_minus : float
        Pair and triplet LTD amplitudes.
    """

    learning_rate: float = 0.0004
    tau_trace: int = 20
    tau_x: float = 101.0
    tau_y: float = 125.0
    A2_plus: float = 5e-10
    A3_plus: float = 6.2e-3
    A2_minus: float = 7e-3
    A3_minus: float = 2.3e-4
    w_max: float = 10.0
    mu_weight: float = 0.6
    update_freq: int = 100
    clip_weights: bool = True
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
            update_weights_freq=self.update_freq,
            clip_weights=self.clip_weights,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
            min_weight_inh=self.min_weight_inh,
            max_weight_inh=self.max_weight_inh,
            use_triplet=True,
            tau_x=self.tau_x,
            tau_y=self.tau_y,
            A2_plus=self.A2_plus,
            A3_plus=self.A3_plus,
            A2_minus=self.A2_minus,
            A3_minus=self.A3_minus,
            x_tar_mode="mean",
            x_tar_pct_se=60.0,
            x_tar_pct_ee=30.0,
        )


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
    x_tar_static_se: float = 0.2
    x_tar_static_ee: float = 0.2

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
            x_tar_static_se=self.x_tar_static_se,
            x_tar_static_ee=self.x_tar_static_ee,
            update_weights_freq=self.update_freq,
            clip_weights=self.clip_weights,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
            min_weight_inh=self.min_weight_inh,
            max_weight_inh=self.max_weight_inh,
        )


@dataclass
class RewardSTDP:
    """Reward-modulated STDP on the feedforward SE weights (supervised, V1).

    Excitatory neurons are assigned to classes (fixed). A count-product
    eligibility (#pre x #post spikes per sample) is gated by a per-neuron reward
    (+1 for target-class neurons, -1 otherwise), centered by a running baseline,
    and applied once per sample at the sample boundary. Pass as `learner` to
    model.train() in place of TraceSTDP.

    V1 scope: only SE weights are plastic — keep inh_learner=None (static
    inhibition as a fixed WTA scaffold) and no recurrent plasticity. Use a
    Normalize regularizer and the homeostatic spike-adaptation threshold.

    Parameters
    ----------
    learning_rate : float
        Step size for the reward weight update.
    baseline_decay : float
        EMA rate for the reward baseline (0 disables centering).
    class_assignment : {"mod", "random"}
        "mod": exc neuron j -> class j % N_classes; "random": seeded shuffle.
    seed : int
        Seed for random class assignment.
    """

    learning_rate: float = 0.0005
    baseline_decay: float = 0.01
    class_assignment: str = "mod"
    seed: int = 0
    shuffle_labels: bool = False   # control: reward on random targets (signal = noise)

    def _to_runner_kwargs(self) -> dict:
        return dict(
            use_reward=True,
            reward_learning_rate=self.learning_rate,
            reward_baseline_decay=self.baseline_decay,
            reward_class_assignment=self.class_assignment,
            reward_seed=self.seed,
            reward_shuffle_labels=self.shuffle_labels,
        )
