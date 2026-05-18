from dataclasses import dataclass


@dataclass
class Sleep:
    """Simulated sleep: periodic synaptic downscaling with noisy spontaneous activity.

    At every ``frequency`` timesteps the network enters a sleep episode lasting
    ``duration`` timesteps.  During sleep, input drive is silenced, Gaussian
    noise is injected into the membrane, and weights are multiplicatively
    scaled toward their target level one small step at a time.

    Parameters
    ----------
    duration : int
        Length of each sleep episode in timesteps.
    frequency : int
        Gap between sleep onsets (timesteps of awake processing between episodes).
    mode : str
        Controls what the weight target means:

        ``'static'``  — every nonzero weight is pushed toward the fixed scalar
        ``w_target`` set in the WeightsSpec (peak_se / peak_ee).

        ``'layer'``   — the total synaptic drive across the whole layer is
        restored to its value at the start of training.

        ``'neuron'``  — each postsynaptic neuron's incoming weight sum is
        individually restored to its initial value (most local form).
    """

    duration: int = 300
    frequency: int = 1050
    mode: str = "static"

    def _to_runner_kwargs(self) -> dict:
        return dict(
            sleep=True,
            normalize_weights=False,
            sleep_duration=self.duration,
            reg_frequency=self.frequency,
            reg_mode=self.mode,
        )


@dataclass
class Normalize:
    """Deterministic weight normalization at fixed intervals (no noise, no sleep).

    At every ``frequency`` timesteps the excitatory weight blocks are rescaled
    without any change to membrane dynamics.  A lighter-weight alternative to
    Sleep when you want weight regularization without the spontaneous-activity
    component.

    Parameters
    ----------
    frequency : int
        Number of awake timesteps between normalization steps.
    mode : str
        Same three options as Sleep: ``'static'``, ``'layer'``, ``'neuron'``.
    """

    frequency: int = 1050
    mode: str = "static"

    def _to_runner_kwargs(self) -> dict:
        return dict(
            sleep=False,
            normalize_weights=True,
            sleep_duration=0,
            reg_frequency=self.frequency,
            reg_mode=self.mode,
        )
