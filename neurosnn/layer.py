from dataclasses import dataclass, field

from neurosnn.membrane import LIF
from neurosnn.weights import WeightsSpec, receptive_fields


@dataclass
class Layer:
    """Descriptor for one E/I population layer.

    Bundles population sizes with membrane dynamics and connectivity specs.
    No computation happens here — the Model reads this at train() time.

    Example
    -------
    layer = Layer(
        N_exc=400,
        N_inh=100,
        membrane=membrane.LIF(tau_m_exc=20.0),
        weights=weights.receptive_fields(peak_se=0.1),
    )
    """

    N_exc: int = 400
    N_inh: int = 100
    membrane: LIF = field(default_factory=LIF)
    weights: WeightsSpec = field(default_factory=receptive_fields)
