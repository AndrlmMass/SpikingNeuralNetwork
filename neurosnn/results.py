from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainResult:
    """Yielded after each training batch.

    weights is a reference to the live weight matrix — no copy is made.
    accuracy and phi are None until the evaluator has been fitted (first batch).
    spikes is shape (n_images, N_exc) mean firing rates — only present when
    return_spikes=True is passed to model.train().
    stats is a dict of mean neuron/synapse diagnostics — only present when
    track_stats=True or track_weights=True is passed to model.train().
    """

    epoch: int
    batch: int
    weights: np.ndarray
    accuracy: Optional[float] = None
    phi: Optional[float] = None
    spikes: Optional[np.ndarray] = None
    stats: Optional[dict] = None


@dataclass
class EvalResult:
    """Returned by model.validate() and model.test().

    spikes is shape (n_images, N_exc) mean firing rates — only present when
    return_spikes=True is passed to model.validate() / model.test().
    """

    accuracy: Optional[float]
    phi: Optional[float]
    split: str = "val"
    spikes: Optional[np.ndarray] = None
