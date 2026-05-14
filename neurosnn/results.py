from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainResult:
    """Yielded after each training batch.

    weights is a reference to the live weight matrix — no copy is made.
    accuracy and phi are None until the evaluator has been fitted (first batch).
    """

    epoch: int
    batch: int
    weights: np.ndarray
    accuracy: Optional[float] = None
    phi: Optional[float] = None


@dataclass
class EvalResult:
    """Returned by model.validate() and model.test()."""

    accuracy: Optional[float]
    phi: Optional[float]
    split: str = "val"
