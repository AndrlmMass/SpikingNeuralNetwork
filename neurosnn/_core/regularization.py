from dataclasses import dataclass
from numba import njit
import numpy as np


@njit
def post_sleep(weights, scale, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale[i]
    return weights


@njit
def post_norm(weights, initial_sum_nz, nz_rows, nz_cols, n_post):
    current_sum = np.zeros(n_post)
    for i in range(nz_rows.size):
        current_sum[nz_cols[i]] += weights[nz_rows[i], nz_cols[i]]
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= initial_sum_nz[i] / (
            current_sum[nz_cols[i]] + 1e-8
        )
    return weights


@njit
def layer(weights, scale, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale
    return weights


@njit
def static(weights, target, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] = target
    return weights


@dataclass
class Normalizer:
    mode: str
    initial_sum: np.ndarray
    target: float
    nz_rows: np.ndarray
    nz_cols: np.ndarray
    weight_cols: int
    record_fn: "callable | None" = None

    def __post_init__(self):
        if self.mode == "static":
            self.scale = float(np.sum(self.initial_sum)) / max(self.nz_rows.size, 1)
        else:
            self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        if self.mode == "neuron":
            self.initial_sum_nz = self.initial_sum[self.nz_cols]

    def step(self, weights, t=None):
        if self.mode == "static":
            weights = static(weights, self.scale, self.nz_rows, self.nz_cols)
        elif self.mode == "layer":
            current_sum = weights[self.nz_rows, self.nz_cols].sum()
            self.scale = self.initial_sum / current_sum
            weights = layer(weights, self.scale, self.nz_rows, self.nz_cols)
        else:  # neuron
            weights = post_norm(
                weights,
                self.initial_sum_nz,
                self.nz_rows,
                self.nz_cols,
                self.weight_cols,
            )
        if self.record_fn is not None:
            self.record_fn(weights, t)
        return weights


@dataclass
class Sleep:
    mode: str
    duration: int
    w_target: float
    initial_sums: np.ndarray
    nz_rows: np.ndarray
    nz_cols: np.ndarray
    record_fn: "callable | None" = None

    def __post_init__(self):
        self.sleep_lambda = 1.0 / self.duration
        self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        if self.mode == "static":
            self.w_target = float(np.sum(self.initial_sums)) / max(self.nz_rows.size, 1)

    def onset(self, weights):
        if self.mode == "static":
            current_w = weights[self.nz_rows, self.nz_cols]
            self.scale = (self.w_target / (current_w + 1e-8)) ** self.sleep_lambda
        elif self.mode == "layer":
            rho = self.initial_sums / (weights[self.nz_rows, self.nz_cols].sum() + 1e-8)
            self.scale = rho**self.sleep_lambda
        elif self.mode == "neuron":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            rho = self.initial_sums / (current_sum + 1e-8)
            self.scale = rho[self.nz_cols] ** self.sleep_lambda

    def step(self, weights, t=None):
        if self.mode == "layer":
            weights = layer(weights, self.scale, self.nz_rows, self.nz_cols)
        else:  # neuron and static both use per-weight multiplicative scaling
            weights = post_sleep(weights, self.scale, self.nz_rows, self.nz_cols)
        if self.record_fn is not None:
            self.record_fn(weights, t)
        return weights
