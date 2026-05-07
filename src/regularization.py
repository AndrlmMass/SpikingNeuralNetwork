from dataclasses import dataclass
from numba import njit
import numpy as np


@njit(cache=True)
def post_sleep(weights, scale, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale[i]
    return weights


@njit(cache=True)
def post_norm(weights, initial_sum_nz, nz_rows, nz_cols, n_post):
    current_sum = np.zeros(n_post)
    for i in range(nz_rows.size):
        current_sum[nz_cols[i]] += weights[nz_rows[i], nz_cols[i]]
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= initial_sum_nz[i] / (
            current_sum[nz_cols[i]] + 1e-8
        )
    return weights


@njit(cache=True)
def static_or_layer(weights, scale, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale
    return weights


@dataclass
class Normalizer:
    mode: str
    initial_sum: np.ndarray
    target: float
    nz_rows: np.ndarray
    nz_cols: np.ndarray
    weight_cols: int

    def __post_init__(self):
        if self.mode == "static":
            self.scale = self.target
        else:
            self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        if self.mode == "post":
            self.initial_sum_nz = self.initial_sum[self.nz_cols]  # (nnz,)

    def step(self, weights):
        if self.mode == "static":
            return static_or_layer(weights, self.scale, self.nz_rows, self.nz_cols)

        elif self.mode == "layer":
            current_sum = weights[self.nz_rows, self.nz_cols].sum()
            self.scale = self.initial_sum / current_sum
            return static_or_layer(weights, self.scale, self.nz_rows, self.nz_cols)

        else:
            return post_norm(
                weights,
                self.initial_sum_nz,
                self.nz_rows,
                self.nz_cols,
                self.weight_cols,
            )


@dataclass
class Sleep:
    mode: str
    duration: int
    w_target: float
    initial_sums: np.ndarray
    nz_rows: np.ndarray
    nz_cols: np.ndarray

    def __post_init__(self):
        self.sleep_lambda = 1.0 / self.duration
        if self.mode == "static":
            self.scale = self.w_target**self.sleep_lambda

    def onset(self, weights):
        """Call once when sleep begins — precomputes scale from current weights"""
        if self.mode == "layer":
            rho = self.initial_sums / (weights[self.nz_rows, self.nz_cols].sum() + 1e-8)
            self.scale = rho**self.sleep_lambda
        elif self.mode == "post":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            rho = self.initial_sums / (current_sum + 1e-8)
            self.scale = rho[self.nz_cols] ** self.sleep_lambda

    def step(self, weights):
        """Call each sleep timestep — pure dispatch to @njit"""
        if self.mode == "post":
            return post_sleep(weights, self.scale, self.nz_rows, self.nz_cols)
        else:
            return static_or_layer(
                weights,
                self.scale,
                self.nz_rows,
                self.nz_cols,
            )
