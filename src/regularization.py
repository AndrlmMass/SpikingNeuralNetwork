from dataclasses import dataclass
from numba import njit
import numpy as np


@njit(cache=True)
def static_norm(weights, target, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] = target
    return weights


@njit(cache=True)
def dynamic(weights, scale, nz_rows, nz_cols):
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale[i]
    return weights


@njit(cache=True)
def static_sleep(
    weights,
    w_target_pow,
    sleep_lambda_complement,
    nz_rows,
    nz_cols,
):
    # Apply decay to columns [N_x, N_post] (excitatory weights)
    for i in range(nz_rows.size):
        w = weights[nz_rows[i], nz_cols[i]]
        weights[nz_rows[i], nz_cols[i]] = w_target_pow * w**sleep_lambda_complement
    return weights


@dataclass
class Normalizer:
    mode: str
    initial_sum: np.ndarray
    target: float
    nz_rows: np.ndarray
    nz_cols: np.ndarray

    def __post_init__(self):
        self.scale = np.ones(self.nz_rows.size, dtype=np.float64)

    def step(self, weights):
        if self.mode == "static":
            return static_norm(weights, self.target, self.nz_rows, self.nz_cols)

        elif self.mode == "layer":
            current_sum = weights[self.nz_rows, self.nz_cols].sum()
            self.scale[:] = self.initial_sum.sum() / current_sum
        elif self.mode == "post":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            self.scale[:] = self.initial_sum.sum() / current_sum
        return dynamic(weights, self.scale, self.nz_rows, self.nz_cols)


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
        self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        self.w_target_pow = self.w_target**self.sleep_lambda
        self.sleep_lambda_complement = 1.0 - self.sleep_lambda
        if self.mode == "post" and np.ndim(self.initial_sums) == 0:
            raise ValueError(
                "post mode requires initial_sums to be a per-neuron vector"
            )
        if self.mode == "layer" and np.ndim(self.initial_sums) != 0:
            raise ValueError("layer mode requires initial_sums to be a scalar")

    def onset(self, weights):
        """Call once when sleep begins — precomputes scale from current weights"""
        if self.mode == "layer":
            rho = self.initial_sums / (weights[self.nz_rows, self.nz_cols].sum() + 1e-8)
            self.scale[:] = rho**self.sleep_lambda
        elif self.mode == "post":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            rho = self.initial_sums / (current_sum[self.nz_cols] + 1e-8)
            self.scale[:] = rho**self.sleep_lambda

    def step(self, weights):
        """Call each sleep timestep — pure dispatch to @njit"""
        if self.mode == "static":
            return static_sleep(
                weights,
                self.w_target_pow,
                self.sleep_lambda_complement,
                self.nz_rows,
                self.nz_cols,
            )
        else:
            return dynamic(weights, self.scale, self.nz_rows, self.nz_cols)
