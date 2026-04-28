import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def static_norm(weights, target, nz_rows, nz_cols):
    for i in prange(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] = target
    return weights


@njit(cache=True, parallel=True)
def dynamic(weights, scale, nz_rows, nz_cols):
    for i in prange(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale[i]
    return weights


@njit(parallel=True, cache=True)
def static_sleep(
    weights,
    w_target,
    sleep_lambda,
    nz_rows,
    nz_cols,
):
    # Apply decay to columns [N_x, N_post] (excitatory weights)
    for i in prange(nz_rows.size):
        w = weights[nz_rows[i], nz_cols[i]]
        weights[nz_rows[i], nz_cols[i]] = w * (w_target / w) ** sleep_lambda
    return weights


class Normalizer:
    def __init__(self, mode, initial_sum, target, nz_rows, nz_cols):
        self.mode = mode
        self.initial_sum = initial_sum
        self.target = target
        self.nz_rows = nz_rows
        self.nz_cols = nz_cols

    def step(self, weights):
        if self.mode == "layer":
            current_sum = weights[self.nz_rows, self.nz_cols].sum()
            self.scale = np.full(
                self.nz_rows.size, self.initial_sum.sum() / current_sum
            )
        elif self.mode == "post":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            self.scale = self.initial_sum / (current_sum[self.nz_cols] + 1e-8)
        if self.mode == "static":
            return static_norm(weights, self.target, self.nz_rows, self.nz_cols)
        return dynamic(weights, self.scale, self.nz_rows, self.nz_cols)


class SleepRegularizer:
    def __init__(self, mode, duration, w_target, initial_sums, nz_rows, nz_cols):
        self.mode = mode
        self.sleep_lambda = 1.0 / duration
        self.w_target = w_target
        self.nz_rows = nz_rows
        self.nz_cols = nz_cols
        if mode == "post" and np.ndim(initial_sums) == 0:
            raise ValueError(
                "post mode requires initial_sums to be a per-neuron vector"
            )
        if mode == "layer" and np.ndim(initial_sums) != 0:
            raise ValueError("layer mode requires initial_sums to be a scalar")
        self.initial_sums = initial_sums

    def onset(self, weights):
        """Call once when sleep begins — precomputes scale from current weights"""
        if self.mode == "layer":
            rho = self.initial_sums / (weights[self.nz_rows, self.nz_cols].sum() + 1e-8)
            self.scale = np.full(self.nz_rows.size, rho**self.sleep_lambda)
        elif self.mode == "post":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            rho = self.initial_sums / (current_sum[self.nz_cols] + 1e-8)
            self.scale = rho**self.sleep_lambda

    def step(self, weights):
        """Call each sleep timestep — pure dispatch to @njit"""
        if self.mode == "static":
            return static_sleep(
                weights, self.w_target, self.sleep_lambda, self.nz_rows, self.nz_cols
            )
        else:
            return dynamic(weights, self.scale, self.nz_rows, self.nz_cols)
