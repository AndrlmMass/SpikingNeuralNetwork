from dataclasses import dataclass
from numba import njit
import numpy as np


@njit
def post_sleep(weights, scale, nz_rows, nz_cols):
    '''
    The "post" sleep method applies a post-neuron-scale based on the ratio 
    between the starting sum of weights and the current sum of weights at time t.
    It loops over all nonzero weight combinations and apply precomputed scale.  
    '''
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale[i]
    return weights


@njit
def post_norm(weights, initial_sum_nz, nz_rows, nz_cols, n_post):
    '''
    Same as post_sleep function, but the downscaling is performed immediately. 
    '''
    # create empty array for current sum of incoming presynaptic weights per post-neuron
    current_sum = np.zeros(n_post)
    # loop over all combinations of weights and update incoming sum
    for i in range(nz_rows.size):
        current_sum[nz_cols[i]] += weights[nz_rows[i], nz_cols[i]]
    # apply downscaling per weight combination 
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= initial_sum_nz[i] / (
            current_sum[nz_cols[i]] + 1e-8
        )
    return weights


@njit
def layer(weights, scale, nz_rows, nz_cols):
    '''
    Downscales weights based on layer-scale (scale computed outside function)
    '''
    for i in range(nz_rows.size):
        weights[nz_rows[i], nz_cols[i]] *= scale
    return weights


@njit
def static(weights, target, nz_rows, nz_cols):
    '''
    Static regularization sets all weights to their original starting value.
    Yes, this is a terrible method and only used for contrasting with layer
    and neuron scaling targets. 
    '''
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

    '''
    Normalizes weights with three different scaling methods: per neuron, per layer or static target.
    Updates scale every time the object is called by computing ratio between current weights and 
    starting weight sums.  
    '''

    def __post_init__(self):
        # Scale all weights to original weight values if static mode
        if self.mode == "static":
            self.scale = float(np.sum(self.initial_sum)) / max(self.nz_rows.size, 1)
        # Layer and neuron: create placeholder array
        else:
            self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        # Create per-neuron initial sum for post-neuron 
        if self.mode == "neuron":
            self.initial_sum_nz = self.initial_sum[self.nz_cols]

    def step(self, weights, t=None):
        '''
        Updates all scaling ratios and initial sums, then passes to respective target function
        '''
        if self.mode == "static":
            weights = static(weights, self.scale, self.nz_rows, self.nz_cols)
        elif self.mode == "layer":
            # Compute current layer (e.g. input sum of weights) at time t
            current_sum = weights[self.nz_rows, self.nz_cols].sum()
            # Update scale by computing ratio
            self.scale = self.initial_sum / current_sum
            # Call layer scaling function
            weights = layer(weights, self.scale, self.nz_rows, self.nz_cols)
        else:  
            # Apply neurons-specific scaling
            weights = post_norm(
                weights,
                self.initial_sum_nz,
                self.nz_rows,
                self.nz_cols,
                self.weight_cols,
            )
        # If tracking weights during regularization, append weights and timepoint
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

    '''
    Similar to normalization, but the regularization occurs over d timesteps. 
    Designed to reach same targets, but allows STDP and noisy membrane potentials
    during downscaling. 
    '''

    def __post_init__(self):
        # Compute target scaling to ensure equal end-target for regularization
        self.sleep_lambda = 1.0 / self.duration
        # Pre-allocate scaling arrays for memory efficiency
        self.scale = np.ones(self.nz_rows.size, dtype=np.float64)
        # Assign original weights as targets for static regime
        if self.mode == "static":
            self.w_target = float(np.sum(self.initial_sums)) / max(self.nz_rows.size, 1)

    def onset(self, weights):
        '''
        Occurs at each initiation of sleep regularizer object. 
        Computes current weights and scaling ratioes per regime (static, layer or neuron).
        '''
        # Scale weights down towards original weights during static regime
        if self.mode == "static":
            current_w = weights[self.nz_rows, self.nz_cols]
            self.scale = (self.w_target / (current_w + 1e-8)) ** self.sleep_lambda
        # Compute layer-wise scaling target ratios and apply time exponent
        elif self.mode == "layer":
            rho = self.initial_sums / (weights[self.nz_rows, self.nz_cols].sum() + 1e-8)
            self.scale = rho**self.sleep_lambda
        # Compute current incoming sum per post-neuron and apply neuron-specific downscaling
        elif self.mode == "neuron":
            current_sum = np.bincount(
                self.nz_cols,
                weights[self.nz_rows, self.nz_cols],
                minlength=weights.shape[1],
            )
            rho = self.initial_sums / (current_sum + 1e-8)
            self.scale = rho[self.nz_cols] ** self.sleep_lambda

    def step(self, weights, t=None):
        '''
        Calls regime functions and uses predefined scaling factors for each sleep initialization
        '''
        if self.mode == "layer":
            weights = layer(weights, self.scale, self.nz_rows, self.nz_cols)
        else:  # neuron and static both use per-weight multiplicative scaling
            weights = post_sleep(weights, self.scale, self.nz_rows, self.nz_cols)
        # Record weights and timepoint if record_fn is iniated
        if self.record_fn is not None:
            self.record_fn(weights, t)
        return weights
