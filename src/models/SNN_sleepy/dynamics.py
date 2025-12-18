import numpy as np

def create_learning_bounds(weights, ex, st, ih, beta):
    # Calculate initial weight sums for normalization (used in train_network)
    sum_weights_exc = np.sum(np.abs(weights[: ex, st : ih])) * beta
    sum_weights_inh = np.sum(np.abs(weights[ex : ih, st : ex])) * beta
    sum_weights_total = np.sum(np.abs(weights)) * beta

    return sum_weights_exc, sum_weights_inh, sum_weights_total