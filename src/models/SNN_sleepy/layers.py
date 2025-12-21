"""
Network layer creation and weight initialization.

This module provides functions for creating the SNN architecture:
- Weight matrix initialization with excitatory/inhibitory structure
- Array allocation for membrane potentials and spikes
"""

import numpy as np


def create_weights(
    N_exc,
    N_inh,
    N_x,
    N,
    w_dense_ee,
    w_dense_ei,
    w_dense_ie,
    w_dense_se,
    se_weights,
    ee_weights,
    ei_weights,
    ie_weights,
    rng,
):
    """
    Create weight matrix for the SNN with excitatory and inhibitory connections.
    
    Parameters
    ----------
    N_exc : int
        Number of excitatory neurons
    N_inh : int
        Number of inhibitory neurons
    N_x : int
        Number of input (stimulus) neurons
    N : int
        Total number of neurons
    w_dense_* : float
        Connection density (probability) for each connection type
    *_weights : float
        Weight values for each connection type
    rng : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    weights : ndarray
        Weight matrix of shape (N, N)
    """
    weights = np.zeros(shape=(N, N))

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory

    # Create weights based on affinity rates
    mask_ee = rng.random((N_exc, N_exc)) < w_dense_ee
    mask_ei = rng.random((N_exc, N_inh)) < w_dense_ei
    mask_ie = rng.random((N_inh, N_exc)) < w_dense_ie
    mask_se = rng.random((N_x, N_exc)) < w_dense_se

    # input poisson weights
    weights[:st, st:ex][mask_se] = se_weights

    # hidden excitatory weights
    weights[st:ex, st:ex][mask_ee] = ee_weights
    weights[st:ex, ex:ih][mask_ei] = ei_weights

    # hidden inhibitory weights
    weights[ex:ih, st:ex][mask_ie] = ie_weights

    # remove excitatory self-connecting (diagonal) weights
    np.fill_diagonal(weights[st:ex, st:ex], 0)

    # remove recurrent connections from exc to inh
    inh_mask = weights[st:ex, ex:ih].T != 0
    weights[ex:ih, st:ex][inh_mask] = 0

    return weights


def create_arrays(
    N,
    st,
    ih,
    spike_threshold_default,
    resting_membrane,
    runtime,
    data,
):
    """
    Create arrays for membrane potentials and spikes.
    
    Parameters
    ----------
    N : int
        Total number of neurons
    resting_membrane : float
        Resting membrane potential (mV)
    runtime : int
        Total runtime timesteps
    data : ndarray or None
        Input spike data
        
    Returns
    -------
    tuple
        (mp, spikes)
    """

    # create membrane potential arrays
    membrane_potential = np.zeros((runtime, ih - st))
    membrane_potential[0] = resting_membrane

    spikes = None
    if data is not None and runtime > 0:
        spikes = np.zeros((runtime, N), dtype=np.int8)
        spikes[:, :st] = data

    # create missing arrays
    I_syn = np.zeros(N - st)
    spike_times = np.zeros(N)
    a = np.zeros(N - st)

    # create spike threshold array
    spike_threshold = np.full(
        shape=(ih - st),
        fill_value=spike_threshold_default,
        dtype=float,
    )

    return (
        membrane_potential,
        spikes,
        I_syn, 
        spike_times, 
        a,
        spike_threshold,
        )


