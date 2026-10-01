"""The timing STDP rule must be Hebbian: pre-before-post strengthens a synapse,
post-before-pre weakens it.

Regression test for the sign inversion in spike_timing: spike_times holds the
time SINCE each neuron's last spike, and dt was computed as t_post - t_pre, so
causal pairs were depressed and anti-causal pairs potentiated.

"Strengthen" means a larger magnitude: more positive for an excitatory synapse,
more negative for an inhibitory one.
"""
import os
import sys

import numpy as np
import pytest
from numba.typed import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weight_funcs import spike_timing  # noqa: E402


def _dw(pre_kind, t_pre, t_post, ltd_scale=1.0):
    """Weight change on one synapse onto post neuron 1.

    Layout follows the model: [input | excitatory | inhibitory]. Neuron 0 is an
    input (excitatory pre), neuron 1 the excitatory post, neuron 2 inhibitory.
    """
    w = np.zeros((3, 3))
    pre = 0 if pre_kind == "exc" else 2
    w[pre, 1] = 0.5 if pre_kind == "exc" else -0.5
    idx = List()
    idx.append(np.array([pre], dtype=np.int64))   # pres of neuron 1
    idx.append(np.array([], dtype=np.int64))      # pres of neuron 2
    st = np.full(3, 1000.0)                       # others silent for ages
    st[pre], st[1] = t_pre, t_post
    sp = (st == 0).astype(np.int64)
    out = spike_timing(spike_times=st, tau_LTP=10.0, tau_LTD=10.0,
                       learning_rate_exc=0.01, learning_rate_inh=0.01, N_inh=1,
                       weights=w.copy(), N_x=1, spikes=sp, nonzero_pre_idx=idx,
                       ltd_scale=ltd_scale)
    return out[pre, 1] - w[pre, 1]


@pytest.mark.parametrize("pre_kind", ["exc", "inh"])
def test_causal_pairing_strengthens(pre_kind):
    dw = _dw(pre_kind, t_pre=5, t_post=0)          # pre fired 5 steps ago
    assert (dw > 0) if pre_kind == "exc" else (dw < 0)


@pytest.mark.parametrize("pre_kind", ["exc", "inh"])
def test_anticausal_pairing_weakens(pre_kind):
    dw = _dw(pre_kind, t_pre=0, t_post=5)          # post fired 5 steps ago
    assert (dw < 0) if pre_kind == "exc" else (dw > 0)


def test_closer_pairs_change_more():
    assert abs(_dw("exc", 2, 0)) > abs(_dw("exc", 8, 0))


def test_ltd_scale_scales_only_depression():
    assert _dw("exc", 5, 0, ltd_scale=0.3) == pytest.approx(_dw("exc", 5, 0))        # LTP unchanged
    assert _dw("exc", 0, 5, ltd_scale=0.3) == pytest.approx(0.3 * _dw("exc", 0, 5))  # LTD scaled
