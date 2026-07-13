"""
Unit tests for the reward-modulated STDP kernel (milestone 1) — mechanics only,
no network run. Verifies:
  1. count-product eligibility gives the exact expected Δw,
  2. flipping the reward sign flips the weight-change direction,
  3. a silent post-neuron (count 0) gets no update,
  4. RewardLearner accumulates spike counts across timesteps and resets after step,
  5. target-class synapses potentiate while non-target depress in one step.

With mu_weight=0 the soft bound is 1.0 (x**0 == 1 for x>0), so Δw = lr*reward*e
exactly and we can assert precise values.

Run:  python experiments/RF_article/rstdp/test_reward_kernel.py
      (or: pytest experiments/RF_article/rstdp/test_reward_kernel.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from neurosnn._core.synapses import reward_STDP, RewardLearner


# Toy topology: 2 input (0,1) -> 2 exc (2,3), fully connected SE. No inhibition.
N_X, N_EXC = 2, 2
N_TOTAL = N_X + N_EXC
PRE_IDX = np.array([[0, 1], [0, 1]], dtype=np.int64)  # pres of exc neuron 2 and 3
LR, W_MAX, W_MIN, MU = 0.1, 10.0, 0.0, 0.0             # mu=0 -> bound == 1.0


def base_weights():
    w = np.zeros((N_TOTAL, N_TOTAL), dtype=np.float64)
    for j in (0, 1):
        for i in (2, 3):
            w[j, i] = 1.0
    return w


def test_eligibility_exact():
    w = base_weights()
    # pre0=3, pre1=1 spikes; post2=2 spikes, post3=0 spikes
    spike_count = np.array([3.0, 1.0, 2.0, 0.0])
    reward_post = np.array([1.0, 1.0])  # both targets, +1
    reward_STDP(LR, spike_count, reward_post, w, N_X, PRE_IDX, W_MAX, W_MIN, MU)
    # i=2: e(0,2)=3*2=6 -> Δ=0.1*1*6=0.6 ; e(1,2)=1*2=2 -> Δ=0.2
    assert np.isclose(w[0, 2], 1.6), w[0, 2]
    assert np.isclose(w[1, 2], 1.2), w[1, 2]
    # i=3 silent (count 0) -> untouched
    assert np.isclose(w[0, 3], 1.0) and np.isclose(w[1, 3], 1.0)
    print("ok  eligibility exact")


def test_sign_flip():
    w_pos = base_weights()
    w_neg = base_weights()
    spike_count = np.array([3.0, 0.0, 2.0, 0.0])
    reward_STDP(LR, spike_count, np.array([+1.0, 0.0]), w_pos, N_X, PRE_IDX, W_MAX, W_MIN, MU)
    reward_STDP(LR, spike_count, np.array([-1.0, 0.0]), w_neg, N_X, PRE_IDX, W_MAX, W_MIN, MU)
    assert w_pos[0, 2] > 1.0, w_pos[0, 2]           # potentiated
    assert w_neg[0, 2] < 1.0, w_neg[0, 2]           # depressed
    # symmetric magnitude at mu=0
    assert np.isclose(w_pos[0, 2] - 1.0, 1.0 - w_neg[0, 2])
    print("ok  sign flip")


def test_silent_post_no_update():
    w = base_weights()
    spike_count = np.array([5.0, 5.0, 0.0, 0.0])     # both post neurons silent
    reward_STDP(LR, spike_count, np.array([1.0, -1.0]), w, N_X, PRE_IDX, W_MAX, W_MIN, MU)
    assert np.allclose(w, base_weights())
    print("ok  silent post -> no update")


def test_learner_accumulate_and_reset():
    learner = RewardLearner(
        learning_rate=LR, N_x=N_X, N_exc=N_EXC,
        nonzero_pre_idx=[[0, 1], [0, 1]],
        neuron_class=np.array([0, 1]),               # exc2->class0, exc3->class1
        w_max=W_MAX, w_min=W_MIN, mu_weight=MU, baseline_decay=0.0,  # baseline stays 0
    )
    w = base_weights()
    # three timesteps: input0 fires each step (3), input1 never; exc2 fires twice, exc3 once
    steps = [
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 1, 1]),
        np.array([1, 0, 0, 0]),
    ]
    for s in steps:
        learner.accumulate(s)
    assert np.array_equal(learner.spike_count, [3, 0, 2, 1]), learner.spike_count

    w = learner.step(w, target_label=0)  # class 0 is target -> exc2 potentiate, exc3 depress
    # exc2 (target): e(0,2)=3*2=6 -> +0.6 ; e(1,2)=0 -> none
    assert np.isclose(w[0, 2], 1.6), w[0, 2]
    assert np.isclose(w[1, 2], 1.0), w[1, 2]
    # exc3 (non-target): e(0,3)=3*1=3 -> reward -1 -> -0.3
    assert np.isclose(w[0, 3], 0.7), w[0, 3]
    # counts reset after step
    assert np.array_equal(learner.spike_count, [0, 0, 0, 0])
    print("ok  learner accumulate + reset + target/non-target directions")


def main():
    test_eligibility_exact()
    test_sign_flip()
    test_silent_post_no_update()
    test_learner_accumulate_and_reset()
    print("\nAll reward-STDP kernel unit tests passed.")


if __name__ == "__main__":
    main()
