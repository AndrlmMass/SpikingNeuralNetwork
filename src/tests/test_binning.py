"""Each presented image must become exactly one feature row.

Regression test for label-change segmentation, which merged consecutive images
of the same class into one averaged sample and split an image in two where a
sleep marker (-2) fell inside it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import bin_spikes_by_label_no_breaks  # noqa: E402

STEPS = 4


def _images(labels, n_neurons=3):
    """Image k fires neuron (k mod n) on every step, so rows are identifiable."""
    spikes = np.zeros((len(labels) * STEPS, n_neurons))
    for k in range(len(labels)):
        spikes[k * STEPS:(k + 1) * STEPS, k % n_neurons] = 1
    return spikes, np.repeat(labels, STEPS)


def test_consecutive_same_label_images_stay_separate():
    spikes, labels = _images([7, 7, 7, 2])
    X, y = bin_spikes_by_label_no_breaks(spikes, labels, steps_per_sample=STEPS)
    assert X.shape[0] == 4
    np.testing.assert_array_equal(y, [7, 7, 7, 2])
    np.testing.assert_array_equal(X[1], [0, 1, 0])     # image 1 alone, not averaged


def test_sleep_marker_does_not_split_an_image():
    spikes, labels = _images([3, 5])
    labels = labels.copy()
    labels[STEPS + 1] = -2                            # sleep inserted mid-image
    X, y = bin_spikes_by_label_no_breaks(spikes, labels, steps_per_sample=STEPS)
    assert X.shape[0] == 2
    np.testing.assert_array_equal(y, [3, 5])


def test_misaligned_blocks_raise():
    spikes, labels = _images([1, 2])
    with pytest.raises(ValueError):
        bin_spikes_by_label_no_breaks(spikes[1:], labels[1:], steps_per_sample=STEPS)


def test_legacy_mode_unchanged_without_steps():
    spikes, labels = _images([7, 7, 2])
    X, _ = bin_spikes_by_label_no_breaks(spikes, labels)
    assert X.shape[0] == 2                            # old behaviour: merged
