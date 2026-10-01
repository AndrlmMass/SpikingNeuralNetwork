"""Spikes inside a sleep episode must come only from that step's threshold
crossings.

Regression test for the "sticky spikes" bug: each sleep step started from a
copy of the previous step's spike vector, and update_spikes only ever sets
spikes to 1, so spikes accumulated. In a real MNIST run every one of the 250
network neurons read as firing on every step from ~50 steps into each episode.

The fast tests check the helper that builds each sleep step's spike vector.
The slow test runs the real training path for one sleep episode and checks the
invariant on every call; enable it with SNN_SLOW_TESTS=1 (takes a few minutes).
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from train import sleep_step_spikes  # noqa: E402

ST, IH = 6, 10  # 6 input neurons, 4 network neurons


def test_network_spikes_are_cleared_every_step():
    prev = np.ones(IH, dtype=np.int8)          # everything fired last step
    cur = sleep_step_spikes(prev, ST)
    assert cur.dtype == prev.dtype and cur.shape == prev.shape
    assert not cur[ST:IH].any()


def test_suppressed_input_is_zero():
    prev = np.ones(IH, dtype=np.int8)
    assert not sleep_step_spikes(prev, ST, held_input=None)[:ST].any()


def test_held_input_replays_the_image_cyclically():
    rng = np.random.default_rng(0)
    held = rng.integers(0, 2, size=(3, ST)).astype(np.int8)  # 3-step "image"
    prev = np.ones(IH, dtype=np.int8)
    for k in range(7):
        cur = sleep_step_spikes(prev, ST, held_input=held, sleep_iter=k)
        np.testing.assert_array_equal(cur[:ST], held[k % 3])
        assert not cur[ST:IH].any()


def test_previous_vector_is_not_modified():
    prev = np.ones(IH, dtype=np.int8)
    sleep_step_spikes(prev, ST)
    assert prev.all()


@pytest.mark.skipif(os.environ.get("SNN_SLOW_TESTS") != "1",
                    reason="runs a real training episode; set SNN_SLOW_TESTS=1")
@pytest.mark.parametrize("components", ["downscale,noise,stdp,suppress",
                                        "downscale,noise,stdp"])
def test_real_sleep_episode_starts_every_step_from_zero(components, monkeypatch):
    """Wrap update_spikes in the real training path and check, for every
    call inside the first sleep episode, that no network spike is carried in.
    Sleep calls receive a freshly built vector (no .base); wake calls receive a
    row view of the batch array."""
    import runpy
    import train

    repo = os.path.dirname(SRC)
    orig = train.update_spikes
    seen = {"sleep": 0, "max_after": 0, "input_on": 0}

    class Done(Exception):
        pass

    def wrapped(**kw):
        st, ih, sp = kw["st"], kw["ih"], kw["spikes"]
        is_sleep = sp.base is None
        if is_sleep:
            assert not sp[st:ih].any(), "network spikes carried into a sleep step"
            seen["input_on"] += int(sp[:st].any())
        out = orig(**kw)
        if is_sleep:
            seen["sleep"] += 1
            seen["max_after"] = max(seen["max_after"], int(out[1][st:ih].sum()))
            if seen["sleep"] == 350:
                raise Done
        return out

    monkeypatch.setattr(train, "update_spikes", wrapped)
    monkeypatch.chdir(repo)
    tag = "_test_sleep_spikes_" + components.replace(",", "_")
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--dataset", "mnist", "--seed", "42", "--runs", "1",
        "--out-tag", tag, "--num-steps", "100", "--check-sleep-interval", "3500",
        "--clip-always", "--no-plots", "--reg-method", "sleep",
        "--sleep-rate", "0.1", "--sleep-decay-rate", "0.9999576013637494",
        "--sleep-max-iters", "35000", "--on-timeout", "give_up",
        "--sleep-termination", "band", "--sleep-components", components])
    try:
        with pytest.raises(Done):
            runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")
    finally:
        partial = os.path.join(repo, "results", f"results_{tag}.json")
        if os.path.exists(partial):
            os.remove(partial)
    assert seen["sleep"] == 350
    # a real episode, not all-or-nothing activity
    assert seen["max_after"] < 250
    if "suppress" in components:
        assert seen["input_on"] == 0
    else:
        assert seen["input_on"] > 0
