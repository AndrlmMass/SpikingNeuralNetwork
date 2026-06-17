# Learning (Trace STDP)

Plastic synapses (the SE and EE pathways) are updated by **spike-trace STDP** with a
BCM-style soft weight bound. You configure it with `learner.TraceSTDP`.

## What it does

Every neuron carries a **spike trace** — a low-pass-filtered record of its recent firing
that jumps up on each spike and decays exponentially otherwise. When a postsynaptic neuron
fires, the weights of its active presynaptic partners are adjusted according to how their
traces compare to a population target `x_tar`:

- presynaptic trace **above** target → the synapse **potentiates** (grows)
- presynaptic trace **below** target → the synapse **depresses** (shrinks)

A multiplicative `(w_max − w)` term softly bounds growth so weights saturate gracefully
rather than exploding.

```python
import neurosnn as snn

learner = snn.learner.TraceSTDP(
    learning_rate=0.0004,
    tau_trace=20,
    w_max=10.0,
    mu_weight=0.6,
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `0.0008` | STDP step size |
| `tau_trace` | `25` | Spike-trace time constant (ms) |
| `w_max` | `10.0` | Soft weight bound (BCM-style) |
| `mu_weight` | `0.6` | Soft-bound exponent (higher → harder suppression near `w_max`) |
| `update_freq` | `100` | Apply STDP every N timesteps |
| `clip_weights` | `False` | Hard-clip weights to `[min, max]` after each update |
| `min_weight_exc` / `max_weight_exc` | `0.01` / `25.0` | Hard bounds for excitatory weights (if `clip_weights=True`) |
| `min_weight_inh` / `max_weight_inh` | `-25.0` / `-0.01` | Hard bounds for inhibitory weights (if `clip_weights=True`) |

## Updating only at postsynaptic spikes

The rule is applied **when a postsynaptic neuron fires**, looping over its presynaptic
partners that have a non-zero trace. This is a single-trace formulation: the sign of the
update comes from the `(x_pre − x_tar)` term rather than from separate pre- and post-spike
events. Weights are updated every `update_freq` timesteps rather than every step, which
amortises the cost of the JIT-compiled inner loop.

## What `x_tar` is

`x_tar` is a **population-mean trace** computed separately for the two plastic pathways:

- `x_tar_se` — mean spike trace across the stimulus (input) neurons
- `x_tar_ee` — mean spike trace across the excitatory neurons

Because it tracks the population's average activity, `x_tar` acts as a moving homeostatic
balance point: presynaptic neurons that fire more than their population average drive
potentiation, those below average drive depression.

??? note "The weight-update equation and its lineage"
    For a postsynaptic spike on neuron *i*, each active presynaptic neuron *j* updates:

    $$
    \Delta w_{ji} = \eta \,\bigl(x_j^{\text{pre}} - x_\text{tar}\bigr)\,\bigl(w_\text{max} - w_{ji}\bigr)^{\mu}
    $$

    where η is `learning_rate`, $x_j^{\text{pre}}$ is the presynaptic trace, `x_tar` is the
    pathway population mean, and μ is `mu_weight`.

    This corresponds to the power-law / target-trace family described by
    [Diehl & Cook (2015)](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full).
    Their paper presents four STDP variants; this implementation combines the explicit
    `x_tar` offset with a power-law soft bound and gates updates on postsynaptic spikes only.
    Unlike the original, `x_tar` here is computed dynamically as the population-mean trace
    rather than held as a fixed constant. The kernel lives in
    `neurosnn/_core/synapses.py` (`trace_STDP`, `update_x_tar`).

## See also

- [Neuron dynamics](neuron-dynamics.md) — where the spike trace is maintained.
- [Sleep & homeostasis](sleep-homeostasis.md) — the complementary, periodic weight rescaling.
