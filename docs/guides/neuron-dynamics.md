# Neuron dynamics

Neurons in `neurosnn` are **Leaky Integrate-and-Fire (LIF)** units with optional
spike-frequency adaptation and optional membrane noise. You configure them through a
`membrane.LIF` spec attached to each [`Layer`](connectivity.md).

## What it does

Each neuron integrates incoming synaptic current onto its membrane potential, which leaks
back toward rest over time. When the potential crosses threshold, the neuron emits a spike
and resets. Excitatory and inhibitory populations have independent time constants and
resistances, so you can give them distinct temporal profiles.

```python
import neurosnn as snn

layer = snn.Layer(
    N_exc=400, N_inh=100,
    membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0,
        resting_potential=-70.0,
        spike_threshold=-55.0,
        reset_potential=-80.0,
    ),
)
```

## Parameters

All time constants are in **ms**; all potentials are in **mV**.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_m_exc` / `tau_m_inh` | `30.0` | Membrane time constant (E / I) |
| `tau_syn_exc` / `tau_syn_inh` | `30.0` | Synaptic current time constant (E / I) |
| `membrane_resistance_exc` / `..._inh` | `30.0` | Membrane resistance (E / I) |
| `resting_potential` | `-70.0` | Resting membrane potential |
| `reset_potential` | `-80.0` | Post-spike reset potential |
| `spike_threshold` | `-55.0` | Spike threshold |
| `min_mp` / `max_mp` | `-100.0` / `40.0` | Hard clamp on membrane potential |
| `mean_noise` / `var_noise` | `0.0` / `1.0` | Gaussian membrane noise (μ, σ²) |
| `spike_adaptation` | `True` | Enable spike-frequency adaptation |
| `tau_adaptation` | `100.0` | Adaptation time constant |
| `delta_adaptation` | `1.0` | Per-spike adaptation increment |

## Spike-frequency adaptation

When `spike_adaptation=True`, each neuron carries an adaptation variable `a` that raises its
effective threshold after every spike and decays back over `tau_adaptation`. This creates a
relative refractory period: neurons that have just fired are briefly harder to re-excite,
which discourages a few units from dominating the response.

## Membrane noise

`mean_noise` and `var_noise` define a Gaussian perturbation added to the membrane potential.
In normal training this is typically left off; it is injected during **sleep** episodes to
drive spontaneous activity (see [Sleep & homeostasis](sleep-homeostasis.md)).

??? note "The update equations"
    Per timestep `dt`, the synaptic current and membrane potential evolve as:

    $$
    \Delta I_\text{syn} = \frac{-I_\text{syn} + \text{drive}}{\tau_\text{syn}}\, dt
    $$

    $$
    \Delta V_m = \frac{-(V_m - V_\text{rest}) + R_m\, I_\text{syn}}{\tau_m}\, dt
    $$

    where `drive` is the summed weight of presynaptic neurons that spiked this step. The
    potential is then clamped to `[min_mp, max_mp]`. If `V_m > threshold`, the neuron spikes
    and `V_m` is set to `reset_potential`.

    With adaptation enabled, the threshold is `threshold = threshold_default + a`, where

    $$
    \Delta a = -\frac{a}{\tau_\text{adaptation}}\, dt, \qquad a \mathrel{+}= \delta_\text{adaptation}\ \text{on each spike.}
    $$

    These updates live in `neurosnn/_core/neurons.py` (`update_membrane_potential`,
    `update_spikes`) and are JIT-compiled with Numba.

## See also

- [Connectivity & weights](connectivity.md) — what `drive` is summed from.
- [The training loop](training-loop.md) — how per-neuron state can be tracked via `track_stats`.
