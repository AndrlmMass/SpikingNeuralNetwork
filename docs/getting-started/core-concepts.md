# Core concepts

This page builds the mental model behind the API. Once these five ideas click, the rest of
the library reads naturally.

## 1. Four objects

A training run is assembled from four user-facing objects:

| Object | Role | Built with |
|--------|------|-----------|
| `Model` | Owns the dataset and training bookkeeping; runs the simulation | `snn.Model(...)` |
| `Layer` | One excitatory/inhibitory population pair and its connectivity | `snn.Layer(...)` |
| `learner` | The synaptic learning rule | `snn.learner.TraceSTDP(...)` |
| `regularizer` | Optional periodic weight homeostasis | `snn.regularizer.Sleep(...)` / `.Normalize(...)` |

`Model` is the orchestrator. You pass the other three into `model.train()`.

## 2. A Layer is an E/I population

Each `Layer` holds two neuron populations — **excitatory** (`N_exc`) and **inhibitory**
(`N_inh`) — plus a [`membrane.LIF`](../guides/neuron-dynamics.md) spec describing their
dynamics and a [`weights`](../guides/connectivity.md) spec describing their connectivity.

The input (stimulus) neurons are not part of the `Layer`; their count comes from
`Model(input_size=...)`. Connections are organised into four **pathways**:

| Pathway | Meaning | Sign | Plastic? |
|---------|---------|------|----------|
| **SE** | stimulus → excitatory | + | learned via STDP |
| **EE** | excitatory → excitatory | + | learned via STDP |
| **EI** | excitatory → inhibitory | + | fixed |
| **IE** | inhibitory → excitatory | − | fixed |

!!! note "Single-layer for now"
    `model.train()` currently accepts a list of layers but enforces exactly one. Multi-layer
    stacking is not yet supported.

## 3. train() is a generator

`model.train(...)` does not block until finished and return a history. It returns a Python
**generator** that yields one [`TrainResult`](../reference/api.md#trainresult) after every
batch:

```python
for result in model.train(layers=[layer], learner=learner, epochs=3):
    # runs once per batch — you decide what to do here
    ...
```

This puts you in control of the cadence of evaluation, logging, and plotting. You can call
`model.validate()` from inside the loop without disturbing training state, and
`model.test()` after it. The underlying simulation state persists across epochs, so the loop
resumes seamlessly.

## 4. Units are biological

Everything is expressed in biophysical units, not abstract ones:

- **Time** is in milliseconds. One timestep is `dt = 1.0 ms` by default, and `num_steps`
  controls how many milliseconds each image is presented for.
- **Membrane potentials** are in millivolts (resting ≈ −70 mV, threshold ≈ −55 mV).

This makes parameters directly comparable to the neuroscience literature.

## 5. Learning, then homeostasis

Two distinct mechanisms shape the weights:

- The **learner** ([Trace STDP](../guides/learning-stdp.md)) changes synapses continuously
  based on spike timing — this is where representations are formed.
- The **regularizer** ([Sleep / Normalize](../guides/sleep-homeostasis.md)) periodically
  rescales weights to counteract runaway growth — this is homeostasis, applied every
  `frequency` timesteps rather than continuously.

You can train with a learner alone; the regularizer is optional and orthogonal.

## Where to go next

You now have the mental model. Pick a facet from the [Guides](../guides/neuron-dynamics.md)
or browse exact signatures in the [API reference](../reference/api.md).
