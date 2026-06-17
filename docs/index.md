# neurosnn

*A bio-inspired spiking neural network library for sleep-homeostasis research.*

---

`neurosnn` trains biologically plausible **spiking neural networks** (SNNs) on CPU. It
combines Leaky Integrate-and-Fire neurons, spike-trace STDP learning, structured
excitatory/inhibitory connectivity, and an optional **sleep-like synaptic homeostasis**
mechanism. The numeric core is JIT-compiled with [Numba](https://numba.pydata.org/), so
no GPU is required.

The package replicates and extends the model of
[Zenke et al. (2015)](https://www.nature.com/articles/ncomms7922) and is the software basis
for two research articles: a Zenke replication study and a sleep-protocol homeostasis
article.

## The 30-second tour

A full training run is four objects and a loop:

```python
import neurosnn as snn

layer = snn.Layer(N_exc=400, N_inh=100)          # an E/I population + its connectivity
learner = snn.learner.TraceSTDP()                # how synapses change
model = snn.Model(image_dataset="mnist")         # data + bookkeeping

for result in model.train(layers=[layer], learner=learner, epochs=1):
    if result.accuracy is not None:
        print(result.epoch, result.batch, result.accuracy)

print("Test accuracy:", model.test().accuracy)
```

Adding biologically-inspired **sleep** is a single extra argument:

```python
regularizer = snn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")
model.train(layers=[layer], learner=learner, regularizer=regularizer, epochs=1)
```

## How the pieces fit together

```
Model  ──▶  Layer
              ├── membrane.LIF        neuron dynamics (units: ms / mV)
              └── weights.*           connectivity density and peak amplitudes
            learner.TraceSTDP         spike-trace STDP with soft weight bound
            regularizer.Sleep         periodic noise-driven synaptic downscaling
                        .Normalize    deterministic weight rescaling (no noise)
```

`Model` owns the data and the training bookkeeping. You hand it one or more `Layer`
descriptors, a `learner`, and optionally a `regularizer`. `model.train()` returns a Python
**generator** that yields one [`TrainResult`](reference/api.md#trainresult) per batch, so you
stay in control of evaluation, logging, and plotting.

## Where to go next

New here? Read these in order:

1. **[Installation](getting-started/installation.md)** — install from PyPI.
2. **[Quickstart](getting-started/quickstart.md)** — a fully annotated training script.
3. **[Core concepts](getting-started/core-concepts.md)** — the mental model behind the API.

Want depth on a specific facet? Jump to a guide:

| Guide | What it covers |
|-------|----------------|
| [Neuron dynamics](guides/neuron-dynamics.md) | LIF membrane, spike adaptation, membrane noise |
| [Connectivity & weights](guides/connectivity.md) | E/I populations, the SE/EE/EI/IE pathways, RF factories |
| [Learning (Trace STDP)](guides/learning-stdp.md) | The spike-trace learning rule, `x_tar`, soft weight bounds |
| [Sleep & homeostasis](guides/sleep-homeostasis.md) | `Sleep` vs `Normalize`, the static/layer/neuron modes |
| [Data & input encoding](guides/data-encoding.md) | Datasets, Poisson spike coding, Gabor filtering |
| [The training loop](guides/training-loop.md) | The generator pattern, results, `validate()`/`test()` |
| [Diagnostics & metrics](guides/diagnostics.md) | The `phi` metric, PCA/LR accuracy, stat tracking |

Looking for exact signatures? See the **[API reference](reference/api.md)**. Reproducing the
papers? See **[Reproducing experiments](research/experiments.md)**.

!!! tip "Conventions used in these docs"
    Collapsible blocks like the one below appear throughout. They hold optional depth —
    equations, biological background, advanced parameters — that you can skip on a first
    read and open when you want more.

??? note "Example: open me for the extra detail"
    All time constants in `neurosnn` are in **milliseconds** and all membrane potentials are
    in **millivolts**. One simulation timestep is `dt = 1.0 ms` by default.
