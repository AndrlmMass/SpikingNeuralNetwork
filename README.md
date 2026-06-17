# neurosnn

*A bio-inspired spiking neural network library for sleep-homeostasis research.*

[![PyPI version](https://img.shields.io/pypi/v/neurosnn)](https://pypi.org/project/neurosnn/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

`neurosnn` trains biologically plausible spiking neural networks (SNNs) with Leaky
Integrate-and-Fire neurons, spike-trace STDP, and optional sleep-like synaptic homeostasis.
Neuron dynamics are JIT-compiled via Numba for fast CPU simulation — no GPU required.

The package replicates and extends the model of
[Zenke et al. (2015)](https://www.nature.com/articles/ncomms7922) and is the software basis
for a Zenke replication study and a sleep-protocol homeostasis article.

## 📖 Documentation

**Full documentation lives at the [neurosnn docs site](https://andrlmmass.github.io/SpikingNeuralNetwork/).**

The docs are organised like a book:

- **Getting started** — [Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [Core concepts](docs/getting-started/core-concepts.md)
- **Guides** — [Neuron dynamics](docs/guides/neuron-dynamics.md) · [Connectivity & weights](docs/guides/connectivity.md) · [Learning (Trace STDP)](docs/guides/learning-stdp.md) · [Sleep & homeostasis](docs/guides/sleep-homeostasis.md) · [Data & encoding](docs/guides/data-encoding.md) · [Training loop](docs/guides/training-loop.md) · [Diagnostics](docs/guides/diagnostics.md)
- **Reference** — [API reference](docs/reference/api.md) · [Datasets](docs/reference/datasets.md) · [Glossary](docs/reference/glossary.md)
- **Research** — [Background & citation](docs/research/background.md) · [Reproducing experiments](docs/research/experiments.md)

## Installation

```bash
pip install neurosnn
```

Requires **Python ≥ 3.12**. Core dependencies (`numpy`, `numba`, `scikit-learn`, `torch`,
`matplotlib`) are installed automatically. See the
[installation guide](docs/getting-started/installation.md) for details.

## Quickstart

```python
import neurosnn as snn

layer = snn.Layer(N_exc=400, N_inh=100)      # an E/I population + its connectivity
learner = snn.learner.TraceSTDP()            # how synapses change
model = snn.Model(image_dataset="mnist")     # data + bookkeeping

# model.train() is a generator yielding one TrainResult per batch
for result in model.train(layers=[layer], learner=learner, epochs=1):
    if result.accuracy is not None:
        print(result.epoch, result.batch, result.accuracy)

print("Test accuracy:", model.test().accuracy)
```

Adding sleep-like homeostasis is one extra argument:

```python
regularizer = snn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")
model.train(layers=[layer], learner=learner, regularizer=regularizer, epochs=1)
```

See the [Quickstart](docs/getting-started/quickstart.md) for a fully annotated example.

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve   # http://127.0.0.1:8000
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
