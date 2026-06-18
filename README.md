# neurosnn

*A bio-inspired spiking neural network library for sleep-homeostasis research.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](https://pypi.org/project/neurosnn/)

---

`neurosnn` is biologically-inspired spiking neural network (SNN) package. It leverages No-Python Just-In-Time (**NJIT**) C-compiled parallel CPUs to achieve efficient runtime on projects at scale (4000 it/s) with planned multi-layer extensions. Users can leverage Leaky Fire-and-Integrate (**LIF**) neurons, adaptive spiking thresholds, parallel runtime plotting of network activity, two regularization modes (napping or normalization) and trace-based STDP. 

The package was originally inspired by the clustering-benefits observed by unsupervised SNNs developed by
[Zenke et al. (2015)](https://www.nature.com/articles/ncomms7922) and later more closely [Diehl et al. (2015)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full?ref=https://githubhelp.com), however the current implementation is a standalone library that provides more options and less rigid deployments compared to the aforementioned studies.

## 📖 Documentation

**Full documentation lives at the [neurosnn docs site](https://andrlmmass.github.io/SpikingNeuralNetwork/).**

The docs are organised like a book:

- **Getting started** — [Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [Core concepts](docs/getting-started/core-concepts.md)
- **Guides** — [Neuron dynamics](docs/guides/neuron-dynamics.md) · [Connectivity & weights](docs/guides/connectivity.md) · [Learning (Trace STDP)](docs/guides/learning-stdp.md) · [Sleep & homeostasis](docs/guides/sleep-homeostasis.md) · [Data & encoding](docs/guides/data-encoding.md) · [Training loop](docs/guides/training-loop.md) · [Diagnostics](docs/guides/diagnostics.md)
- **Reference** — [API reference](docs/reference/api.md) · [Datasets](docs/reference/datasets.md) · [Glossary](docs/reference/glossary.md)
- **Research** — [Background & citation](docs/research/background.md) · [Reproducing experiments](docs/research/experiments.md)

## Installation
> [!CAUTION]
> **DISCLAIMER:** The project has not been submitted as a PyPi yet and is not ready for production use. Use at own risk. 


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

---

### `membrane.LIF`

All time constants in **ms**, all potentials in **mV**.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_m_exc` / `tau_m_inh` | `30.0` | Membrane time constant (E / I) |
| `tau_syn_exc` / `tau_syn_inh` | `30.0` | Synaptic time constant (E / I) |
| `membrane_resistance_exc` / `..._inh` | `30.0` | Membrane resistance (E / I) |
| `resting_potential` | `-70.0` | Resting membrane potential |
| `reset_potential` | `-80.0` | Post-spike reset potential |
| `spike_threshold` | `-55.0` | Spike threshold |
| `min_mp` / `max_mp` | `-100.0` / `40.0` | Hard clamp on membrane potential |
| `mean_noise` / `var_noise` | `0.0` / `1.0` | Gaussian membrane noise (μ, σ²) |
| `spike_adaptation` | `True` | Enable spike-frequency adaptation |
| `tau_adaptation` | `100.0` | Adaptation time constant |
| `delta_adaptation` | `1.0` | Per-spike adaptation increment |

---

### `weights` — factory functions

All three factories accept the same **shared parameters**:

| Parameter | Description |
|-----------|-------------|
| `density_se/ee/ei/ie: float` | Structural connection probability for each pathway |
| `peak_se/ee/ei/ie: float` | Peak initial weight magnitude (inhibitory peaks are negative) |

**`snn.weights.random(...)`** — uniformly sparse random connectivity; no spatial structure.

**`snn.weights.receptive_fields(..., rf_scale=1.0)`** — Gaussian/Mexican-hat structured RFs; `rf_scale` scales all spatial sigmas globally.

**`snn.weights.oriented_receptive_fields(...)`** — oriented elliptical Gaussian RFs for the S→E pathway; additional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_x` | `3.0` | Major-axis sigma of each RF (pixels) |
| `gamma` | `0.4` | Aspect ratio σ_y / σ_x; 1.0 = isotropic |
| `n_orientations` | `4` | Number of orientation groups cycling across E neurons |
| `r_cut_factor` | `3.0` | Hard elliptical cutoff at `r_cut_factor × σ` |
| `sigma_x_lognormal_std` | `0.0` | Log-normal size diversity; 0 = uniform |

---

### `learner.TraceSTDP`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `0.0008` | STDP step size |
| `tau_trace` | `25` | Spike-trace time constant (ms) |
| `w_max` | `10.0` | Soft weight bound (BCM-style) |
| `mu_weight` | `0.6` | BCM softness exponent |
| `update_freq` | `100` | Apply STDP every N timesteps |
| `clip_weights` | `False` | Hard-clip weights to [min, max] after each update |
| `min_weight_exc` / `max_weight_exc` | `0.01` / `25.0` | Hard bounds for excitatory weights (if `clip_weights=True`) |
| `min_weight_inh` / `max_weight_inh` | `-25.0` / `-0.01` | Hard bounds for inhibitory weights (if `clip_weights=True`) |

---

### `regularizer.Sleep` / `regularizer.Normalize`

Both support three **`mode`** values:

| Mode | Behaviour |
|------|-----------|
| `"static"` | Restore to fixed target sums computed at initialisation |
| `"layer"` | Restore total synaptic drive across all neurons in the layer |
| `"neuron"` | Restore per-neuron incoming synaptic drive (strongest homeostasis) |

**`Sleep`** parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | `50 ms` | Timesteps per sleep episode |
| `frequency` | `1050 ms` | Timesteps between sleep episodes |
| `mode` | `"neuron"` | Target restoration mode (see above) |

**`Normalize`** parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency` | `1050 ms` | Timesteps between normalisation events |
| `mode` | `"neuron"` | Target restoration mode (see above) |

---

### `TrainResult`

Yielded after every training batch.

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Current epoch index (0-based) |
| `batch` | `int` | Batch index within epoch |
| `weights` | `np.ndarray` | Live weight matrix reference (shape N × N) |
| `accuracy` | `float \| None` | Training batch accuracy (None until first evaluation) |
| `phi` | `float \| None` | Phi clustering metric (None until first evaluation) |
| `spikes` | `np.ndarray \| None` | Mean firing rates shape (n_images, N_exc), only if `return_spikes=True` |
| `stats` | `dict \| None` | Neuron/synapse diagnostics, only if `track_stats=True` |

### `EvalResult`

Returned by `model.validate()` and `model.test()`.

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | `float \| None` | Classification accuracy |
| `phi` | `float \| None` | Phi clustering metric |
| `split` | `str` | `"val"` or `"test"` |
| `spikes` | `np.ndarray \| None` | Mean firing rates, only if `return_spikes=True` |

---

## Supported datasets

| Key | Dataset |
|-----|---------|
| `"mnist"` | MNIST handwritten digits |
| `"kmnist"` | Kuzushiji-MNIST |
| `"fmnist"` / `"fashionmnist"` | Fashion-MNIST |
| `"notmnist"` | notMNIST letters A–J |
| `"geomfig"` | Geometric figures (toy dataset; fast for local testing) |

Standard datasets are downloaded automatically on first use via `torchvision`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
