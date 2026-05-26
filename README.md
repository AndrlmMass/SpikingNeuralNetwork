# neurosnn

*A bio-inspired spiking neural network library for sleep-homeostasis research.*

[![PyPI version](https://img.shields.io/pypi/v/neurosnn)](https://pypi.org/project/neurosnn/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## About

`neurosnn` is a Python library for training biologically plausible spiking neural networks (SNNs) with Leaky Integrate-and-Fire neurons, spike-trace STDP, and optional sleep-like synaptic homeostasis. Neuron dynamics are JIT-compiled via Numba for fast CPU simulation; no GPU is required.

The package replicates and extends the model of [Zenke et al. (2015)](https://www.nature.com/articles/ncomms7922) and is the software basis for two research articles: the Zenke et al. replication study and a sleep-protocol homeostasis article (citation to be added on publication).

---

## Installation

```bash
pip install neurosnn
```

Requires **Python ≥ 3.12**. Core dependencies (`numpy`, `numba`, `scikit-learn`, `torch`, `matplotlib`) are installed automatically.

---

## Quickstart

```python
import neurosnn as snn

# 1. Define connectivity
weights = snn.weights.random(
    density_se=0.05, density_ee=0.05, density_ei=0.05, density_ie=0.05,
    peak_se=1.0,     peak_ee=0.5,     peak_ei=1.0,     peak_ie=-0.7,
)

# 2. Build a layer (E/I population pair)
layer = snn.Layer(
    N_exc=1024,
    N_inh=225,
    membrane=snn.membrane.LIF(
        tau_m_exc=20.0,   tau_m_inh=15.0,
        tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0,
        membrane_resistance_inh=15.0,
        resting_potential=-70.0,
        reset_potential=-80.0,
        spike_threshold=-55.0,
    ),
    weights=weights,
)

# 3. Configure STDP
learner = snn.learner.TraceSTDP(
    learning_rate=0.0004,
    tau_trace=20,
    w_max=10.0,
    mu_weight=0.6,
)

# 4. Configure the model (data + training bookkeeping)
model = snn.Model(
    input_size=784,
    classes=list(range(10)),
    num_steps=350,
    image_dataset="mnist",
    all_images_train=5000, batch_image_train=1000,
    all_images_val=1000,   batch_image_val=1000,
    all_images_test=1000,  batch_image_test=1000,
)

# 5. Training loop — model.train() is a generator
for result in model.train(layers=[layer], learner=learner, epochs=3):
    if result.batch % 5 == 0 and result.accuracy is not None:
        val = model.validate()
        print(
            f"epoch {result.epoch + 1}  batch {result.batch}"
            f"  train {result.accuracy:.3f}  val {val.accuracy:.3f}"
        )

print(f"Test accuracy: {model.test().accuracy:.3f}")
```

Adding **sleep regularisation** requires only one extra argument:

```python
regularizer = snn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")

for result in model.train(layers=[layer], learner=learner,
                          regularizer=regularizer, epochs=3):
    ...
```

---

## Core concepts

`model.train()` returns a Python generator. Each iteration processes one training batch and yields a `TrainResult`; `model.validate()` can be called at any point inside the loop without resetting state. Weights are built from the `Layer` spec on the **first** generator iteration, so inspection or plotting before training starts is possible after calling `next()` once.

```
Model  ──▶  Layer
              ├── membrane.LIF        neuron dynamics (all units: ms / mV)
              └── weights.*           connectivity density and peak amplitudes
            learner.TraceSTDP         spike-trace STDP with soft weight bound
            regularizer.Sleep         periodic noise-driven synaptic downscaling
                        .Normalize    deterministic weight rescaling (no noise)
```

---

## API Reference

### `Model`

```python
snn.Model(
    input_size: int = 225,           # flattened input pixels (784 for MNIST)
    classes: list = None,            # class labels; defaults to [0..9]
    random_state: int = 0,
    num_steps: int = 350,            # timesteps per image presentation (ms)
    all_images_train: int = 1000,    # total training images per epoch
    batch_image_train: int = 100,    # images per training batch
    all_images_val: int = 200,
    batch_image_val: int = 100,
    all_images_test: int = 200,
    batch_image_test: int = 100,
    image_dataset: str = "mnist",    # see Supported datasets
    max_rate_hz: float = 90.0,       # peak Poisson rate for input encoding
    gain: float = 1.0,               # global input gain
    gabor: bool = False,             # apply Gabor filter to inputs
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `train(layers, learner=None, regularizer=None, epochs=1, ...)` | `Generator[TrainResult]` | Build network and stream training batches |
| `validate()` | `EvalResult` | Evaluate on validation split (call inside training loop) |
| `test()` | `EvalResult` | Evaluate on test split after training |

---

### `Layer`

```python
snn.Layer(
    N_exc: int = 400,                   # excitatory neuron count
    N_inh: int = 100,                   # inhibitory neuron count
    membrane: LIF = LIF(),              # membrane dynamics spec
    weights: WeightsSpec = receptive_fields(),  # connectivity spec
)
```

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
| `duration` | `300` | Timesteps per sleep episode |
| `frequency` | `1050` | Timesteps between sleep episodes |
| `mode` | `"static"` | Target restoration mode (see above) |

**`Normalize`** parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency` | `1050` | Timesteps between normalisation events |
| `mode` | `"static"` | Target restoration mode (see above) |

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

## Citation

If you use `neurosnn` in your work, please cite:

**Baseline model:**
> Zenke, F., Gerstner, W. & Ganguli, S. (2015). The temporal paradox of Hebbian learning and homeostatic plasticity. *Nature Communications*, 6, 6922. https://doi.org/10.1038/ncomms7922

**Sleep-homeostasis article** *(citation to be added on publication)*

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
