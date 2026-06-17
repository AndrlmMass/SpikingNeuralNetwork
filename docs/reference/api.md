# API reference

The public API is everything exported from the top-level `neurosnn` package:

```python
import neurosnn as snn
# snn.Model, snn.Layer, snn.TrainResult, snn.EvalResult
# snn.membrane, snn.weights, snn.learner, snn.regularizer
```

---

## `Model`

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
| `train(layers, learner=None, regularizer=None, epochs=1, ...)` | `Generator[TrainResult]` | Build the network and stream training batches |
| `validate()` | `EvalResult` | Evaluate on the validation split (callable inside the training loop) |
| `test()` | `EvalResult` | Evaluate on the test split after training |

See [The training loop](../guides/training-loop.md) for the full `train()` argument list.

---

## `Layer`

```python
snn.Layer(
    N_exc: int = 400,                            # excitatory neuron count
    N_inh: int = 100,                            # inhibitory neuron count
    membrane: LIF = LIF(),                       # membrane dynamics spec
    weights: WeightsSpec = receptive_fields(),   # connectivity spec
)
```

---

## `membrane.LIF`

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

Details: [Neuron dynamics](../guides/neuron-dynamics.md).

---

## `weights` factory functions

Shared parameters across all three factories:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `density_se` / `ee` / `ei` / `ie` | `0.05` / `0.01` / `0.05` / `0.05` | Connection probability per pathway |
| `peak_se` / `ee` / `ei` / `ie` | `0.1` / `0.3` / `0.3` / `-0.2` | Peak initial weight (IE negative) |

**`snn.weights.random(...)`** — uniformly sparse random connectivity; no spatial structure.

**`snn.weights.receptive_fields(..., rf_scale=1.0, sigma_ee_mean=0.0, sigma_ee_lognormal_std=0.0, sigma_se_mean=0.0, sigma_se_lognormal_std=0.0)`** — Gaussian / Mexican-hat structured RFs.

**`snn.weights.oriented_receptive_fields(...)`** — oriented elliptical Gaussian RFs for SE. Additional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_x` | `3.0` | Major-axis sigma of each RF (pixels) |
| `gamma` | `0.4` | Aspect ratio σ_y / σ_x; `1.0` = isotropic |
| `n_orientations` | `4` | Number of orientation groups |
| `r_cut_factor` | `3.0` | Hard elliptical cutoff at `r_cut_factor × σ` |
| `sigma_x_lognormal_std` | `0.0` | Log-normal size diversity; `0` = uniform |
| `sigma_x_lognormal_max` | `0.0` | Upper clip on RF size; `0` = no clip |
| `orientation_mode` | `"block"` | `"block"` or `"interleaved"` |

Details: [Connectivity & weights](../guides/connectivity.md).

---

## `learner.TraceSTDP`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `0.0008` | STDP step size |
| `tau_trace` | `25` | Spike-trace time constant (ms) |
| `w_max` | `10.0` | Soft weight bound (BCM-style) |
| `mu_weight` | `0.6` | BCM softness exponent |
| `update_freq` | `100` | Apply STDP every N timesteps |
| `clip_weights` | `False` | Hard-clip weights after each update |
| `min_weight_exc` / `max_weight_exc` | `0.01` / `25.0` | Hard bounds (E), if `clip_weights=True` |
| `min_weight_inh` / `max_weight_inh` | `-25.0` / `-0.01` | Hard bounds (I), if `clip_weights=True` |

Details: [Learning (Trace STDP)](../guides/learning-stdp.md).

---

## `regularizer.Sleep` / `regularizer.Normalize`

Both support three `mode` values:

| Mode | Behaviour |
|------|-----------|
| `"static"` | Restore to a fixed target computed at initialisation |
| `"layer"` | Restore total synaptic drive across the layer |
| `"neuron"` | Restore per-neuron incoming synaptic drive (strongest homeostasis) |

**`Sleep`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | `300` | Timesteps per sleep episode |
| `frequency` | `1050` | Timesteps between sleep episodes |
| `mode` | `"static"` | Target restoration mode |
| `record_fn_se` / `record_fn_ee` | `None` | Optional per-event weight-recording callbacks |

**`Normalize`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency` | `1050` | Timesteps between normalisation events |
| `mode` | `"static"` | Target restoration mode |
| `record_fn_se` / `record_fn_ee` | `None` | Optional per-event weight-recording callbacks |

Details: [Sleep & homeostasis](../guides/sleep-homeostasis.md).

---

## `TrainResult`

Yielded after every training batch.

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Current epoch index (0-based) |
| `batch` | `int` | Batch index within epoch |
| `weights` | `np.ndarray` | Live weight matrix reference (N × N) |
| `accuracy` | `float \| None` | Training batch accuracy (`None` until first evaluation) |
| `phi` | `float \| None` | Phi clustering metric (`None` until first evaluation) |
| `spikes` | `np.ndarray \| None` | Mean firing rates `(n_images, N_exc)`, only if `return_spikes=True` |
| `stats` | `dict \| None` | Neuron/synapse diagnostics, only if `track_stats=True` |

## `EvalResult`

Returned by `model.validate()` and `model.test()`.

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | `float \| None` | Classification accuracy |
| `phi` | `float \| None` | Phi clustering metric |
| `split` | `str` | `"val"` or `"test"` |
| `spikes` | `np.ndarray \| None` | Mean firing rates, only if `return_spikes=True` |
