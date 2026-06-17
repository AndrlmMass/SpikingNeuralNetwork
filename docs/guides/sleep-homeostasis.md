# Sleep & homeostasis

STDP on its own tends to let total synaptic weight drift upward. `neurosnn` provides two
**regularizers** that periodically restore weights toward their initial scale. Both are
optional and plug into `model.train()` through the `regularizer=` argument.

## Two regularizers

| Regularizer | Behaviour |
|-------------|-----------|
| `regularizer.Sleep` | A biologically-inspired *sleep episode*: input is silenced, Gaussian membrane noise drives spontaneous activity, and weights are downscaled **gradually** over the episode while STDP keeps acting. |
| `regularizer.Normalize` | A deterministic, instantaneous weight rescaling — no sleep episode, no noise. A lighter-weight alternative. |

```python
import neurosnn as snn

# Sleep
reg = snn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")

# or deterministic normalization
reg = snn.regularizer.Normalize(frequency=1050, mode="neuron")

model.train(layers=[layer], learner=learner, regularizer=reg, epochs=3)
```

## The three modes

Both regularizers share a `mode` that determines *what* is held constant:

| Mode | What it restores | Locality |
|------|------------------|----------|
| `"static"` | Every non-zero weight is pushed toward a single fixed scalar (the mean initial weight) | Global, most aggressive |
| `"layer"` | The total synaptic drive summed across the whole layer is restored to its initial value | Layer-wide |
| `"neuron"` | Each postsynaptic neuron's incoming weight sum is restored individually | Per-neuron, most local, preserves learned structure best |

`"neuron"` is the strongest homeostatic constraint that still preserves relative weight
differences within each neuron's inputs; `"static"` discards learned differentiation by
collapsing weights toward a uniform value.

## Parameters

**`Sleep`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | `300` | Timesteps per sleep episode |
| `frequency` | `1050` | Timesteps between sleep onsets |
| `mode` | `"static"` | Target restoration mode (see above) |
| `record_fn_se` / `record_fn_ee` | `None` | Optional callbacks to record SE/EE weights during an episode |

**`Normalize`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency` | `1050` | Timesteps between normalisation events |
| `mode` | `"static"` | Target restoration mode (see above) |
| `record_fn_se` / `record_fn_ee` | `None` | Optional callbacks to record SE/EE weights |

## Choosing between them

- Use **`Normalize`** when you want pure homeostasis: a clean, deterministic rescaling with
  no other side effects. Good for ablations and for isolating the effect of weight scaling.
- Use **`Sleep`** when the *process* of sleep matters — the interaction of spontaneous
  noise-driven activity with STDP during gradual downscaling is itself the object of study.

??? note "How the scaling is computed per mode"
    At each event the regularizer computes a per-weight multiplier from the current weights
    and the stored initial targets:

    - **static** — target is `sum(initial) / n_nonzero` (the mean initial weight); each
      weight is set toward that scalar.
    - **layer** — multiplier is `initial_total_sum / current_total_sum`, applied to all
      weights so the layer's total drive returns to baseline.
    - **neuron** — for each postsynaptic neuron, multiplier is
      `initial_column_sum / current_column_sum`, restoring that neuron's incoming sum.

    For `Sleep`, the multiplier is additionally raised to the power `1/duration` so the
    restoration is spread smoothly across the episode rather than applied in one step. The
    kernels live in `neurosnn/_core/regularization.py` (`Sleep`, `Normalizer`).

## See also

- [Learning (Trace STDP)](learning-stdp.md) — the continuous plasticity this counterbalances.
- [Diagnostics & metrics](diagnostics.md) — using `record_fn` hooks to study weight evolution.
