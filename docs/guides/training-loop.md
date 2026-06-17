# The training loop

`model.train()` is a **generator**. This page explains the pattern, the result objects, and
how to evaluate and inspect mid-run.

## The generator pattern

`train()` yields one [`TrainResult`](#trainresult) per batch. The body of your `for` loop
runs once per batch, and you decide what happens there:

```python
for result in model.train(layers=[layer], learner=learner, epochs=3):
    if result.batch % 5 == 0 and result.accuracy is not None:
        val = model.validate()
        print(result.epoch, result.batch, result.accuracy, val.accuracy)

test = model.test()
print("Test:", test.accuracy)
```

Simulation state persists across epochs and across `validate()` calls — evaluating inside
the loop does not reset training.

## `train()` arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `layers` | — | List with exactly one `Layer` |
| `learner` | `TraceSTDP()` | The learning rule |
| `regularizer` | `None` | Optional `Sleep` / `Normalize` |
| `epochs` | `1` | Number of passes over the training set |
| `train_weights` | `True` | If `False`, run the network without updating weights |
| `dt` | `1.0` | Simulation timestep (ms) |
| `track_stats` | `False` | Collect neuron/synapse diagnostics into `result.stats` |
| `track_weights` | `False` | Collect weight-update diagnostics |
| `return_spikes` | `False` | Include per-image firing rates in results |

## TrainResult

Yielded after every training batch.

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Current epoch index (0-based) |
| `batch` | `int` | Batch index within the epoch |
| `weights` | `np.ndarray` | Live reference to the weight matrix (N × N) |
| `accuracy` | `float \| None` | Training-batch accuracy (`None` until first evaluation) |
| `phi` | `float \| None` | Phi clustering metric (`None` until first evaluation) |
| `spikes` | `np.ndarray \| None` | Mean firing rates `(n_images, N_exc)`, only if `return_spikes=True` |
| `stats` | `dict \| None` | Neuron/synapse diagnostics, only if `track_stats=True` |

## EvalResult

Returned by `model.validate()` and `model.test()`.

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | `float \| None` | Classification accuracy |
| `phi` | `float \| None` | Phi clustering metric |
| `split` | `str` | `"val"` or `"test"` |
| `spikes` | `np.ndarray \| None` | Mean firing rates, only if `return_spikes=True` |

## Inspecting before training

Because weights are constructed on the **first** generator iteration, you can peek at the
initial network before any learning by advancing the generator once:

```python
gen = model.train(layers=[layer], learner=learner, epochs=1)
first = next(gen)            # builds the network, runs one batch
initial_weights = first.weights
# ... then continue iterating
for result in gen:
    ...
```

## See also

- [Diagnostics & metrics](diagnostics.md) — what `stats`, `phi`, and `return_spikes` give you.
- [API reference](../reference/api.md) — full `Model` constructor signature.
