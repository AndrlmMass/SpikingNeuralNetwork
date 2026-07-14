# Diagnostics & metrics

`neurosnn` evaluates representations with a linear readout and a clustering metric, and can
optionally record fine-grained neuron, synapse, and weight statistics during training.

## Accuracy via PCA + logistic regression

Classification accuracy is measured by reading out the excitatory population's firing rates:
spikes are (optionally) reduced with PCA, then a logistic-regression classifier is fit on
the training split and scored on validation/test. This is reported as `accuracy` on
[`TrainResult`](training-loop.md#trainresult) and [`EvalResult`](training-loop.md#evalresult).

Relevant `train()` controls: `accuracy_method`, `use_LR`, `use_pca`, `pca_variance`
(default `0.95`).

## The phi metric

`phi` (η², eta-squared) measures **class separation** in the representation: the fraction of
total representational variance explained by the class labels. It ranges from 0 (classes
indistinguishable) to 1 (perfectly separated) and is independent of sample size, which makes
it suitable for comparing splits of different sizes. It is reported alongside `accuracy`.

??? note "How phi is computed"
    With between-class sum of squares (BCSS) and within-class sum of squares (WCSS),

    $$
    \phi = \eta^2 = \frac{\text{BCSS}}{\text{BCSS} + \text{WCSS}}.
    $$

    The implementation lives in `neurosnn/_evaluation/evaluation.py` (`Evaluator`, `Phi`).

## Tracking neuron & synapse statistics

Pass `track_stats=True` (and optionally `track_weights=True`) to `train()` to populate
`result.stats` with averaged per-batch diagnostics:

```python
for result in model.train(layers=[layer], learner=learner, track_stats=True):
    if result.stats is not None:
        print(result.stats)        # ~15 mean scalars: membrane, current, trace, x_tar, ...
```

These cover membrane-potential and synaptic-current deltas, adaptation, spike thresholds,
spike traces, the STDP first term, and the per-pathway `x_tar` values — useful for
diagnosing whether the network is in a healthy dynamical regime.

## Recording weight evolution during regularization

`Sleep` and `Normalize` accept `record_fn_se` / `record_fn_ee` callbacks invoked at each
regularization event, letting you log how SE/EE weights change across sleep vs. wake phases:

```python
history = []
reg = snn.regularizer.Sleep(
    duration=200, frequency=1050, mode="neuron",
    record_fn_ee=lambda weights, t: history.append((t, weights.copy())),
)
```

## Plotting utilities

The package ships internal plotting helpers (under `neurosnn/_plot/`) for accuracy curves,
PCA scatter plots of the spike representation, weight-matrix heatmaps, and weight-trajectory
plots. These are used by the bundled experiments; see
[Reproducing experiments](../research/experiments.md).

## See also

- [The training loop](training-loop.md) — where these results surface.
- [Learning (Trace STDP)](learning-stdp.md) — what the `x_tar` and weight-delta stats mean.
