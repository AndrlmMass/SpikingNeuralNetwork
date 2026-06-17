# Reproducing experiments

The `experiments/` folder contains the reproducible scripts behind the articles. They are
organised by research theme and follow a consistent **run → aggregate → analyse** pattern.

## Layout

```
experiments/
├── noise_article/
│   ├── sleep_norm_comparison/      # Regularizer type/mode comparison
│   │   ├── run_experiment.py       #   single run (one config/seed)
│   │   └── analyse_phase1.py       #   post-hoc statistics
│   ├── sleep_noise_optimization/   # Sleep/noise parameter grid
│   │   ├── run_sleep_tuning.py
│   │   ├── aggregate_phase2.py     #   pool results across jobs
│   │   └── plot_phase2_heatmaps.py
│   ├── GLMM/                       # Generalized linear mixed-model analysis
│   └── visualize_mnist_pipeline.py # Dataset/encoding preview
├── RF_article/
│   ├── RF_v_random/                # Receptive fields vs. random connectivity
│   │   ├── run_experiment.py
│   │   └── analyse_phase.py
│   └── RF_size_tuning/             # RF-size sweep
│       ├── run_experiment.py
│       ├── aggregate_rf_size.py
│       └── plot_rf_size_heatmaps.py
└── generic_testing/
    ├── run_test.py                 # Minimal smoke test
    └── inspect_rfs.py              # Receptive-field visualisation
```

## The standard pattern

1. **Run** (`run_experiment.py` / `run_test.py`) — parses CLI args (seed, hyperparameters,
   dataset), builds a `Model` + `Layer` + `TraceSTDP` (+ optional `Sleep`/`Normalize`),
   trains, and logs results to JSON/JSONL. Designed to run as one job in a job array.
2. **Aggregate** (`aggregate_*.py`) — pools the per-job result files into a single table.
3. **Analyse / plot** (`analyse_*.py`, `plot_*.py`) — statistical tests, GLMM fitting, and
   publication figures (heatmaps, curves).

All run scripts seed the RNG, Numba, and Torch from `--seed` for reproducibility.

## Quick local check

For a fast end-to-end smoke test that doesn't need a large download, use the geomfig dataset
via the generic test:

```bash
python experiments/generic_testing/run_test.py
```

`experiments/generic_testing/inspect_rfs.py` renders the receptive fields produced by the
[`weights`](../guides/connectivity.md) factories, and
`experiments/noise_article/visualize_mnist_pipeline.py` previews the
[image-to-spike pipeline](../guides/data-encoding.md).

## Representative usage

```python
import neurosnn

model = neurosnn.Model(input_size=225, num_steps=350, all_images_train=1000)
layer = neurosnn.Layer(
    N_exc=400, N_inh=100,
    weights=neurosnn.weights.receptive_fields(),
)
regularizer = neurosnn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")

for result in model.train(layers=[layer], regularizer=regularizer,
                          track_stats=True, epochs=1):
    if result.batch % 10 == 0:
        val = model.validate()
        # log {epoch, accuracy: val.accuracy, phi: val.phi}

test = model.test()
```

## See also

- [Background & citation](background.md) — what each article studies.
- [Quickstart](../getting-started/quickstart.md) — the minimal training script.
