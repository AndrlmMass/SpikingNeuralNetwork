# neurosnn — Developer Guide

This document describes the internal architecture of the `neurosnn` package: how the files are organised, how data flows through a training run, and where to look when you want to modify or extend the library.

---

## Repository layout

```
SpikingNeuralNetwork/
├── neurosnn/               # The installable package (TBD)
│   ├── __init__.py         # Public API surface
│   ├── model.py            # User-facing Model class
│   ├── layer.py            # Layer descriptor (config only)
│   ├── membrane.py         # LIF neuron spec (config only)
│   ├── weights.py          # Weight factory functions + WeightsSpec
│   ├── learner.py          # TraceSTDP spec (config only)
│   ├── regularizer.py      # Sleep / Normalize specs (config only)
│   ├── results.py          # TrainResult / EvalResult dataclasses
│   ├── _core/              # NJIT-compiled inner loops
│   │   ├── neurons.py      # Membrane update + spike detection
│   │   ├── synapses.py     # Trace-STDP weight update
│   │   ├── regularization.py  # Sleep / normalization scaling
│   │   ├── trainer.py      # Timestep state machine
│   │   └── trackers.py     # Per-batch statistics accumulator
│   ├── _network/           # Network construction + training runner
│   │   ├── model.py        # SNNModel (internal dimension bookkeeping)
│   │   ├── init_weights.py # WeightFactory (matrix initialisation)
│   │   ├── runner.py       # Runner (epoch/batch loop, yields TrainResult)
│   │   └── io.py           # CheckpointManager (save/load .npy)
│   ├── _data/
│   │   └── get_data.py     # ImageDataStreamer (loads + Poisson-encodes)
│   ├── _evaluation/
│   │   └── evaluation.py   # Evaluator (PCA + LogReg + Phi metric)
│   ├── _plot/
│   │   └── training.py     # PCAScatterDisplay, plot_accuracy
│   └── _utils/
│       ├── logger.py        # HistoryTracker (JSONL training logs)
│       ├── performance.py   # Background plot threads, RAM reporting
│       ├── weight_sampler.py  # WeightSampler (callback-based tracking)
│       └── helper.py        # Miscellaneous array utilities
├── docs/                   # MkDocs documentation source
├── experiments/            # Research experiment scripts
├── data/                   # Downloaded datasets (auto-created)
├── model/                  # Saved weight checkpoints
├── results/                # Training logs
├── pyproject.toml          # Build + dependency declaration
└── mkdocs.yml              # Docs site config
```

---

## Public API vs. internal modules

Everything the user touches is in the flat `neurosnn/` namespace:

| Public name | File | Role |
|---|---|---|
| `snn.Model` | `model.py` | Entry point — owns data + training |
| `snn.Layer` | `layer.py` | Config container only |
| `snn.membrane.LIF` | `membrane.py` | LIF neuron parameters |
| `snn.weights.random/receptive_fields/oriented_receptive_fields` | `weights.py` | Weight initialisation factories |
| `snn.learner.TraceSTDP` | `learner.py` | STDP hyper-parameters |
| `snn.regularizer.Sleep` / `.Normalize` | `regularizer.py` | Homeostasis config |
| `snn.TrainResult` / `snn.EvalResult` | `results.py` | Result containers |

The `_core/`, `_network/`, `_data/`, `_evaluation/`, and `_utils/` sub-packages are internal. They are not re-exported and their APIs can change freely.

---

## Configuration → runtime conversion

The user-facing classes (`Layer`, `LIF`, `TraceSTDP`, etc.) are all plain dataclasses — no computation happens at construction time. Each one exposes a `_to_runner_kwargs()` method (or equivalent) that `Model.train()` calls to flatten all parameters into keyword arguments before passing them down to the low-level objects.

This split keeps the public API declarative and easy to serialise, while keeping the internal runtime objects free from public-API stability concerns.

---

## Network dimensions and index layout

`SNNModel` (in `_network/model.py`) is the single source of truth for all index arithmetic. The full neuron array has shape `(N,)` where:

```
N = N_x + N_exc + N_inh

Slice           Name    Meaning
[:st]           input   Poisson spike input (N_x neurons, st = N_x)
[st:ex]         exc     Excitatory population (N_exc neurons, ex = st + N_exc)
[ex:ih]         inh     Inhibitory population (N_inh neurons, ih = ex + N_inh)
```

All weight matrices follow the same convention — rows are presynaptic neurons, columns are postsynaptic neurons.

| Matrix | Shape | Pathway |
|---|---|---|
| `weights_se` | `(N_x, N_exc)` | Stimulus → Excitatory |
| `weights_ee` | `(N_exc, N_exc)` | Excitatory → Excitatory |
| `weights_ei` | `(N_exc, N_inh)` | Excitatory → Inhibitory |
| `weights_ie` | `(N_inh, N_exc)` | Inhibitory → Excitatory |

The combined matrix `weights` is `(N, N)` and is kept dense; sparsity is tracked separately via `nonzero_pre_idx` arrays used by the STDP kernel.

---

## Data flow: one training batch

```
ImageDataStreamer.get_batch()
    → (T, N_x) Poisson spike array  (T = num_steps)

Build full spike array (T, N):
    spikes[:, :st]  = input spikes
    spikes[:, st:]  = 0  (will be filled by neuron dynamics)

Trainer.step(weights, mp, spikes, ...)
    For t in 0..T-1:
        update_membrane_potential()  [NJIT]
            → mp_new, I_syn_exc, I_syn_inh
        update_spikes()              [NJIT]
            → spikes[t], spike_threshold, adaptation, spike_trace
        update_x_tar()
            → population-mean trace (STDP reference)
        if t % update_freq == 0:
            trace_STDP()             [NJIT]
                → weight deltas applied in-place
        if t % reg_frequency == 0:
            Sleep or Normalize       [NJIT]
                → weight scaling applied in-place

    → returns (weights, spike_record)

Evaluator.score(spike_record, labels)
    → accuracy, phi

yield TrainResult(epoch, batch, weights, accuracy, phi, ...)
```

---

## NJIT-compiled kernels

All hot-path kernels live in `_core/` and are compiled by Numba on first call (typically a few seconds of warm-up; subsequent calls are fast). The four main kernels:

### `neurons.update_membrane_potential` (`_core/neurons.py`)

Implements the discrete LIF update for excitatory and inhibitory populations:

```
I_syn[i] += (-I_syn[i] + sum(weights[:, i] * spikes)) * dt / tau_syn
mp[i]    += (-(mp[i] - V_rest) + R * I_syn[i]) / tau_m * dt
mp[i]    += N(mean_noise, var_noise)   # only during sleep episodes
```

**Key inputs**: `mp (N,)`, `spikes (N,)`, full weight matrix, time constants  
**Key outputs**: `mp_new (N,)`, `I_syn_exc`, `I_syn_inh`

### `neurons.update_spikes` (`_core/neurons.py`)

Detects threshold crossings and applies reset, spike-frequency adaptation, and trace decay:

```
if mp[i] > threshold[i]:
    spikes[i] = 1
    mp[i]     = V_reset
    a[i]     += delta_adaptation          # per-spike adaptation increment
    trace[i] += 1.0
threshold[i]  = base_threshold + a[i]    # adaptive threshold
a[i]         *= exp(-dt / tau_adaptation)
trace[i]     *= exp(-dt / tau_trace)
```

**Key outputs**: `spikes (N,)`, updated `threshold`, adaptation `a`, `spike_trace`

### `synapses.trace_STDP` (`_core/synapses.py`)

BCM-style multiplicative STDP applied only at postsynaptic spikes:

```
for each postsynaptic neuron i that fired:
    for each presynaptic neuron j (stored as sparse index list):
        Δw = lr * (trace[j] - x_tar) * (w_max - weights[j,i])^mu
        weights[j,i] += Δw
```

`x_tar` is the population-mean spike trace, computed once per STDP call by `update_x_tar()`. The soft bound `(w_max - w)^mu` prevents runaway weight growth without hard clipping.

### `regularization.post_sleep / post_norm` (`_core/regularization.py`)

Both modes compute a per-weight or per-neuron scale factor comparing current weight sums to a stored initial sum, then apply it multiplicatively:

```
# "neuron" mode:
scale[j, i] = initial_sum_post[i] / current_sum_post[i]
weights[j, i] *= scale[j, i]

# "layer" mode:
scale = initial_sum_layer / current_sum_layer
weights *= scale
```

`Sleep` additionally injects Gaussian noise into the membrane potential during sleep episodes and silences the input drive.

---

## Weight initialisation

`WeightFactory` (`_network/init_weights.py`) is called once during `Model.train()` setup. It supports three strategies controlled by the `WeightsSpec`:

| Strategy | S→E pathway | E→E / E→I / I→E |
|---|---|---|
| `random()` | Uniform sparse random | Uniform sparse random |
| `receptive_fields()` | Gaussian RF centred on grid | Gaussian RF |
| `oriented_receptive_fields()` | Elliptical Gaussian, cycling orientations | Gaussian RF |

All strategies honour the `density_*` probabilities (structural connection mask) and `peak_*` magnitudes. Receptive field methods use toroidal boundary conditions so edge neurons get the same RF quality as centre neurons.

After construction, `WeightFactory.sparse_indices()` returns the `nonzero_pre_idx` arrays — jagged lists padded with `-1` — that the STDP kernel uses to skip zero-weight entries.

---

## Data loading and encoding (`_data/get_data.py`)

`ImageDataStreamer` loads the full dataset into RAM at construction time. `get_batch(start, count, partition)` returns a `(T, N_x)` binary array:

1. A batch of `count` images is retrieved from the preloaded arrays.
2. Each pixel value `p ∈ [0,1]` is converted to a Poisson spike train at rate `max_rate_hz * p` Hz, sampled at `dt = 1 ms` per timestep over `T = num_steps` steps.
3. If `gabor=True`, images are convolved with oriented Gabor filters before encoding.

Supported dataset keys: `"mnist"`, `"kmnist"`, `"fmnist"`, `"notmnist"`, `"geomfig"`. Standard datasets are downloaded automatically via `torchvision` on first use; `"notmnist"` additionally requires `deeplake`.

---

## Evaluation (`_evaluation/evaluation.py`)

`Evaluator` is fitted once per evaluation call:

1. Spike records `(n_images, N_exc)` are mean firing rates over the presentation window.
2. PCA reduces to a fixed number of components.
3. A `LogisticRegression` is fit on the PCA-projected training data and scored on held-out data.
4. **Phi (η²)**: `BCSS / total_SS` — the fraction of total spike-rate variance explained by class labels. This is scale-independent (unlike F-statistics), bounded in [0, 1], and interpretable as clustering quality.

---

## Checkpointing and logging

`CheckpointManager` (`_network/io.py`) saves the weight matrix as a `.npy` file and a JSON config alongside it. On load it matches saved configs against the current run parameters and falls back to the most recent checkpoint if no exact match is found.

`HistoryTracker` (`_utils/logger.py`) appends one JSON line per evaluation event to a `.jsonl` file in the `results/` directory. This is independent of the `TrainResult` generator and runs automatically.

---

## Adding a new dataset

1. Add a branch to `ImageDataStreamer.__init__()` in `_data/get_data.py` that loads images into `self.images` (shape `(n, H, W)` or `(n, N_x)`) and `self.labels` (shape `(n,)`).
2. Register the key string in the `image_dataset` argument handling.
3. No other changes are required — encoding, batching, and label handling are generic.

## Adding a new weight-initialisation strategy

1. Add a factory function to `weights.py` that returns a `WeightsSpec` with the desired parameters.
2. Add the corresponding builder in `WeightFactory._fill_*()` inside `_network/init_weights.py`.
3. Export the new factory from `neurosnn/__init__.py`.

## Adding a new regularisation mode

1. Add an NJIT-compiled scaling function in `_core/regularization.py`.
2. Add a new dataclass in `regularizer.py` with a `_to_runner_kwargs()` method.
3. Wire it up in `_core/trainer.py` alongside the existing `Sleep` / `Normalize` dispatch.

---

## Development setup

```bash
git clone https://github.com/AndrlmMass/SpikingNeuralNetwork.git
cd SpikingNeuralNetwork
pip install -e .                  # installs core deps from pyproject.toml
pip install torch torchvision     # required for dataset loading
pip install deeplake              # optional: needed only for "notmnist"
pip install -e ".[docs]"          # optional: MkDocs site
mkdocs serve                      # preview docs at http://127.0.0.1:8000
```

**Numba warm-up**: The first call to any NJIT function triggers compilation. In a development session this adds ~5–10 s. Subsequent calls in the same process are fast. If you change a kernel, restart the Python process to pick up the recompiled version.

**Reproducibility**: Call `seed_numba_rng(seed)` (from `_core/neurons.py`) before training to seed Numba's thread-local RNG; this is separate from `np.random.seed`. The `random_state` argument to `Model` covers the NumPy-side RNG.
