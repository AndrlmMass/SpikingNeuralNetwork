# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational neuroscience research codebase implementing Spiking Neural Networks (SNNs) with bio-inspired learning. The core research focus is **sleep-like synaptic homeostasis**: simulating sleep protocols that use noise-driven synaptic downscaling to improve noise tolerance and generalization. The paper extends Zenke et al. (2015).

## Environment Setup

```bash
conda env create --file environment.yml
conda activate SNN_env
```

To update an existing environment:
```bash
conda env update --name SNN_env --file environment.yml --prune
```

Key dependencies: Python 3.12, NumPy 2.0+, Numba 0.61+ (JIT), PyTorch 2.6, scikit-learn.

## Running Experiments

**Quick test (toy geometric dataset, ~1 min):**
```bash
python experiments/noise_article/sleep_norm_comparison/run_experiment.py --dataset geomfig --epochs 1
```

**Sleep vs. normalization comparison (main article experiment):**
```bash
python experiments/noise_article/sleep_norm_comparison/run_experiment.py \
  --dataset mnist --epochs 5 --reg-type sleep --reg-mode static --seed 0
```

**Phase 2 sleep/noise grid sweep:**
```bash
python experiments/noise_article/sleep_noise_optimization/run_sleep_tuning.py \
  --sleep-duration 100 --var-noise 0.1 --reg-type sleep --reg-mode static
```

**HPC submission (SLURM + Singularity):** `run_CPU.sh` / `run_GPU.sh`

**Common flags:** `--dataset` {mnist, kmnist, fmnist, notmnist, geomfig}, `--epochs`, `--seed`, `--var-noise`, `--sleep-duration`, `--reg-type` {sleep, normalize, none}, `--reg-mode` {static, layer, neuron}, `--val-every`, `--output-dir`

## Post-Training Analysis

```bash
cd experiments/GLM && Rscript analysis.r   # GLMM in R → pred.xlsx
cd ../..
python -c "from neurosnn._plot.training import plot_glmm_with_raw_accuracy; \
  plot_glmm_with_raw_accuracy('experiments/GLM/pred.xlsx', 'experiments/GLM/Results_.xlsx', 'out.pdf')"
```

There are no automated tests — validation is done by running experiments and inspecting results.

## Architecture

The package is `neurosnn/` with a two-layer design: a **user-facing API** and an **internal core**.

### User-Facing API (`neurosnn/`)

| File | Class | Role |
|------|-------|------|
| `model.py` | `Model` | Top-level entry point. Holds data config; calling `.train()` yields `TrainResult` per batch. |
| `layer.py` | `Layer` | Dataclass describing one E/I population pair (N_exc, N_inh, membrane dynamics, connectivity). |
| `learner.py` | `TraceSTDP` | Spike-trace STDP rule with BCM-style soft weight bounds. |
| `regularizer.py` | `Sleep` / `Normalize` | Regularization strategies: noise-driven sleep downscaling vs. deterministic normalization. |
| `membrane.py` | `LIF` | Leaky integrate-and-fire neuron parameter config (all in mV/ms). |
| `weights.py` | `WeightsSpec` | Weight density and peak amplitude for SE/EE/EI/IE connection types. |
| `results.py` | `TrainResult`, `EvalResult` | Dataclasses returned from training and evaluation. |

### Internal Core

**`neurosnn/_core/`** — per-batch training mechanics:
- `trainer.py`: Main training loop; orchestrates membrane updates → spike generation → STDP → regularization.
- `neurons.py`: `NeuronState` / `MembranePotential` — **Numba-JIT compiled** hot paths for neuron dynamics.
- `synapses.py`: `Learner` (STDP variants) and `Clipper` (hard weight clipping).
- `regularization.py`: `Sleep` and `Normalizer` implementations.
- `trackers.py`: `TrainTracker` — live statistics accumulation during a batch.

**`neurosnn/_network/`** — network lifecycle:
- `model.py` (`SNNModel`): Encapsulates network structure (N_exc, N_inh, N_x populations), data stream, and weight factories.
- `runner.py` (`Runner`): Persistent state manager across epochs — maintains membrane potentials for biological realism. Yields `TrainResult` per batch.
- `init_weights.py` (`WeightFactory`): Builds sparse connectivity matrices with Gaussian receptive fields or random patterns.
- `io.py` (`CheckpointManager`): Model serialization and recovery.

**`neurosnn/_data/get_data.py`** (`ImageDataStreamer`): Loads MNIST variants, converts images to Poisson spike trains, streams batches on demand.

**`neurosnn/_evaluation/evaluation.py`** (`Evaluator`): Fits PCA classifiers; computes accuracy and the **phi metric** (within/between-class scatter — the primary clustering quality measure).

### Training Data Flow

```
Model.train()
  └─→ SNNModel  (data streaming, weight factory)
        └─→ Runner  (persistent state across epochs)
              └─→ Trainer  (per-batch)
                    ├─ NeuronState / MembranePotential  [Numba JIT]
                    ├─ Learner  (STDP weight updates)
                    ├─ Sleep / Normalizer  (regularization)
                    └─ TrainTracker  (stats)
              └─→ Evaluator  (PCA, accuracy, phi)
        └─→ TrainResult  (yielded to caller)
  └─→ HistoryTracker  (logs to JSON)
```

### Key Design Decisions

- **Lazy initialization**: Data and weights are only materialized when `train()` is called.
- **Numba JIT on hot paths**: `neurons.py` dynamics are compiled; avoid pure-Python loops in that module.
- **Persistent membrane state**: `Runner` keeps neuron state across epochs — this is intentional for biological realism, not a bug.
- **Auto-generated output dirs**: Experiment scripts derive output paths from hyperparameter values for sweep reproducibility.
- **Phi metric**: The primary evaluation signal beyond accuracy — measures representational quality via PCA scatter.

## Branches

- `sleep-redeploy` — active development
- `paper-repo` / `SNN_paper_repo` — reproducible article results; treat as stable baseline
