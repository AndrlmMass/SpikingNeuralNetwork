# Source Code Structure

This directory contains the source code for the Spiking Neural Network with Sleep-like Dynamics.

## Directory Structure

```
src/
├── __init__.py              # Package entry point
│
├── models/                  # Neural network models and core logic
│   ├── __init__.py
│   ├── snn.py              # Main SNN class (snn_sleepy) and trainer
│   ├── dynamics.py         # STDP, membrane potential, weight update logic
│   ├── plasticity.py       # Sleep cycles and spike timing
│   └── layers.py           # Network architecture and state initialization
│
├── datasets/               # Data management
│   ├── __init__.py
│   ├── loaders.py          # Data streaming and batching
│   ├── mnist.py            # MNIST-family dataset handling
│   ├── geomfig.py          # Geometric figure generation
│   └── base.py             # Dataset base classes
│
├── evaluation/             # Metrics and visualization
│   ├── __init__.py
│   ├── plots.py            # Visualization (spikes, weights, accuracy)
│   ├── metrics.py          # Analysis metrics (Phi, t-SNE)
│   ├── classifiers.py      # Post-hoc classifiers (PCA+LR)
│   └── paper_figures.py    # Paper-ready figure composition
│
├── config/                 # Configuration
│   ├── __init__.py
│   └── defaults.py         # Default hyperparameters
│
├── methods/                # Helper methods
│   └── __init__.py
│
├── scripts/                # Execution scripts
│   ├── __init__.py
│   └── train_model.py      # Main entry point for training
│
├── snntorch_comparison/    # Baseline models
│   ├── __init__.py
│   └── main.py             # SNNTorch implementation
│
└── utils/                  # Utilities
    ├── __init__.py
    └── platform.py         # OS-specific helpers
```

## Quick Start

### Training

To run the standard training pipeline on the "geomfig" dataset:

```bash
python -m src.scripts.train_model --dataset geomfig
```

Key arguments:
- `--dataset`: `mnist`, `fashion`, `kmnist`, `geomfig`, etc.
- `--sleep-rate`: 0.0 to 1.0 (default 0.0)
- `--runs`: Number of independent runs (default 1)
- `--test-mode`: Enable for a quick dry-run (fewer epochs/samples)

### programmatic Usage

```python
from src.models.snn import snn_sleepy

# Initialize
snn = snn_sleepy(classes=[0, 1, 2, 3])

# Prepare data
snn.prepare_data(image_dataset="geomfig")

# Train
snn.train_network(train_weights=True, sleep=True)
```
