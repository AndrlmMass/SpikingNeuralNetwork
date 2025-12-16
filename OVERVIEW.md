# Noise Analysis

This directory contains scripts for analyzing the effects of noise in spiking neural networks (SNNs), particularly in the context of sleep-like states and their impact on network performance and stability. The implementation is based on simulating SNNs with excitatory and inhibitory neurons, using spike-encoded MNIST data.

## Overview

The primary script is `snn_sleepy_trainer.py` (formerly `main.py`), which initializes an SNN model using the `snn_sleepy` class from `snn_sleepy_core.py` (formerly `big_comb.py`) and executes a pipeline of data preparation, network setup, training, and analysis. The goal is to study how noise and sleep mechanisms affect learning and representation in the network.

The `snn_sleepy` class manages the entire workflow, including data handling, weight initialization, training loops with noise injection, and post-training analyses like t-SNE and phi metrics.

## Detailed Process Explanation

The process in `main.py` follows these main steps, each corresponding to a method in the `snn_sleepy` class:

1. **Initialization**:
   - An instance of `snn_sleepy` is created with parameters such as the number of excitatory neurons (`N_exc`), inhibitory neurons (`N_inh`), input size (`N_x`), and classes (e.g., MNIST digits 0-9).
   - This sets up the network structure: stimulation layer (inputs), excitatory layer, and inhibitory layer.
   - Directory changes ensure operations are within the `noise_analysis` folder.

2. **Data Preparation (`prepare_data` method)**:
   - Loads or generates spike-encoded data from the MNIST dataset for training and testing.
   - Parameters control the number of images, whether to add noise (e.g., `noisy_data=True`, `noise_level=0.005`), breaks between images (`add_breaks`), gain, and more.
   - Supports validation splits and time-varying inputs.
   - Data is cached in `data/sdata` or `data/mdata` directories to avoid regeneration unless forced.
   - Outputs: Spike trains (`data_train`, `data_test`) and labels (`labels_train`, `labels_test`).
   - Optional plotting: Spike plots or heatmaps of activity.

3. **Network Setup (`prepare_training` method)**:
   - Initializes weight matrices for connections: stimulation-to-excitatory (SE), excitatory-to-excitatory (EE), excitatory-to-inhibitory (EI), inhibitory-to-excitatory (IE).
   - Weights are created with specified densities (e.g., `w_dense_ee=0.1`) and values (e.g., `ei_weights=0.5`).
   - Sets up arrays for membrane potentials (`mp_train`, `mp_test`) and spikes (`spikes_train`, `spikes_test`), initialized to resting potential (e.g., -70 mV).
   - Optional: Plots initial weights or network structure.

4. **Training (`train` method)**:
   - Trains the network over epochs (calculated from total images and batch size).
   - For each epoch:
     - Fetches a batch of training and test data.
     - Creates fresh arrays for membrane potentials and spikes.
     - Runs the training simulation (`train_network` from `train.py`) on training data, with options for:
       - Weight training using STDP rules (timing-dependent or trace-based).
       - Noise injection in potentials, thresholds, or weights.
       - Spike adaptation and synaptic time constants.
       - Sleep simulation (if enabled, checks for sleep intervals and synchronizes).
       - Weight decay with rates for excitatory and inhibitory weights.
     - Tests on test data without training weights.
     - Computes metrics: Phi (clustering quality via PCA and within/between-class scatter) and accuracy (top responders per class).
   - Tracks performance per epoch and supports early stopping.
   - Optional plotting: Spikes, membrane potentials, weights, thresholds, traces, top responders, epoch performance.
   - Saves model to `model/` directory if specified, including weights, spikes, etc.
   - If comparing decay rates, runs multiple simulations and plots phi vs. accuracy.

5. **Analysis (`analysis` method)**:
   - Performs dimensionality reduction and visualization:
     - t-SNE on excitatory spikes for train/test data to visualize clustering.
     - PCA analysis on spikes.
   - Calculates and prints phi metrics if enabled.
   - Uses saved spikes and labels from training.

## Supporting Files and Their Roles

- **`snn_sleepy_core.py`** (formerly `big_comb.py`): Defines the `snn_sleepy` class and integrates all components.
- **`snn_dynamics.py`** (formerly `train.py`): Implements the core training loop (`train_network`) with neuron dynamics, spike generation, and weight updates.
- **`dataset_loaders.py`** (formerly `get_data.py`): Functions to load MNIST and convert to spike trains (`create_data`), with noise and encoding options.
- **`create_network.py`**: Helpers to generate weights (`create_weights`) and initialize arrays (`create_arrays`).
- **`plot.py`**: Plotting utilities for spikes, heatmaps, membrane potentials, weights, thresholds, traces, top responders, phi-accuracy, and epoch training.
- **`analysis.py`**: Analysis functions like t-SNE, PCA, and phi calculation (`calculate_phi`).
- **`weight_funcs.py`**: Likely contains weight update rules (e.g., STDP functions).
- **`environment.yml`**: Conda environment specification for dependencies.
- **`figures/` and `data/`**: Output directories for plots and data.
- **`model/`**: Stores saved models.

This documentation provides a comprehensive overview to help future LLMs or developers understand and extend the noise analysis in this SNN project. 