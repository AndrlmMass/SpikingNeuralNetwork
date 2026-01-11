# SpikingNeuralNetwork PhD project
This project aims to replicate and expand the model developed in [Zenke et al.'s 2015 article](https://www.nature.com/articles/ncomms7922) and is accompanied by our sleep-protocol article (link will be added when published).

## Install repository

    git clone  https://github.com/AndrlmMass/SpikingNeuralNetwork.git

## Branch

The article repo is located on the `SNN_paper_repo`-branch. To switch branch, run

```
git switch SNN_paper_repo
```

## Set up the conda environment

We have an `environment.yml` file that contains all packages to build and work with the script. 

**Install** the `conda` environment:

    conda env create --file environment.yml

**Activate** the `conda` environment:

    conda activate SNN_env

**Update** the `environment.yml` with existing env by:

    conda env update --name SNN_env --file environment.yml --prune'

## Experiments

### Toy geometric experiment

To run the toy geometric experiment:

    python src/main.py --sleep-rate 0.0 0.2 --dataset geomfig --runs 5

**Optional flags:**
- `--plot-weights-per-epoch`: Generate weight plots after each epoch
- `--preview-dataset`: Preview training data once per dataset

**Disclaimer:**
Note that the bar-plot comparing the performance between sleep and non-sleep is not included in this repository.

### Sleep Rate Comparison Experiment

To run the sleep rate comparison experiment across multiple datasets:

    python src/main.py --sleep-rate 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --dataset mnist fmnist kmnist notmnist --runs 5

**Optional flags:**
- `--plot-weights-per-epoch`: Generate weight plots after each epoch
- `--track-excel`: Save results to `GLM/Results_.xlsx` file for tracking
- `--preview-dataset`: Preview training data once per dataset

Then run the snntorchhmm script with the following:

Navigate to the snntorchhmm directory:

```
cd snntorchhmm
```

Run the sleep rate comparison experiment for each dataset (the `--dataset` argument accepts one dataset at a time):

    python main.py --sleep-interval-pct 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --dataset MNIST FMNIST KMNIST NOTMNIST --runs 5 

**Note:** The snntorchhmm script uses uppercase dataset names (MNIST, FMNIST, KMNIST, NOTMNIST). Each dataset must be run separately as the `--dataset` argument accepts only one value at a time.

**Optional flags:**
- `--preview-samples`: Preview training data once per dataset
- `--track-excel`: Save results to `GLM/Results_.xlsx` file for tracking (same file as src/main.py)

## Statistical Analysis and Visualization

After running the experiments with `--track-excel` flag, you can perform statistical analysis and create comparison plots.

### Run GLMM Analysis

Navigate to the GLM directory:

```
cd GLM
```

Run the R analysis script:

```
Rscript analysis.r
```

This script:
- Reads `Results_.xlsx` from the current directory (GLM/)
- Fits a Generalized Linear Mixed Model (GLMM) with beta family
- Outputs `pred.xlsx` with model predictions and confidence intervals

### Create Comparison Plot

Navigate back to the repository root:

```
cd ..
```

Run the Python plotting script from the `src` directory:

```python
from plot import plot_glmm_with_raw_accuracy

plot_glmm_with_raw_accuracy(
    pred_path="GLM/pred.xlsx",
    results_path="GLM/Results_.xlsx",
    output_path="sleep_duration_comparison_with_accuracy.pdf",
    show_plot=False
)
```

This creates a comparison plot showing:
- GLMM predicted accuracy with confidence intervals (filled markers with error bars)
- Observed mean accuracy from raw data (hollow markers)
- Separate panels for each dataset (MNIST, KMNIST, Fashion-MNIST, NotMNIST)
- Separate markers for each model (SNN_sleepy: circles, snntorch: triangles)

**Complete workflow:**
1. Run experiments with `--track-excel` flag for both models (results saved to `GLM/Results_.xlsx`)
2. Navigate to `GLM/` directory and run `Rscript analysis.r` (generates `pred.xlsx`)
3. Run the plotting function from repository root to create the comparison figure

Both the R analysis script and Python plotting function work with the long format data structure (columns: `Sleep_duration`, `Model`, `Run`, `Lambda`, `Seed`, `Dataset`, `Accuracy`) saved by the experiment scripts.