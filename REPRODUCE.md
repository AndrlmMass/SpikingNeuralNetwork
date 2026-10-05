# Reproducing the sleep paper's results

This covers the three studies in the revised paper: the sleep-duration sweep,
the comparison with conventional regularization, and the sleep-component
ablation. It also covers the beta GLMMs, tables and figures built on them. One
script drives everything: [`src/reproduce.py`](src/reproduce.py). Run all
commands from the repository root.

## 1. Environment

**Python.** Create the conda environment from
[`environment_linux.yml`](environment_linux.yml) or
[`environment_windows.yml`](environment_windows.yml). Both define `noise_env`
with Python 3.12 and torch 2.6.

```bash
conda env create --file environment_linux.yml
conda activate noise_env
```

The paper's runs used Python 3.13.5, torch 2.7.1 (CPU), numpy 2.1.3,
snntorch 0.9.4, numba 0.61.0 and deeplake 3.9.52, on Windows 11 with CPU only.
Every run is seeded, but different library versions can still change
accuracies slightly. notMNIST needs **deeplake 3.x**: version 4 changed the API.

**R** with `glmmTMB` and `emmeans`. The paper used R 4.5.2, glmmTMB 1.1.13
(TMB 1.9.18) and emmeans 2.0.4.

```r
install.packages(c("glmmTMB", "emmeans"))
```

If `Rscript` is not on your PATH, pass `--rscript /path/to/Rscript` or set
`RSCRIPT`.

## 2. Quick check: tables and figures from the recorded runs (minutes)

The per-run summaries of all 660 training runs are committed, so every GLMM,
table and figure can be rebuilt without training:

```bash
python src/reproduce.py --stages analysis
```

| Paper item | File | Script |
|---|---|---|
| Sleep-duration sweep table | `results/glmm/sweep_table.tex` | `src/glmm/fit_sweep.R` |
| Sweep figure | `figures/sweep_ratio_BW.pdf` | `src/glmm/plot_glmms_bw.py` (`fig_sweep`) |
| Sleep vs. conventional regularization table (`tab:betareg_baselines`) | `results/glmm/baselines_table.tex` | `src/glmm/fit_baselines.R` |
| Regularization figure | `figures/baselines_methods_BW.pdf` | `src/glmm/plot_glmms_bw.py` (`fig_baselines`) |
| Ablation table (`tab:betareg_ablation`) | `results/glmm/ablation_table.tex` | `src/glmm/fit_ablation_datasets.R` |
| Ablation figure | `figures/ablation_slope_BW.pdf` | `src/glmm/plot_glmms_bw.py` (`fig_ablation`) |

The CSVs in `results/glmm/` hold the same estimates at full precision. Each R
script's header describes its model, and the full console output (model
summaries, likelihood-ratio tests) goes to `results/reproduce/analysis_*.log`.

## 3. Full reproduction: retrain everything (~240 CPU-hours)

```bash
python src/reproduce.py --workers 15 --sleep-ratio 0.3
```

The script runs the stages below in order. Each training run is a separate
single-threaded process taking 15–30 minutes. With 15 workers the whole
pipeline takes about 16 hours. Runs that already have a result file are skipped,
so an interrupted run resumes when you start the same command again. A failed
run is retried once. Logs go to `results/reproduce/`.

| Stage | Runs | Script | Output |
|---|---|---|---|
| `data` | – | downloads MNIST, KMNIST, Fashion-MNIST (torchvision) and notMNIST (deeplake hub) once | `data/` |
| `sweep` | 220 = 11 ratios × 4 datasets × 5 seeds | [`src/sweep.py`](src/sweep.py) | `results/sweep/` |
| `baselines` | 100 = 5 methods × 4 datasets × 5 seeds | [`src/experiment.py`](src/experiment.py) | `results/baselines/` |
| `ablation` | 85 (MNIST) + 255 (other three) = (2⁴ combinations + no-sleep reference) × 5 seeds | [`src/ablation.py`](src/ablation.py) | `results/ablation/`, `results/ablation_datasets/` |
| `analysis` | – | [`src/glmm/`](src/glmm/) | `results/glmm/`, `figures/` |

**Running a subset.** Pass the stages you want, e.g.
`--stages ablation,analysis`. Run `data` first on a machine with network access.
Without it, parallel runs race the same download.

**On a SLURM cluster**, [`slurm/run_sweep.sh`](slurm/run_sweep.sh),
[`slurm/run_baselines.sh`](slurm/run_baselines.sh) and
[`slurm/run_ablation.sh`](slurm/run_ablation.sh) submit the same grids as array
jobs. Each study maps one `--task-id` to one run (`python src/<study>.py --list`
shows the mapping), and any arguments after the script name are passed through,
e.g. `sbatch slurm/run_baselines.sh --sleep-ratio 0.3`. The ablation script's
header shows how to submit its second grid (the other three datasets). Collect
the per-run records with `python src/<study>.py --collect`, then run
`python src/reproduce.py --stages analysis`.

Before submitting, edit the `#SBATCH` partition and mail lines, and set
`PROJECT_ROOT` to your checkout. The scripts expect a Singularity image
`noise_env.sif`; its header shows how to build it.

## 4. Settings fixed before the evaluation runs

- **Seeds.** Evaluation runs use seeds 42–46. Settings that were tuned used the
  separate seeds 100–102.
- **Data.** Each run trains on 4,000 images and tests on 1,000 (`main.py`
  defaults). MNIST, KMNIST and Fashion-MNIST draw from their official
  train/test splits; for notMNIST the split is built in `src/get_data.py`.
- **Sleep ratio (30%).** The baselines and ablation use the ratio with the
  highest test accuracy in the sweep, pooled over datasets and seeds
  (`experiment.resolve_sleep_ratio`). The sweep is almost flat: every nonzero
  ratio lies within 1.5 percentage points, and 30% beats 90% by 0.1 points. A
  retrained sweep can therefore select a different ratio, so
  `--sleep-ratio 0.3` holds the paper's value.
- **Continuous-decay rate (1e-5 per timestep, `experiment.DECAY_RATE`).** It
  was tuned for accuracy on MNIST with seeds 100–101 (2,000 train / 500 test
  images), over rates from 2.5e-6 to 2e-4. The commands and results are in
  `results/decay_tune/`.
- **Network operating point.** Input drive, learning rates, sleep pull
  strength (`RHO` in `sweep.py`) and weight targets are the defaults in
  `src/main.py` and `src/sweep.py`. They were set after the STDP-sign
  correction; [`DIARY.md`](DIARY.md) records how and why.

## 5. Code map

| File | Role |
|---|---|
| [`src/main.py`](src/main.py) | Trains and evaluates one network: one run, one seed |
| `src/train.py`, `src/create_network.py`, `src/weight_funcs.py`, `src/get_data.py` | Model, sleep phase, regularizers, data loading |
| [`src/sweep.py`](src/sweep.py), [`src/experiment.py`](src/experiment.py), [`src/ablation.py`](src/ablation.py) | The three study grids; each supports `--list`, `--task-id`, `--all`, `--collect` |
| [`src/glmm/common.R`](src/glmm/common.R) | Shared GLMM setup (beta family, logit link) |
| `src/glmm/fit_sweep.R`, `fit_baselines.R`, `fit_ablation_datasets.R` | One beta GLMM per study, with its table |
| [`src/glmm/plot_glmms_bw.py`](src/glmm/plot_glmms_bw.py) | The three figures |
| [`src/reproduce.py`](src/reproduce.py) | Runs all of the above |
| `src/tests/` | Unit tests for STDP sign, sleep spikes, binning, one-shot equivalence (`pytest src/tests`) |

Not part of the paper's pipeline: `src/reduction.py` (an exploratory follow-up),
and the older scripts in `experiments/`, `snntorch_model/`, `multinomial/` and
`multiple_log_reg/`.
