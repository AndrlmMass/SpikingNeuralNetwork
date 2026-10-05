#!/bin/bash
# Sleep component ablation — Reviewer 3, Major Point 2.
#
# Full 2^4 factorial over the four sleep components, plus a no-sleep reference:
# (16 + 1) x 1 dataset (MNIST) x 5 seeds = 85 runs.
#
#   downscale  power-law pull of each weight toward w_target
#   noise      Gaussian membrane noise during sleep
#   stdp       plasticity active during sleep (the "replay" component)
#   suppress   sensory drive zeroed during sleep
#
# A full factorial rather than leave-one-out, because the paper's replay claim is
# itself an interaction — noise is meant to matter *because* STDP is active to
# consolidate what it reactivates — and leave-one-out cannot detect that.
#
# The sleep ratio is read from the sweep (results/sweep/, per-run records or the
# committed sweep_summary.csv). Without either, the driver falls back to 0.1 and
# says so loudly. To hold the paper's value regardless, pass --sleep-ratio 0.3.
#
#   sbatch slurm/run_ablation.sh --sleep-ratio 0.3          # MNIST grid, 85 runs
#   python src/ablation.py --collect
#
# The paper's other three datasets are a second grid of 255 runs; arguments after
# the script name are passed through to src/ablation.py:
#
#   sbatch --array=0-254 slurm/run_ablation.sh --sleep-ratio 0.3 \
#       --datasets fmnist,kmnist,notmnist --out-dir results/ablation_datasets
#   python src/ablation.py --collect --datasets fmnist,kmnist,notmnist \
#       --out-dir results/ablation_datasets
#
# Check the mapping before submitting:  python src/ablation.py --list [same args]
#
#SBATCH --job-name=snn_ablation
#SBATCH --array=0-84
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=orion
#SBATCH --output=slurm/logs/ablation_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/andreama/projects/SNN-plots-dec08}"
cd "${PROJECT_ROOT}" || { echo "ERROR: cannot cd to ${PROJECT_ROOT}" >&2; exit 1; }

mkdir -p slurm/logs results/ablation

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg

echo "========================================"
echo "Job    : ${SLURM_JOB_ID:-local}   Task : ${TASK_ID}"
echo "Node   : $(hostname)   Started : $(date)"
echo "Root   : ${PROJECT_ROOT}"
echo "========================================"

if [ -z "$(ls -A results/sweep/*.json 2>/dev/null)" ] && [ ! -f results/sweep/sweep_summary.csv ]; then
    echo "WARNING: results/sweep/ is empty — the sleep ratio will fall back to" >&2
    echo "         0.1 rather than the sweep optimum. Run the sweep first." >&2
fi

if [ ! -f noise_env.sif ]; then
    echo "ERROR: noise_env.sif not found." >&2
    echo "Build it ONCE on the login node before submitting an array:" >&2
    echo "  singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3" >&2
    echo "  singularity exec noise_env.sif conda env create -f environment_linux.yml -n noise_env" >&2
    exit 1
fi

singularity exec noise_env.sif conda run -n noise_env python \
    src/ablation.py --task-id "${TASK_ID}" "$@"

STATUS=$?
echo "Finished : $(date)  (exit ${STATUS})"
exit ${STATUS}
