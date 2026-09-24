#!/bin/bash
# Sleep-ratio sweep — re-run of the paper's main MNIST-family experiment.
#
# 11 sleep ratios (0%-100%) x 4 datasets x 5 seeds = 220 independent runs.
#
# The published sweep capped every ratio at or above 28.6% to exactly 28.57%
# (sleep_max_iters=10000 against a window of round(35000 * ratio)), so
# beta_30..beta_100 in Table 1 are eight estimates of one condition. This re-run
# scales check_sleep_interval with num_steps, which keeps the largest window
# (3500) below the cap and makes all 11 ratios faithful. See src/sweep.py.
#
# The array index is NOT decoded here: src/sweep.py owns the grid and maps one
# --task-id onto one cell, so the two cannot disagree. Check the mapping first:
#
#   python src/sweep.py --list
#
# Submit:      sbatch slurm/run_sweep.sh
# Collect:     python src/sweep.py --collect
# Rerun some:  sbatch --array=57,63 slurm/run_sweep.sh
#
# Measured ~2.3 min for a ratio-0 cell and ~1.47x that at ratio 1.0, so the
# 30-minute limit is generous. Peak RSS ~1.5 GB.
#
#SBATCH --job-name=snn_sweep
#SBATCH --array=0-219
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=orion
#SBATCH --output=slurm/logs/sweep_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

# Override with:  PROJECT_ROOT=/some/other/path sbatch slurm/run_sweep.sh
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/andreama/projects/SNN-plots-dec08}"
cd "${PROJECT_ROOT}" || { echo "ERROR: cannot cd to ${PROJECT_ROOT}" >&2; exit 1; }

mkdir -p slurm/logs results/sweep

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# The matvec here is ~250x475 — far below the size where threaded BLAS pays for
# itself, and an unpinned OpenBLAS spawns one worker per core and spends most of
# its time synchronizing them. One BLAS thread per task, many tasks.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg

echo "========================================"
echo "Job    : ${SLURM_JOB_ID:-local}   Task : ${TASK_ID}"
echo "Node   : $(hostname)   Started : $(date)"
echo "Root   : ${PROJECT_ROOT}"
echo "Threads: BLAS=1  NUMBA=${NUMBA_NUM_THREADS}"
echo "========================================"

if [ ! -f noise_env.sif ]; then
    echo "ERROR: noise_env.sif not found." >&2
    echo "Build it ONCE on the login node before submitting an array —" >&2
    echo "letting 220 tasks race the same build corrupts it:" >&2
    echo "  singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3" >&2
    echo "  singularity exec noise_env.sif conda env create -f environment_linux.yml -n noise_env" >&2
    exit 1
fi

singularity exec noise_env.sif conda run -n noise_env python \
    src/sweep.py --task-id "${TASK_ID}"

STATUS=$?
echo "Finished : $(date)  (exit ${STATUS})"
exit ${STATUS}
