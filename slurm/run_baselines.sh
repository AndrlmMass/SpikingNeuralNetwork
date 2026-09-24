#!/bin/bash
# Conventional-stabilization baselines for the Neurocomputing revision
# (Reviewer 3, Major Point 1).
#
# Grid: 5 regularization methods x 4 datasets x 5 seeds = 100 independent runs.
#
#   none         unregularized STDP (divergent reference)
#   sleep        this paper's power-law decay toward an absolute target
#   decay        continuous multiplicative shrinkage, every timestep, no target
#   norm_layer   instantaneous layer-wise rescale to the initial total |w|
#   norm_neuron  synaptic scaling (per postsynaptic neuron)
#
# Unlike the earlier sweep scripts, the array index is NOT decoded here.
# experiment.py owns the grid and maps a single --task-id onto one cell, so the
# two can never disagree about what job 57 is. Check the mapping before
# submitting with:
#
#   python src/experiment.py --list
#
# Submit with:
#   sbatch slurm/run_baselines.sh
#
# Then gather the finished cells into one CSV for the GLMM:
#   python src/experiment.py --collect
#
# Rerun individual failures by id:
#   sbatch --array=57,63 slurm/run_baselines.sh
#
#SBATCH --job-name=snn_baselines
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=orion
#SBATCH --output=slurm/logs/baselines_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

# Override with:  PROJECT_ROOT=/some/other/path sbatch slurm/run_baselines.sh
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/andreama/projects/SNN-plots-dec08}"
cd "${PROJECT_ROOT}" || { echo "ERROR: cannot cd to ${PROJECT_ROOT}" >&2; exit 1; }

mkdir -p slurm/logs results/baselines

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ---- threading ------------------------------------------------------------
# The matrix-vector products in this model are small (a few hundred rows), far
# below the size where threaded BLAS pays for itself: an unpinned OpenBLAS
# spawns one worker per core and then spends most of its time synchronizing
# them. Pinning BLAS to a single thread roughly halves per-timestep cost.
# Numba's prange kernels are a separate pool and do benefit from the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg

echo "========================================"
echo "Job    : ${SLURM_JOB_ID:-local}   Task : ${TASK_ID}"
echo "Node   : $(hostname)   Started : $(date)"
echo "Root   : ${PROJECT_ROOT}"
echo "Threads: BLAS=1  NUMBA=${NUMBA_NUM_THREADS}"
echo "========================================"

# ---- singularity env (no-op if already built) -----------------------------
if [ ! -f noise_env.sif ]; then
    echo "ERROR: noise_env.sif not found." >&2
    echo "Build it ONCE on the login node before submitting an array —" >&2
    echo "letting 100 tasks race the same build corrupts it:" >&2
    echo "  singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3" >&2
    echo "  singularity exec noise_env.sif conda env create -f environment_linux.yml -n noise_env" >&2
    exit 1
fi

# ---- run one grid cell ----------------------------------------------------
singularity exec noise_env.sif conda run -n noise_env python \
    src/experiment.py --task-id "${TASK_ID}"

STATUS=$?
echo "Finished : $(date)  (exit ${STATUS})"
exit ${STATUS}
