#!/bin/bash
# Phase 2 — sleep duration × membrane noise grid.
#
# 8 durations × 8 noise levels × 5 seeds = 320 independent runs.
# Submit with:
#   bash experiments/submit_phase2.sh
#
# Encoding:
#   grid_idx  = TASK_ID // 5          (0-63)
#   seed      = TASK_ID %  5          (0-4)
#   dur_idx   = grid_idx // 8         (0-7)
#   noise_idx = grid_idx %  8         (0-7)
#
# Regularizer is fixed to sleep/static (best condition from Phase 1).
# Override BEST_REG_TYPE / BEST_REG_MODE via environment if needed:
#   BEST_REG_TYPE=normalize BEST_REG_MODE=adaptive bash experiments/submit_phase2.sh
#
#SBATCH --job-name=snn_p2
#SBATCH --array=0-319%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -euo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn
cd "${PROJECT_ROOT}"

# ---- parameter grid -------------------------------------------------------
SLEEP_DURATIONS=(50 100 150 200 300 400 500 600)
VAR_NOISES=(0.0 0.1 0.5 1.0 2.0 4.0 6.0 8.0)

# Regularizer fixed (override after Phase 1 results are in)
REG_TYPE=${BEST_REG_TYPE:-sleep}
REG_MODE=${BEST_REG_MODE:-static}

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
GRID_IDX=$(( TASK_ID / 5 ))
SEED=$(( TASK_ID % 5 ))
DUR_IDX=$(( GRID_IDX / 8 ))
NOISE_IDX=$(( GRID_IDX % 8 ))

SLEEP_DURATION=${SLEEP_DURATIONS[$DUR_IDX]}
VAR_NOISE=${VAR_NOISES[$NOISE_IDX]}

OUTPUT_DIR="${PROJECT_ROOT}/experiments/results/phase2/sd${SLEEP_DURATION}_vn${VAR_NOISE}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "grid_idx=${GRID_IDX}  dur_idx=${DUR_IDX}  noise_idx=${NOISE_IDX}"
echo "sleep_dur=${SLEEP_DURATION}  var_noise=${VAR_NOISE}  seed=${SEED}"
echo "reg=${REG_TYPE}/${REG_MODE}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- singularity env (no-op if already built) -----------------------------
if [ ! -f noise_env.sif ]; then
    singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
    singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
fi

# ---- run ------------------------------------------------------------------
singularity exec noise_env.sif conda run -n noise_env python \
    experiments/run_experiment.py \
    --reg-type       "${REG_TYPE}" \
    --reg-mode       "${REG_MODE}" \
    --sleep-duration "${SLEEP_DURATION}" \
    --var-noise      "${VAR_NOISE}" \
    --seed           "${SEED}" \
    --output-dir     "${OUTPUT_DIR}"

echo "Finished : $(date)"
