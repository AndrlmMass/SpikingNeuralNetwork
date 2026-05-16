#!/bin/bash
# Phase 1 — regularizer comparison.
#
# 5 conditions × 5 seeds = 25 independent runs.
# Submit with:
#   bash experiments/submit_phase1.sh
#
# Conditions (by SLURM_ARRAY_TASK_ID // 5):
#   0  sleep     / static
#   1  sleep     / adaptive
#   2  normalize / static
#   3  normalize / adaptive
#   4  none      (baseline)
#
# Sleep duration and noise are held fixed at their main.py defaults so
# only the regularizer type/mode varies.
#
#SBATCH --job-name=snn_p1
#SBATCH --array=0-24
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

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONDITION_IDX=$(( TASK_ID / 5 ))
SEED=$(( TASK_ID % 5 ))

REG_TYPES=(sleep     sleep     normalize normalize none)
REG_MODES=(static    adaptive  static    adaptive  static)

REG_TYPE=${REG_TYPES[$CONDITION_IDX]}
REG_MODE=${REG_MODES[$CONDITION_IDX]}

# Fixed across all Phase 1 runs
SLEEP_DURATION=100
VAR_NOISE=0.1

OUTPUT_DIR="${PROJECT_ROOT}/experiments/results/phase1/${REG_TYPE}_${REG_MODE}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "condition=${REG_TYPE}/${REG_MODE}  seed=${SEED}"
echo "sleep_dur=${SLEEP_DURATION}  var_noise=${VAR_NOISE}"
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
