#!/bin/bash
# Phase 1 — receptive fields vs random weights comparison.
#
# 2 conditions × 10 seeds = 20 independent runs.
# Submit with:
#   sbatch experiments/RF_article/slurm_phase1.sh
#
# Conditions (by SLURM_ARRAY_TASK_ID // 10):
#   0  rf
#   1  random
#
#SBATCH --job-name=snn_rf
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn
cd "${PROJECT_ROOT}"

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONDITION_IDX=$(( TASK_ID / 10 ))
SEED=$(( TASK_ID % 10 ))

WEIGHT_TYPES=(rf random)
WEIGHT_TYPE=${WEIGHT_TYPES[$CONDITION_IDX]}

OUTPUT_DIR="${PROJECT_ROOT}/experiments/RF_article/results/${WEIGHT_TYPE}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "weight_type=${WEIGHT_TYPE}  seed=${SEED}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- singularity env (no-op if already built) -----------------------------
if [ ! -f noise_env.sif ]; then
    singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
    singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
fi

# ---- run ------------------------------------------------------------------
singularity exec noise_env.sif conda run -n noise_env python \
    experiments/RF_article/run_experiment.py \
    --weight-type "${WEIGHT_TYPE}" \
    --seed        "${SEED}" \
    --output-dir  "${OUTPUT_DIR}"

echo "Finished : $(date)"
