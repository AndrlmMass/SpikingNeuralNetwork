#!/bin/bash
# Phase 2 — sleep duration × membrane noise × reg-mode grid.
#
# 8 durations × 8 noise levels × 3 reg-modes = 192 independent runs (seed fixed to 0).
# Submit with:
#   bash experiments/submit_phase2.sh
#
# Encoding (TASK_ID = 0-191):
#   mode_idx  = TASK_ID %  3          (0-2)
#   cell_idx  = TASK_ID // 3          (0-63)
#   dur_idx   = cell_idx // 8         (0-7)
#   noise_idx = cell_idx %  8         (0-7)
#
# Regularizer type is fixed to sleep; mode varies across static/layer/neuron.
#
#SBATCH --job-name=snn_p2
#SBATCH --array=0-191%20
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

# ---- parameter grid -------------------------------------------------------
SLEEP_DURATIONS=(5 10 25 50 100 150 200 400)
VAR_NOISES=(0.0 0.5 1.0 5.0 10.0 25.0 50.0 100.0)
REG_MODES=(static layer neuron)

# Regularizer type is fixed to sleep for all Phase 2 runs
REG_TYPE=sleep

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=0
MODE_IDX=$(( TASK_ID % 3 ))
CELL_IDX=$(( TASK_ID / 3 ))
DUR_IDX=$(( CELL_IDX / 8 ))
NOISE_IDX=$(( CELL_IDX % 8 ))

SLEEP_DURATION=${SLEEP_DURATIONS[$DUR_IDX]}
VAR_NOISE=${VAR_NOISES[$NOISE_IDX]}
REG_MODE=${REG_MODES[$MODE_IDX]}

OUTPUT_DIR="${PROJECT_ROOT}/experiments/noise_article/sleep_noise_optimization/results/phase2/${REG_MODE}_sd${SLEEP_DURATION}_vn${VAR_NOISE}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "cell_idx=${CELL_IDX}  dur_idx=${DUR_IDX}  noise_idx=${NOISE_IDX}  mode_idx=${MODE_IDX}"
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
    experiments/noise_article/sleep_noise_optimization/run_sleep_tuning.py \
    --reg-type       "${REG_TYPE}" \
    --reg-mode       "${REG_MODE}" \
    --sleep-duration "${SLEEP_DURATION}" \
    --var-noise      "${VAR_NOISE}" \
    --seed           "${SEED}" \
    --output-dir     "${OUTPUT_DIR}"

echo "Finished : $(date)"
