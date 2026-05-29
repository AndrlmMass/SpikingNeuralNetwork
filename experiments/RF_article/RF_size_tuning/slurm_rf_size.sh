#!/bin/bash
# RF size tuning sweep — mean and log-normal std for SE and EE receptive fields.
#
# Grid: 10 RF means × 10 log-normal stds × 5 seeds = 500 independent runs.
# Each run trains for 1 epoch on MNIST.
#
# RF means (pixels)     : 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.0
# Log-normal stds       : 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5
# Seeds                 : 0 1 2 3 4
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0–499):
#   seed      = task_id % 5
#   combo_idx = task_id // 5
#   std_idx   = combo_idx % 10
#   mean_idx  = combo_idx // 10
#
# Submit with:
#   sbatch experiments/RF_article/RF_size_tuning/slurm_rf_size.sh

#SBATCH --job-name=snn_rf_size
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn
cd "${PROJECT_ROOT}"

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % 5 ))
COMBO_IDX=$(( TASK_ID / 5 ))
STD_IDX=$(( COMBO_IDX % 10 ))
MEAN_IDX=$(( COMBO_IDX / 10 ))

RF_MEANS=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.0)
RF_STDS=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5)

RF_MEAN=${RF_MEANS[$MEAN_IDX]}
RF_STD=${RF_STDS[$STD_IDX]}

OUTPUT_DIR="${PROJECT_ROOT}/experiments/RF_article/RF_size_tuning/results/mean${RF_MEAN}_std${RF_STD}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "rf_mean=${RF_MEAN}  rf_lognorm_std=${RF_STD}  seed=${SEED}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- singularity env (no-op if already built) -----------------------------
if [ ! -f noise_env.sif ]; then
    singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
    singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
fi

# ---- run ------------------------------------------------------------------
singularity exec noise_env.sif conda run -n noise_env python \
    experiments/RF_article/RF_size_tuning/run_experiment.py \
    --rf-mean        "${RF_MEAN}" \
    --rf-lognorm-std "${RF_STD}" \
    --seed           "${SEED}" \
    --output-dir     "${OUTPUT_DIR}"

echo "Finished : $(date)"
