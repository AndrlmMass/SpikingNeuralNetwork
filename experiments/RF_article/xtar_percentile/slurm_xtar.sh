#!/bin/bash
# x_tar percentile sweep — SE × EE threshold grid vs the mean baseline.
#
# Grid: 9 SE percentiles × 9 EE percentiles = 81 cells, PLUS 1 "mean" baseline
#       cell, each run with N_SEEDS seeds. Default 82 cells × 3 seeds = 246 runs.
# Each run trains 1 epoch on full MNIST (59k images).
#
# SE / EE percentiles : 10 20 30 40 50 60 70 80 90
# Seeds               : 0 1 2   (N_SEEDS)
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0–245):
#   seed = task_id % N_SEEDS
#   cfg  = task_id // N_SEEDS          # 0..81
#   cfg == 81            -> mean baseline
#   cfg  < 81            -> se_idx = cfg // 9 ; ee_idx = cfg % 9 (percentile)
#
# To change resolution/seeds: edit SE_PCTS/EE_PCTS/N_SEEDS below AND the
# --array range = N_SEEDS * (len(SE_PCTS)*len(EE_PCTS) + 1) - 1.
#
# Build the container image ONCE first (see the container-image block below),
# then submit:
#   cd ${PROJECT_ROOT}
#   singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
#   singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
#   sbatch experiments/RF_article/xtar_percentile/slurm_xtar.sh

#SBATCH --job-name=snn_xtar
#SBATCH --array=0-245
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn2
cd "${PROJECT_ROOT}"

# ---- grid definition ------------------------------------------------------
SE_PCTS=(10 20 30 40 50 60 70 80 90)
EE_PCTS=(10 20 30 40 50 60 70 80 90)
N_SEEDS=3
N_SE=${#SE_PCTS[@]}
N_EE=${#EE_PCTS[@]}
N_GRID=$(( N_SE * N_EE ))         # 81 percentile cells
BASELINE_CFG=${N_GRID}            # cfg index reserved for the mean baseline

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
CFG=$(( TASK_ID / N_SEEDS ))

if [ "${CFG}" -ge "${BASELINE_CFG}" ]; then
    MODE="mean"
    PCT_SE=0
    PCT_EE=0
    TAG="mean_s${SEED}"
else
    MODE="percentile"
    SE_IDX=$(( CFG / N_EE ))
    EE_IDX=$(( CFG % N_EE ))
    PCT_SE=${SE_PCTS[$SE_IDX]}
    PCT_EE=${EE_PCTS[$EE_IDX]}
    TAG="se${PCT_SE}_ee${PCT_EE}_s${SEED}"
fi

OUTPUT_DIR="${PROJECT_ROOT}/experiments/RF_article/xtar_percentile/results/${TAG}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1
echo "========================================"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "mode=${MODE}  pct_se=${PCT_SE}  pct_ee=${PCT_EE}  seed=${SEED}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- skip if already done (resubmit the full array to fill only the gaps) --
# Remove this guard to force re-running completed cells.
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo "results.json already present — skipping."
    exit 0
fi

# ---- container image ------------------------------------------------------
# Build the image ONCE before submitting the array — never inside the job.
# Concurrent array tasks all building the same noise_env.sif race and clobber
# each other, producing "Build complete" immediately followed by
# "could not open image ... no such file". Pre-build, then fail loudly here if
# the image is missing so a broken task leaves an obvious error instead of an
# empty result dir.
#
#   cd ${PROJECT_ROOT}
#   singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
#   singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
#   ls -l ${PROJECT_ROOT}/noise_env.sif      # verify it resolves on the compute nodes
#
SIF="${PROJECT_ROOT}/noise_env.sif"
if [ ! -f "${SIF}" ]; then
    echo "FATAL: container image not found at ${SIF}" >&2
    echo "Build it once before submitting (see the comment block above)." >&2
    exit 1
fi

# ---- run ------------------------------------------------------------------
singularity exec "${SIF}" conda run -n noise_env python \
    experiments/RF_article/xtar_percentile/run_experiment.py \
    --x-tar-mode   "${MODE}" \
    --x-tar-pct-se "${PCT_SE}" \
    --x-tar-pct-ee "${PCT_EE}" \
    --seed         "${SEED}" \
    --output-dir   "${OUTPUT_DIR}"

echo "Finished : $(date)"
