#!/bin/bash
# x_tar sweep — all three estimators in ONE array: percentile grid, static grid,
# and the mean baseline.
#
# Grid:
#   * percentile : 9 SE × 9 EE = 81 cells
#   * mean       : 1 baseline cell
#   * static     : 6 SE × 6 EE = 36 cells  (fixed absolute trace thresholds)
#   => 118 config cells × N_SEEDS seeds. Default 118 × 3 = 354 runs.
# Each run trains 1 epoch on full MNIST (59k images).
#
# SE / EE percentiles : 10 20 30 40 50 60 70 80 90
# SE static levels    : 0.05 0.1 0.2 0.4 0.8 1.5   (brackets active-input trace:
#                       mean-mode x_tar_se ~0.21, active-trace p90 ~1.1, p99 ~3.1)
# EE static levels    : 0.02 0.05 0.1 0.2 0.4 0.7   (mean-mode x_tar_ee ~0.09,
#                       active-trace p90 ~0.38, p99 ~1.1)
# Seeds               : 0 1 2   (N_SEEDS)
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0–353):
#   seed = task_id % N_SEEDS
#   cfg  = task_id // N_SEEDS          # 0..117
#   cfg  < 81            -> percentile: se_idx = cfg // 9 ; ee_idx = cfg % 9
#   cfg == 81            -> mean baseline
#   cfg >= 82            -> static: s = cfg - 82 ; se_idx = s // 6 ; ee_idx = s % 6
#
# To change resolution/seeds: edit SE_PCTS/EE_PCTS/SE_STAT/EE_STAT/N_SEEDS below
# AND the --array range = N_SEEDS * (81 + 1 + len(SE_STAT)*len(EE_STAT)) - 1.
#
# Build the container image ONCE first (see the container-image block below),
# then submit via the wrapper so the whole array lands in ONE timestamped
# results folder:
#   cd ${PROJECT_ROOT}
#   singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
#   singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
#   bash experiments/RF_article/xtar_percentile/submit_xtar.sh
#
# You can still `sbatch slurm_xtar.sh` directly (e.g. from a web UI). In that
# case RUN_ID isn't exported and the run folder falls back to the array job id,
# which is still identical across all 354 tasks.
#
# Output: only results.json per config, all grouped under
#   results/<RUN_ID>/<config>/results.json
# SLURM .out logs go to results/<RUN_ID>/slurm_logs/ (a subfolder you can
# ignore or delete), NOT the submission directory.

#SBATCH --job-name=snn_xtar
#SBATCH --array=0-353%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
# Fallback log location for direct/UI submissions (the wrapper overrides this
# to land inside results/<RUN_ID>/slurm_logs/). To drop SLURM logs entirely,
# set both of these to /dev/null.
#SBATCH --output=experiments/RF_article/xtar_percentile/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/xtar_percentile/results/slurm_logs/%A_%a.out

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn2
cd "${PROJECT_ROOT}"

# ---- grid definition ------------------------------------------------------
SE_PCTS=(10 20 30 40 50 60 70 80 90)
EE_PCTS=(10 20 30 40 50 60 70 80 90)
SE_STAT=(0.05 0.1 0.2 0.4 0.8 1.5)      # fixed input->exc trace thresholds
EE_STAT=(0.02 0.05 0.1 0.2 0.4 0.7)     # fixed exc->exc trace thresholds
N_SEEDS=3
N_SE=${#SE_PCTS[@]}
N_EE=${#EE_PCTS[@]}
N_GRID=$(( N_SE * N_EE ))         # 81 percentile cells
BASELINE_CFG=${N_GRID}            # cfg index reserved for the mean baseline (81)
STATIC_CFG0=$(( BASELINE_CFG + 1 ))   # first static cfg index (82)
N_SE_STAT=${#SE_STAT[@]}
N_EE_STAT=${#EE_STAT[@]}

# defaults (overridden per-branch); keep the unused mode's args at safe values
PCT_SE=0
PCT_EE=0
STAT_SE=0.2
STAT_EE=0.2

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
CFG=$(( TASK_ID / N_SEEDS ))

if [ "${CFG}" -lt "${N_GRID}" ]; then
    MODE="percentile"
    SE_IDX=$(( CFG / N_EE ))
    EE_IDX=$(( CFG % N_EE ))
    PCT_SE=${SE_PCTS[$SE_IDX]}
    PCT_EE=${EE_PCTS[$EE_IDX]}
    TAG="se${PCT_SE}_ee${PCT_EE}_s${SEED}"
elif [ "${CFG}" -eq "${BASELINE_CFG}" ]; then
    MODE="mean"
    TAG="mean_s${SEED}"
else
    MODE="static"
    S=$(( CFG - STATIC_CFG0 ))
    SE_IDX=$(( S / N_EE_STAT ))
    EE_IDX=$(( S % N_EE_STAT ))
    STAT_SE=${SE_STAT[$SE_IDX]}
    STAT_EE=${EE_STAT[$EE_IDX]}
    # match run_experiment.build_output_dir static tag: levels x1000, integer
    SE_TAG=$(awk "BEGIN{printf \"%d\", ${STAT_SE}*1000}")
    EE_TAG=$(awk "BEGIN{printf \"%d\", ${STAT_EE}*1000}")
    TAG="stat_se${SE_TAG}_ee${EE_TAG}_s${SEED}"
fi

# ---- one results folder per job (shared by all 354 array tasks) -----------
# RUN_ID resolution, in priority order:
#   1. Exported by submit_xtar.sh at submission time (terminal path).
#   2. Derived from this array job's SubmitTime via scontrol — identical for
#      every task, so a UI/web submission (which can't run the wrapper) still
#      gets a single date+time folder.
#   3. Bare array job id, if scontrol can't answer (job already purged, etc.).
# This means it never scatters into one-folder-per-task or per-config stamps,
# whether you submit from the terminal OR the UI.
if [ -z "${RUN_ID:-}" ]; then
    SUBMIT_TIME=$(scontrol show job "${SLURM_ARRAY_JOB_ID}" -o 2>/dev/null \
        | grep -oE 'SubmitTime=[^ ]+' | head -1 | cut -d= -f2)
    if [ -n "${SUBMIT_TIME}" ]; then
        RUN_ID="run_$(date -d "${SUBMIT_TIME}" +%Y%m%d_%H%M%S 2>/dev/null \
            || echo "${SUBMIT_TIME//[:-]/}")"
    else
        RUN_ID="job_${SLURM_ARRAY_JOB_ID}"
    fi
fi
RUN_DIR="${PROJECT_ROOT}/experiments/RF_article/xtar_percentile/results/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header (to the SLURM .out, no separate job.log) -------------------
echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "mode=${MODE}  pct_se=${PCT_SE}  pct_ee=${PCT_EE}  stat_se=${STAT_SE}  stat_ee=${STAT_EE}  seed=${SEED}"
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
    --x-tar-mode      "${MODE}" \
    --x-tar-pct-se    "${PCT_SE}" \
    --x-tar-pct-ee    "${PCT_EE}" \
    --x-tar-static-se "${STAT_SE}" \
    --x-tar-static-ee "${STAT_EE}" \
    --seed            "${SEED}" \
    --output-dir      "${OUTPUT_DIR}"

echo "Finished : $(date)"
