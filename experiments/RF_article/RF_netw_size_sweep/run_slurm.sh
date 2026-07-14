#!/bin/bash
# Network-size sweep — scale (N_exc, N_inh) together with proportional inhibition.
#
# Grid: 3 sizes x N_SEEDS seeds. Default 3 x 3 = 9 runs. Each run trains 1 epoch on
# full MNIST (59k images).
#
#   sizes (N_exc / N_inh) : 1024/225  2116/450  4096/900
#   seeds                 : 0 1 2   (N_SEEDS)
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0..8):
#   seed     = task_id % N_SEEDS
#   size_idx = task_id // N_SEEDS          # 0,1,2 -> the three sizes
#
# To change resolution/seeds: edit SIZES_EXC/SIZES_INH/N_SEEDS below AND the
# --array range = N_SEEDS * len(SIZES_EXC) - 1.
#
# Build the container image ONCE first, then submit. You can either:
#   bash experiments/RF_article/RF_netw_size_sweep/run_slurm.sh     # NOT this — it's the job
#   sbatch experiments/RF_article/RF_netw_size_sweep/run_slurm.sh   # direct submit (UI/CLI)
# A direct sbatch still lands every task in ONE results/<RUN_ID> folder: RUN_ID is
# derived from the array job's SubmitTime via scontrol, identical across tasks.
#
# Output: results.json per cell, grouped under
#   results/<RUN_ID>/exc<N_EXC>_inh<N_INH>_s<SEED>/results.json
# SLURM .out logs go to results/<RUN_ID>/slurm_logs/.

#SBATCH --job-name=snn_size
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
# Fallback log location for direct/UI submissions. To drop SLURM logs entirely,
# set both of these to /dev/null.
#SBATCH --output=experiments/RF_article/RF_netw_size_sweep/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/RF_netw_size_sweep/results/slurm_logs/%A_%a.out

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn3
cd "${PROJECT_ROOT}"

# ---- grid definition ------------------------------------------------------
SIZES_EXC=(1024 2116 4096)
SIZES_INH=(225 450 900)
N_SEEDS=3
N_SIZES=${#SIZES_EXC[@]}

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
SIZE_IDX=$(( TASK_ID / N_SEEDS ))

if [ "${SIZE_IDX}" -ge "${N_SIZES}" ]; then
    echo "FATAL: SIZE_IDX=${SIZE_IDX} out of range (have ${N_SIZES} sizes); check --array range." >&2
    exit 1
fi

N_EXC=${SIZES_EXC[$SIZE_IDX]}
N_INH=${SIZES_INH[$SIZE_IDX]}
TAG="exc${N_EXC}_inh${N_INH}_s${SEED}"

# ---- one results folder per job (shared by all array tasks) ----------------
# RUN_ID resolution, in priority order:
#   1. Exported at submission time (e.g. --export=ALL,RUN_ID=...).
#   2. Derived from this array job's SubmitTime via scontrol — identical for every
#      task, so a UI/web submission still gets a single date+time folder.
#   3. Bare array job id, if scontrol can't answer.
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
RUN_DIR="${PROJECT_ROOT}/experiments/RF_article/RF_netw_size_sweep/results/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "N_exc=${N_EXC}  N_inh=${N_INH}  seed=${SEED}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- skip if already done (resubmit the full array to fill only the gaps) --
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo "results.json already present — skipping."
    exit 0
fi

# ---- container image ------------------------------------------------------
# Build the image ONCE before submitting the array — never inside the job
# (concurrent builds race and clobber each other). Fail loudly if it's missing.
#   cd ${PROJECT_ROOT}
#   singularity build --fakeroot noise_env.sif docker://continuumio/miniconda3
#   singularity exec noise_env.sif conda env create -f environment.yml -n noise_env
SIF="${PROJECT_ROOT}/noise_env.sif"
if [ ! -f "${SIF}" ]; then
    echo "FATAL: container image not found at ${SIF}" >&2
    echo "Build it once before submitting (see the comment block above)." >&2
    exit 1
fi

# ---- run ------------------------------------------------------------------
singularity exec "${SIF}" conda run -n noise_env python \
    experiments/RF_article/RF_netw_size_sweep/run_experiment.py \
    --n-exc      "${N_EXC}" \
    --n-inh      "${N_INH}" \
    --seed       "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --jsonl      "${RUN_DIR}/size_sweep.jsonl"

echo "Finished : $(date)"
