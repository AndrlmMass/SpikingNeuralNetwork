#!/bin/bash
# Dataset sweep — canonical 1024/225 oriented_rf network across static-image datasets,
# trained (TraceSTDP) vs frozen (initial weights fixed). The trained-vs-frozen A/B is the
# objective: does plasticity help, and does it help more where there is more headroom?
#
# Grid: 5 datasets x 2 conditions x N_SEEDS seeds. Default 5 x 2 x 3 = 30 runs, 1 epoch each.
#
#   datasets   : mnist kmnist fmnist cifar10 svhn
#   conditions : trained frozen
#   seeds      : 0 1 2   (N_SEEDS)
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0..23):
#   seed     = task_id % N_SEEDS
#   cell     = task_id // N_SEEDS         # 0..7
#   cond_idx = cell % N_COND             # 0=trained, 1=frozen
#   ds_idx   = cell // N_COND            # 0,1,2,3 -> the four datasets
#
# To change resolution/seeds: edit DATASETS/CONDITIONS/N_SEEDS below AND the
# --array range = N_SEEDS * len(DATASETS) * len(CONDITIONS) - 1.
#
# !! PRE-SUBMISSION (once, on a NETWORKED login node, from PROJECT_ROOT) !!
#   1. Pre-cache torchvision (avoids concurrent download races inside the array):
#        for d in mnist kmnist fmnist cifar10 svhn; do
#          singularity exec noise_env.sif conda run -n noise_env python \
#            experiments/RF_article/RF_dataset_sweep/run_experiment.py \
#            --dataset $d --seed 0 --train-all 200 --val-all 50 --test-all 50 \
#            --output-dir /tmp/precache_$d
#        done
#      Note: CIFAR-10 download ~170 MB, SVHN ~400 MB — do this on a node with internet.
#   2. Confirm noise_env.sif exists (reuse from the size sweep / biosnn run).
#
# Output: results.json per cell, grouped under
#   results/<RUN_ID>/<dataset>_s<SEED>/results.json
# SLURM .out logs go to results/<RUN_ID>/slurm_logs/.

#SBATCH --job-name=snn_dataset
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
# Fallback log location for direct/UI submissions. To drop SLURM logs entirely,
# set both of these to /dev/null.
#SBATCH --output=experiments/RF_article/RF_dataset_sweep/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/RF_dataset_sweep/results/slurm_logs/%A_%a.out

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn3
cd "${PROJECT_ROOT}"

# ---- grid definition ------------------------------------------------------
DATASETS=(mnist kmnist fmnist cifar10 svhn)
CONDITIONS=(trained frozen)
N_SEEDS=3
N_DS=${#DATASETS[@]}
N_COND=${#CONDITIONS[@]}

# ---- decode array task id -------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
CELL=$(( TASK_ID / N_SEEDS ))
COND_IDX=$(( CELL % N_COND ))
DS_IDX=$(( CELL / N_COND ))

if [ "${DS_IDX}" -ge "${N_DS}" ]; then
    echo "FATAL: DS_IDX=${DS_IDX} out of range (have ${N_DS} datasets); check --array range." >&2
    exit 1
fi

DATASET=${DATASETS[$DS_IDX]}
CONDITION=${CONDITIONS[$COND_IDX]}
TAG="${DATASET}_${CONDITION}_s${SEED}"

# trained -> learn weights; frozen -> pass --freeze-weights
FREEZE_FLAG=""
if [ "${CONDITION}" = "frozen" ]; then
    FREEZE_FLAG="--freeze-weights"
fi

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
RUN_DIR="${PROJECT_ROOT}/experiments/RF_article/RF_dataset_sweep/results/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

# ---- log header -----------------------------------------------------------
echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "dataset=${DATASET}  condition=${CONDITION}  seed=${SEED}"
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
# --no-capture-output + python -u stream stdout/stderr live to the .out file.
# Without these, conda run buffers everything and a killed process (OOM/timeout)
# leaves only the bash header — hiding the real error.
singularity exec "${SIF}" conda run --no-capture-output -n noise_env python -u \
    experiments/RF_article/RF_dataset_sweep/run_experiment.py \
    --dataset    "${DATASET}" \
    --seed       "${SEED}" \
    ${FREEZE_FLAG} \
    --output-dir "${OUTPUT_DIR}" \
    --jsonl      "${RUN_DIR}/dataset_sweep.jsonl"

echo "Finished : $(date)"
