#!/bin/bash
# MNIST-family dataset sweep for the canonical 95% setup: oriented elongated RFs +
# reward-STDP (R-STDP) on the excitatory layer + the dense softmax-delta readout
# (the "artificial" readout that reached 95.5% on MNIST 60k x 5ep).
#
# The question: how well does THIS network + prior + reward rule + delta readout
# carry over from MNIST to the rest of the MNIST family? All three datasets are
# 28x28 grayscale 10-class, so N_x=784 and the architecture is byte-identical --
# only the task changes.
#
# Grid: 3 datasets x 3 seeds = 9 runs, 5 epochs each.
#   datasets : mnist kmnist fmnist
#   seeds    : 0 1 2
#
# Readout: the DENSE softmax-delta readout only (--dense-readout). The spiking
# R-STDP readout is a separate line of work and is deliberately NOT in this sweep.
#
# Task ID (SLURM_ARRAY_TASK_ID in 0..8):
#   seed  = task_id % N_SEEDS
#   ds    = task_id // N_SEEDS
#
# RAM note: each array task is an independent process on its own allocation, so
# seeds do NOT share memory -- 3 vs 5 seeds changes total queue time, not per-node
# RAM. One task loads one dataset (~60k images) exactly like the local MNIST runs.
#
# !! PRE-SUBMISSION (once, on a NETWORKED login node, from PROJECT_ROOT) !!
#   1. Pre-cache torchvision so concurrent array tasks don't race the download:
#        for d in MNIST KMNIST FashionMNIST; do
#          singularity exec noise_env.sif conda run -n noise_env python -c \
#            "from torchvision import datasets; getattr(datasets,'$d')(root='data/torchvision', train=True, download=True); getattr(datasets,'$d')(root='data/torchvision', train=False, download=True)"
#        done
#   2. Confirm noise_env.sif exists (reuse from the other sweeps).

#SBATCH --job-name=mnist_family
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
#SBATCH --output=experiments/RF_article/interp/mnist_family_sweep/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/interp/mnist_family_sweep/results/slurm_logs/%A_%a.out

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn3
cd "${PROJECT_ROOT}"

# ---- grid -----------------------------------------------------------------
DATASETS=(mnist kmnist fmnist)
N_SEEDS=3
N_DS=${#DATASETS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
DS_IDX=$(( TASK_ID / N_SEEDS ))

if [ "${DS_IDX}" -ge "${N_DS}" ]; then
    echo "FATAL: DS_IDX=${DS_IDX} out of range (have ${N_DS} datasets); check --array range." >&2
    exit 1
fi

DATASET=${DATASETS[$DS_IDX]}
TAG="${DATASET}_5ep_s${SEED}"

# ---- one results folder per submission, shared by all array tasks ----------
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
RUN_DIR="${PROJECT_ROOT}/experiments/RF_article/interp/mnist_family_sweep/results/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "dataset=${DATASET}  seed=${SEED}  epochs=5"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- skip if already done (resubmit the array to fill only the gaps) -------
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo "results.json already present — skipping."
    exit 0
fi

SIF="${PROJECT_ROOT}/noise_env.sif"
if [ ! -f "${SIF}" ]; then
    echo "FATAL: container image not found at ${SIF}" >&2
    exit 1
fi

# ---- run: canonical 95% config, R-STDP + oriented RFs, delta readout, 5 epochs -
singularity exec "${SIF}" conda run --no-capture-output -n noise_env python -u \
    experiments/RF_article/interp/interp_harness.py \
    --tag           "${TAG}" \
    --dataset       "${DATASET}" \
    --seed          "${SEED}" \
    --prior         oriented \
    --rule          reward \
    --grouped --group-layout block --tiled \
    --dense-readout --readout-lr 0.1 \
    --reward-lr     5e-6 \
    --peak-ei       50 \
    --rf-length     3.0 \
    --rf-thickness  1.2 \
    --center-margin 4.0 \
    --epochs        5 \
    --train-all     59000 \
    --val-all       1000 \
    --test-all      10000 \
    --output-dir    "${OUTPUT_DIR}"

echo "Finished : $(date)"
