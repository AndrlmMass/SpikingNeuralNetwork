#!/bin/bash
# FROZEN CONTROL for phase 2: the byte-identical supervised run (oriented RFs, grouped/tiled,
# dense softmax-delta readout, same splits and test sets as run_slurm.sh) with R-STDP
# switched OFF via --reward-lr 0. W_se never changes; only the readout learns.
#
# Replaces results/frozen_supervised/run_cheap, which used 3 seeds, 55k/11k training
# images and a different driver (run_frozen.py), so it was not like-for-like with the
# 5-seed plastic arms.
#
# Grid: 4 datasets x 5 seeds = 20 runs (SVHN dropped from the article).
#   task_id -> seed = task_id % 5, dataset = task_id // 5
#
# Submit from PROJECT_ROOT:
#   sbatch experiments/RF_article/interp/mnist_family_sweep/run_slurm_frozen.sh

#SBATCH --job-name=rf_phase2_frozen
#SBATCH --array=0-19
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

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn4
cd "${PROJECT_ROOT}"

# ---- grid -----------------------------------------------------------------
# SVHN LAST: dense natural images fire ~3-4x more spikes/epoch than the sparse MNIST
# family (rate-coded cost scales with total spikes) and it carries the full 26032 test
# set, so it is decisively the slowest. Highest task IDs -> scheduled last.
DATASETS=(mnist fmnist kmnist notmnist)
PRIORS=(oriented)
N_SEEDS=5
N_PRIOR=${#PRIORS[@]}
N_DS=${#DATASETS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$(( TASK_ID % N_SEEDS ))
CELL=$(( TASK_ID / N_SEEDS ))
PRIOR_IDX=$(( CELL % N_PRIOR ))
DS_IDX=$(( CELL / N_PRIOR ))

if [ "${DS_IDX}" -ge "${N_DS}" ]; then
    echo "FATAL: DS_IDX=${DS_IDX} out of range (have ${N_DS} datasets); check --array range." >&2
    exit 1
fi
DATASET=${DATASETS[$DS_IDX]}
PRIOR=${PRIORS[$PRIOR_IDX]}
TAG="${DATASET}_frozen_s${SEED}"

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
echo "dataset=${DATASET}  prior=${PRIOR} (FROZEN, reward-lr 0)  seed=${SEED}  epochs=3"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- skip only GENUINELY-complete runs (results.json with a finite test_acc) -----
is_complete() {
    conda run --no-capture-output -n noise_env python - "$1" <<'PY' 2>/dev/null
import json,sys,math
try:
    d=json.load(open(sys.argv[1]))
    a=d.get("test_acc")
    sys.exit(0 if isinstance(d,dict) and isinstance(a,(int,float)) and math.isfinite(a) else 1)
except Exception:
    sys.exit(1)
PY
}

SIF="${PROJECT_ROOT}/noise_env.sif"
if [ ! -f "${SIF}" ]; then
    echo "FATAL: container image not found at ${SIF}" >&2
    exit 1
fi
if singularity exec "${SIF}" bash -c "$(declare -f is_complete); is_complete '${OUTPUT_DIR}/results.json'"; then
    echo "results.json already complete — skipping."
    exit 0
fi

# ---- per-dataset dedicated split sizes (see neurosnn/_data/_partition_indices) ---
# train/val from the dedicated TRAIN split, test = the dedicated TEST split (full).
#   mnist/fmnist/kmnist : 60000 train / 10000 test -> 59k train + 1k val, FULL 10k test
#   svhn                : 73257 train / 26032 test -> 59k train + 1k val, FULL 26032 test
#   notmnist            : no canonical split; 18724 merged -> 15k / 1k / 2724 (seed-fixed)
TRAIN_ALL=59000; VAL_ALL=1000; TEST_ALL=10000
case "${DATASET}" in
    svhn)     TRAIN_ALL=59000; VAL_ALL=1000; TEST_ALL=26032 ;;
    notmnist) TRAIN_ALL=15000; VAL_ALL=1000; TEST_ALL=2724  ;;
esac
echo "Split    : train=${TRAIN_ALL} val=${VAL_ALL} test=${TEST_ALL}  probe-fit=5000  (dataset=${DATASET})"

# ---- run: canonical 95% config, R-STDP + {oriented|random} RFs, delta readout ----
singularity exec "${SIF}" conda run --no-capture-output -n noise_env python -u \
    experiments/RF_article/interp/interp_harness.py \
    --tag           "${TAG}" \
    --dataset       "${DATASET}" \
    --seed          "${SEED}" \
    --prior         "${PRIOR}" \
    --rule          reward \
    --grouped --group-layout block --tiled \
    --dense-readout --readout-lr 0.1 \
    --reward-lr     0 \
    --peak-ei       50 \
    --rf-length     3.0 \
    --rf-thickness  1.2 \
    --center-margin 4.0 \
    --epochs        3 \
    --train-all     "${TRAIN_ALL}" \
    --val-all       "${VAL_ALL}" \
    --test-all      "${TEST_ALL}" \
    --probe-fit-all 5000 \
    --output-dir    "${OUTPUT_DIR}"

echo "Finished : $(date)"
