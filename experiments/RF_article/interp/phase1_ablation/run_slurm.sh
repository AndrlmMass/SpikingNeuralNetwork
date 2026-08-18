#!/bin/bash
# PHASE 1 -- the UNSUPERVISED model: does plasticity help, and does it erode the
# oriented receptive-field prior? Ablation study on the recurrent, NON-grouped
# network (input -> excitatory with E->E recurrency -> inhibitory -> back to
# excitatory; N_exc=N_inh=1024). One ablation at a time off a trace-STDP baseline.
#
# This is the "strip the architecture back to find what matters" phase from the
# paper's narrative arc; the supervised R-STDP model is phase 2 (mnist_family_sweep).
#
# CONDITIONS (7), one departure from baseline each:
#   0 base_ori  oriented + trace-STDP + E->E on + I->E on + vogels off   (baseline)
#   1 base_rnd  RANDOM weights + trace-STDP (+ E->E, I->E)               (the "RFs end
#               up worse than random" headline: baseline vs random control)
#   2 triplet   oriented, TRIPLET-STDP instead of trace
#   3 frozen    oriented, NO plasticity (frozen weights)
#   4 ee_off    oriented, E->E recurrency OFF
#   5 ie_off    oriented, I->E inhibition OFF (--peak-ie 0)
#   6 vogels    oriented, inhibitory plasticity ON (Vogels iSTDP on I->E)
# The mechanism ablations (2-6) run on the ORIENTED net only; coherence is meaningless
# for random weights, and the RF-vs-random contrast is carried by 0 vs 1.
#
# METRICS (logged by interp_harness for every run): orientation coherence + the 2D
# Gaussian RF variance/covariance trajectory (how the prior geometry drifts), the L1-LR
# linear-probe accuracy (fit on 5k train features, --probe-fit-all), grouped/per-neuron
# eta2 (clustering), corrected dead fraction, and participation ratio.
#
# Grid: 7 conditions x 5 datasets x 5 seeds = 175 runs, 3 epochs each.
#   datasets   : mnist fmnist kmnist notmnist svhn   (SVHN last -- slowest, see below)
#   conditions : see above (0..6)
#   seeds      : 0..4
#
# Task ID encoding (SLURM_ARRAY_TASK_ID in 0..174):
#   seed     = task_id % N_SEEDS            # 0..4
#   cell     = task_id // N_SEEDS           # 0..34
#   cond_idx = cell % N_COND               # 0..6
#   ds_idx   = cell // N_COND              # 0..4
#
# !! PRE-SUBMISSION (once, on a NETWORKED login node, from PROJECT_ROOT) !!
#   1. Pre-cache datasets so concurrent array tasks don't race the download:
#        for d in MNIST KMNIST FashionMNIST SVHN; do
#          singularity exec noise_env.sif conda run -n noise_env python -c \
#            "from torchvision import datasets as D; import inspect;"
#        done
#      (SVHN uses split='train'/'test'; notMNIST comes from deeplake -- see get_data.py,
#       set NOTMNIST_LOCAL to a pre-fetched copy for offline nodes.)
#   2. Confirm noise_env.sif exists (reuse from the other sweeps).

#SBATCH --job-name=rf_phase1
#SBATCH --array=0-174
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
#SBATCH --output=experiments/RF_article/interp/phase1_ablation/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/interp/phase1_ablation/results/slurm_logs/%A_%a.out

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn3
cd "${PROJECT_ROOT}"

# ---- grid -----------------------------------------------------------------
# SVHN LAST: its dense natural images fire far more spikes than the sparse MNIST family
# (rate-coded cost scales with total spikes -- ~3-4x slower/epoch) and it carries the
# largest test set (full 26032). Placing it last gives it the highest task IDs, so the
# scheduler drains the fast datasets first and SVHN finishes last.
DATASETS=(mnist fmnist kmnist notmnist svhn)
N_SEEDS=5
N_COND=7
N_DS=${#DATASETS[@]}

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

# ---- decode condition: prior / rule / E->E / I->E / vogels ------------------
# EE and VOGELS are flag strings ("" = omit); PEAK_IE is a number (0 = I->E off).
case ${COND_IDX} in
    0) COND=base_ori; PRIOR=oriented; RULE=trace;   EE="--ee"; PEAK_IE=-2; VOGELS="" ;;
    1) COND=base_rnd; PRIOR=random;   RULE=trace;   EE="--ee"; PEAK_IE=-2; VOGELS="" ;;
    2) COND=triplet;  PRIOR=oriented; RULE=triplet; EE="--ee"; PEAK_IE=-2; VOGELS="" ;;
    3) COND=frozen;   PRIOR=oriented; RULE=frozen;  EE="--ee"; PEAK_IE=-2; VOGELS="" ;;
    4) COND=ee_off;   PRIOR=oriented; RULE=trace;   EE="";     PEAK_IE=-2; VOGELS="" ;;
    5) COND=ie_off;   PRIOR=oriented; RULE=trace;   EE="--ee"; PEAK_IE=0;  VOGELS="" ;;
    6) COND=vogels;   PRIOR=oriented; RULE=trace;   EE="--ee"; PEAK_IE=-2; VOGELS="--use-vogels" ;;
    *) echo "FATAL: COND_IDX=${COND_IDX} out of range" >&2; exit 1 ;;
esac

TAG="${DATASET}_${COND}_s${SEED}"

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
RUN_DIR="${PROJECT_ROOT}/experiments/RF_article/interp/phase1_ablation/results/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "dataset=${DATASET}  cond=${COND}  seed=${SEED}  epochs=3"
echo "  prior=${PRIOR} rule=${RULE} ee='${EE}' peak_ie=${PEAK_IE} vogels='${VOGELS}'"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

# ---- skip only GENUINELY-complete runs (results.json with a finite test_acc) -----
# A run killed mid-training leaves a PARTIAL results.json (config+trajectory, no
# test_acc); a mere -f check would wrongly skip it forever. Re-running overwrites.
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
# train/val are drawn from the dedicated TRAIN split, test is the dedicated TEST split
# (full) so results are comparable to published numbers. notMNIST has no canonical
# split (18724 merged -> seed-fixed 15k/1k/2724). Same setup as phase 2.
TRAIN_ALL=59000; VAL_ALL=1000; TEST_ALL=10000
case "${DATASET}" in
    svhn)     TRAIN_ALL=59000; VAL_ALL=1000; TEST_ALL=26032 ;;   # full dedicated test
    notmnist) TRAIN_ALL=15000; VAL_ALL=1000; TEST_ALL=2724  ;;   # merged, seed-fixed
esac
echo "Split    : train=${TRAIN_ALL} val=${VAL_ALL} test=${TEST_ALL}  probe-fit=5000  (dataset=${DATASET})"

# ---- run: recurrent NON-grouped net (no --grouped/--tiled), unsupervised rule ----
singularity exec "${SIF}" conda run --no-capture-output -n noise_env python -u \
    experiments/RF_article/interp/interp_harness.py \
    --tag           "${TAG}" \
    --dataset       "${DATASET}" \
    --seed          "${SEED}" \
    --prior         "${PRIOR}" \
    --rule          "${RULE}" \
    ${EE} \
    --peak-ie       "${PEAK_IE}" \
    ${VOGELS} \
    --epochs        3 \
    --train-all     "${TRAIN_ALL}" \
    --val-all       "${VAL_ALL}" \
    --test-all      "${TEST_ALL}" \
    --probe-fit-all 5000 \
    --no-plots \
    --output-dir    "${OUTPUT_DIR}"

echo "Finished : $(date)"
