#!/usr/bin/env bash
#
# CIFAR-10 repair run for the mnist_family sweep.
#
# WHY THIS EXISTS
#   run_local.sh asks every dataset for train=59000 / val=1000 / test=10000. The streamer
#   MERGES torchvision's train and test splits into one pool and carves the three partitions
#   from it, so the real budget is (len_train + len_test):
#
#       mnist / fmnist / kmnist   60000 + 10000 = 70000   -> 70000 requested, fits exactly
#       svhn                      73257 + 26032 = 99289   -> fits
#       cifar10                   50000 + 10000 = 60000   -> 70000 requested, DOES NOT FIT
#
#   The old code clamped each count to whatever was left instead of erroring, so CIFAR-10
#   silently got test=0. Both cifar10 seed-0 cells trained for ~23h each and reported
#   test_acc=nan in the final log line, with no test features and no saved weights -- so
#   they cannot be scored offline and must be rerun.
#
#   get_data.py now RAISES on an over-subscribed split, so this can no longer happen
#   silently. This script supplies CIFAR-10 a split that actually fits.
#
# SPLIT CHOICE
#   train=49000 val=1000 test=10000 (= 60000, the full pool). val/test match every other
#   dataset, so the metrics stay directly comparable; only the training volume is smaller,
#   because CIFAR-10 simply has fewer images. Same rationale already applied to notmnist in
#   run_local.sh: the primary RF-vs-random comparison is WITHIN a dataset and both priors
#   see identical volume, so it stays apples-to-apples. Only cross-dataset ABSOLUTE numbers
#   carry the caveat that cifar10 trained on 49k rather than 59k.
#
# CONCURRENCY
#   Defaults to MAX_PAR=2 so this can run ALONGSIDE the main run_local.sh driver (which
#   holds 6 slots on a 24-core box) without starving it.
#
# USAGE
#   experiments/RF_article/interp/mnist_family_sweep/run_cifar10_fix.sh
#   RUN_ID=run_local_20260728_121501 N_SEEDS=3 MAX_PAR=2 run_cifar10_fix.sh
#   SMOKE=1 run_cifar10_fix.sh        # 2-minute validity check, tiny counts, throwaway dir
#
# NOTE for run_local.sh itself: apply these two edits once its driver has exited (editing a
# script bash is currently executing can corrupt the running process):
#   1. in run_one(),  add:  if [ "$ds" = "cifar10" ]; then TR=49000 VA=1000 TE=10000; fi
#   2. in is_complete(), require test_acc to be FINITE, not merely present -- otherwise the
#      two nan cifar10 dirs are treated as complete and skipped forever.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT" || { echo "FATAL: bad PROJECT_ROOT $PROJECT_ROOT" >&2; exit 1; }

CONDA_SH="${CONDA_SH:-/home/andreas/anaconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate noise_env || { echo "FATAL: cannot activate noise_env" >&2; exit 1; }

PRIORS_STR="${PRIORS:-oriented random}"
read -r -a PRIORS <<< "$PRIORS_STR"
N_SEEDS=${N_SEEDS:-3}
MAX_PAR=${MAX_PAR:-2}
EPOCHS=${EPOCHS:-5}

# match run_local.sh's pinning so concurrent procs don't each grab 24 BLAS/numba threads
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
       NUMEXPR_NUM_THREADS=3 NUMBA_NUM_THREADS=3

TR=49000 VA=1000 TE=10000
if [ "${SMOKE:-0}" = "1" ]; then
    # Proves the split is valid and test_acc comes back finite, in ~2 min instead of ~23h.
    RUN_ID=${RUN_ID:-smoke_cifar10_$(date +%Y%m%d_%H%M%S)}
    TR=600 VA=200 TE=200; EPOCHS=1; N_SEEDS=1; PRIORS=(oriented); MAX_PAR=1
fi

RUN_ID=${RUN_ID:-run_local_20260728_121501}
BASE="experiments/RF_article/interp/mnist_family_sweep/results/${RUN_ID}"
mkdir -p "$BASE/logs"

echo "========================================================"
echo "CIFAR-10 repair run   (smoke=${SMOKE:-0})"
echo "RUN_ID   : $RUN_ID"
echo "priors   : ${PRIORS[*]}   seeds: 0..$((N_SEEDS-1))"
echo "split    : train=$TR val=$VA test=$TE   epochs: $EPOCHS   max_par: $MAX_PAR"
echo "results  : $BASE"
echo "started  : $(date)"
echo "========================================================"

# Stricter than run_local.sh's version: test_acc must be present AND FINITE. The broken
# cifar10 cells wrote test_acc=nan, which the presence-only check counted as complete.
is_complete() {
    python - "$1" <<'PY' 2>/dev/null
import json, math, sys
try:
    with open(sys.argv[1]) as f: d = json.load(f)
    v = d.get("test_acc") if isinstance(d, dict) else None
    sys.exit(0 if isinstance(v, (int, float)) and math.isfinite(v) else 1)
except Exception:
    sys.exit(1)
PY
}

run_one() {
    local prior="$1" seed="$2"
    local tag="cifar10_${prior}_s${seed}"
    local out="$BASE/$tag"
    mkdir -p "$out"
    if is_complete "$out/results.json"; then echo "skip  $tag (already complete)"; return 0; fi

    echo "START $tag  $(date +%H:%M:%S)"
    python -u experiments/RF_article/interp/interp_harness.py \
        --tag "$tag" --dataset cifar10 --seed "$seed" \
        --prior "$prior" --rule reward \
        --grouped --group-layout block --tiled \
        --dense-readout --readout-lr 0.1 --reward-lr 5e-6 --peak-ei 50 \
        --rf-length 3.0 --rf-thickness 1.2 --center-margin 4.0 \
        --epochs "$EPOCHS" --train-all "$TR" --val-all "$VA" --test-all "$TE" \
        --no-plots --output-dir "$out" > "$BASE/logs/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && is_complete "$out/results.json"; then
        echo "DONE  $tag  $(date +%H:%M:%S)"
    else
        echo "FAIL  $tag  rc=$rc  (see $BASE/logs/${tag}.log)"
    fi
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 15; done; }

for pass in 1 2; do
    echo "---- PASS $pass ----"
    for ((seed=0; seed<N_SEEDS; seed++)); do
        for prior in "${PRIORS[@]}"; do
            throttle
            run_one "$prior" "$seed" &
        done
    done
    wait
done

echo "========================================================"
done_n=0
for ((seed=0; seed<N_SEEDS; seed++)); do
    for prior in "${PRIORS[@]}"; do
        is_complete "$BASE/cifar10_${prior}_s${seed}/results.json" && done_n=$((done_n+1))
    done
done
echo "COMPLETE: ${done_n}/$(( ${#PRIORS[@]} * N_SEEDS )) cifar10 runs with finite test_acc"
echo "finished: $(date)"
echo "========================================================"
