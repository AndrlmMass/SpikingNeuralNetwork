#!/bin/bash
# Local (no-SLURM) driver for the extended dataset sweep:
#   RECEPTIVE FIELDS (oriented) vs RANDOM feedforward weights -- the article's core
#   comparison -- across the 28x28 grayscale family, at the canonical 95% config
#   (oriented/random W_se init + R-STDP on the excitatory layer + dense softmax-delta
#   readout). Every dataset is grayscaled + resized to 28x28 by the streamer, so N_x=784
#   and the architecture is byte-identical across datasets and across the two priors --
#   only the task and the W_se init change.
#
# Runs the grid as local background processes (the HPC being down), mirroring
# run_slurm.sh's per-run config and its results/<RUN_ID>/<tag>/results.json layout so the
# downstream analysis is identical. Swaps `singularity exec noise_env.sif` for the local
# `noise_env` conda env.
#
# Grid: 6 datasets x {oriented, random} x N_SEEDS seeds, 5 epochs each.
#   datasets : mnist fmnist kmnist cifar10 svhn notmnist   (cifar10/svhn collapsed to
#              28x28 grayscale; notmnist via deeplake, at REDUCED counts -- see below)
#   priors   : oriented random
#   seeds    : 0 .. N_SEEDS-1
#
# ORDERING is SEED-OUTER: the entire single-seed grid (all datasets x both priors, seed 0)
# completes FIRST, then seed 1, then seed 2. So even if the machine doesn't get through all
# runs, you are guaranteed the full RF-vs-random comparison for every dataset early, and the
# extra seeds fill in as time allows. Fully resumable: a present results.json is skipped.
#
# CONCURRENCY is RAM-bounded (each run ~2.5-3 GB; ~22 GB usable -> 6 in flight).
#
# LEAVE-IT-RUNNING usage (inhibits sleep so a closed lid / idle does NOT pause the job):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch --why="SNN RF-vs-random sweep" \
#     nohup experiments/RF_article/interp/mnist_family_sweep/run_local.sh \
#     > experiments/RF_article/interp/mnist_family_sweep/driver.out 2>&1 &
#
# Env overrides: N_SEEDS(3) MAX_PAR(6) EPOCHS(5) RUN_ID DATASETS PRIORS
#   e.g. one-seed first pass:  N_SEEDS=1 run_local.sh

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT" || { echo "FATAL: bad PROJECT_ROOT $PROJECT_ROOT" >&2; exit 1; }

# ---- local env (replaces singularity noise_env.sif) ------------------------
CONDA_SH="${CONDA_SH:-/home/andreas/anaconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate noise_env || { echo "FATAL: cannot activate noise_env" >&2; exit 1; }

# ---- grid ------------------------------------------------------------------
read -r -a DATASETS <<< "${DATASETS:-mnist fmnist kmnist cifar10 svhn notmnist}"
read -r -a PRIORS   <<< "${PRIORS:-oriented random}"
N_SEEDS=${N_SEEDS:-3}
MAX_PAR=${MAX_PAR:-6}
EPOCHS=${EPOCHS:-5}

# per-run thread pinning so MAX_PAR procs don't each spawn 24 BLAS/numba threads
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
       NUMEXPR_NUM_THREADS=3 NUMBA_NUM_THREADS=3

RUN_ID=${RUN_ID:-run_local_$(date +%Y%m%d_%H%M%S)}
BASE="experiments/RF_article/interp/mnist_family_sweep/results/${RUN_ID}"
mkdir -p "$BASE/logs"

echo "========================================================"
echo "RUN_ID   : $RUN_ID"
echo "datasets : ${DATASETS[*]}"
echo "priors   : ${PRIORS[*]}"
echo "seeds    : 0..$((N_SEEDS-1))   epochs: $EPOCHS   max_par: $MAX_PAR"
echo "results  : $BASE"
echo "started  : $(date)"
echo "========================================================"

# A run is COMPLETE only if results.json parses and has a top-level "test_acc" key.
# The harness rewrites results.json at EVERY checkpoint with a partial {config,trajectory}
# (no test_acc) and only stamps test_acc in the final end-of-run write. So a run killed
# mid-training leaves a partial file -- checking mere existence would wrongly skip it, and
# it would never finish. This check re-runs partials (overwriting them) and skips only
# genuinely-completed runs.
is_complete() {
    python - "$1" <<'PY' 2>/dev/null
import json,sys
try:
    with open(sys.argv[1]) as f: d=json.load(f)
    sys.exit(0 if isinstance(d,dict) and "test_acc" in d else 1)
except Exception:
    sys.exit(1)
PY
}

run_one() {
    local ds="$1" prior="$2" seed="$3"
    local tag="${ds}_${prior}_s${seed}"
    local out="$BASE/$tag"
    mkdir -p "$out"
    if is_complete "$out/results.json"; then echo "skip  $tag (already complete)"; return 0; fi

    # notmnist-small (~18.7k total) cannot fill 59k/1k/10k; use a valid reduced split.
    # (The primary RF-vs-random comparison is WITHIN a dataset -- both priors see the same
    #  volume -- so this stays apples-to-apples; only cross-dataset absolute numbers carry
    #  the caveat that notmnist trained on less data.)
    local TR=59000 VA=1000 TE=10000
    if [ "$ds" = "notmnist" ]; then TR=14000 VA=1500 TE=3000; fi

    echo "START $tag  $(date +%H:%M:%S)"
    python -u experiments/RF_article/interp/interp_harness.py \
        --tag "$tag" --dataset "$ds" --seed "$seed" \
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

# Two passes: pass 1 runs the whole grid; pass 2 re-attempts any gaps left by a crash
# (skip-on-results.json makes completed runs no-ops). Transient failures self-heal.
for pass in 1 2; do
    echo "---- PASS $pass ----"
    for ((seed=0; seed<N_SEEDS; seed++)); do
        for ds in "${DATASETS[@]}"; do
            for prior in "${PRIORS[@]}"; do
                throttle
                run_one "$ds" "$prior" "$seed" &
            done
        done
    done
    wait
done

# ---- summary ---------------------------------------------------------------
echo "========================================================"
done_n=0
while IFS= read -r rj; do is_complete "$rj" && done_n=$((done_n+1)); done \
    < <(find "$BASE" -mindepth 2 -name results.json)
total=$(( ${#DATASETS[@]} * ${#PRIORS[@]} * N_SEEDS ))
echo "COMPLETE: ${done_n}/${total} runs finished (results.json with test_acc)"
echo "finished: $(date)"
echo "========================================================"
