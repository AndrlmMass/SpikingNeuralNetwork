#!/bin/bash
# Local (no-SLURM) driver for PHASE 1 -- the unsupervised ablation study on the
# recurrent, NON-grouped network. Local twin of run_slurm.sh (HPC being down),
# mirroring its per-run config and its results/<RUN_ID>/<tag>/results.json layout so
# the downstream analysis is identical. Swaps `singularity exec noise_env.sif` for the
# local `noise_env` conda env.
#
# CONDITIONS (7), one departure from a trace-STDP baseline each:
#   base_ori  oriented + trace-STDP + E->E on + I->E on + vogels off   (baseline)
#   base_rnd  RANDOM weights + trace-STDP                              (RF-vs-random)
#   triplet   oriented, triplet-STDP
#   frozen    oriented, no plasticity
#   ee_off    oriented, E->E off
#   ie_off    oriented, I->E off (--peak-ie 0)
#   vogels    oriented, inhibitory plasticity on (Vogels iSTDP)
#
# Grid: 5 datasets x 7 conditions x N_SEEDS seeds, 3 epochs each.
#   datasets : mnist fmnist kmnist svhn notmnist
#   seeds    : 0 .. N_SEEDS-1
#
# ORDERING is SEED-OUTER: the entire single-seed grid completes first, then seed 1, etc.,
# so you get the full ablation for every dataset early and extra seeds fill in as time
# allows. Fully resumable: a COMPLETE results.json (finite test_acc) is skipped.
#
# CONCURRENCY is RAM-bounded. Each recurrent run holds one dataset in RAM; set MAX_PAR
# to taste (SVHN's full 26k test is the memory high-water point).
#
# LEAVE-IT-RUNNING (inhibits sleep so a closed lid / idle does NOT pause the job):
#   systemd-inhibit --what=sleep:idle:handle-lid-switch --why="SNN phase-1 ablation" \
#     nohup experiments/RF_article/interp/phase1_ablation/run_local.sh \
#     > experiments/RF_article/interp/phase1_ablation/driver.out 2>&1 &
#
# Env overrides: N_SEEDS(5) MAX_PAR(4) EPOCHS(3) RUN_ID DATASETS CONDS

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT" || { echo "FATAL: bad PROJECT_ROOT $PROJECT_ROOT" >&2; exit 1; }

# ---- local env (replaces singularity noise_env.sif) ------------------------
CONDA_SH="${CONDA_SH:-/home/andreas/anaconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate noise_env || { echo "FATAL: cannot activate noise_env" >&2; exit 1; }

# ---- grid ------------------------------------------------------------------
# SVHN LAST (dense natural images -> ~3-4x more spikes/epoch + largest test set); with
# seed-outer ordering it runs last within each seed pass, so the fast datasets land first.
read -r -a DATASETS <<< "${DATASETS:-mnist fmnist kmnist notmnist svhn}"
read -r -a CONDS    <<< "${CONDS:-base_ori base_rnd triplet frozen ee_off ie_off vogels}"
N_SEEDS=${N_SEEDS:-5}
MAX_PAR=${MAX_PAR:-4}
EPOCHS=${EPOCHS:-3}

# per-run thread pinning so MAX_PAR procs don't each spawn all cores
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
       NUMEXPR_NUM_THREADS=3 NUMBA_NUM_THREADS=3

RUN_ID=${RUN_ID:-run_local_$(date +%Y%m%d_%H%M%S)}
BASE="experiments/RF_article/interp/phase1_ablation/results/${RUN_ID}"
mkdir -p "$BASE/logs"

echo "========================================================"
echo "PHASE 1  RUN_ID : $RUN_ID"
echo "datasets : ${DATASETS[*]}"
echo "conds    : ${CONDS[*]}"
echo "seeds    : 0..$((N_SEEDS-1))   epochs: $EPOCHS   max_par: $MAX_PAR"
echo "results  : $BASE"
echo "started  : $(date)"
echo "========================================================"

# A run is COMPLETE only if results.json parses and has a FINITE top-level test_acc.
is_complete() {
    python - "$1" <<'PY' 2>/dev/null
import json,sys,math
try:
    d=json.load(open(sys.argv[1]))
    a=d.get("test_acc")
    sys.exit(0 if isinstance(d,dict) and isinstance(a,(int,float)) and math.isfinite(a) else 1)
except Exception:
    sys.exit(1)
PY
}

# map a condition name -> the flags that depart from the trace-STDP oriented baseline
cond_flags() {   # echoes: PRIOR RULE EE PEAK_IE VOGELS   (EE/VOGELS may be empty)
    case "$1" in
        base_ori) echo "oriented trace   --ee -2 " ;;
        base_rnd) echo "random   trace   --ee -2 " ;;
        triplet)  echo "oriented triplet --ee -2 " ;;
        frozen)   echo "oriented frozen  --ee -2 " ;;
        ee_off)   echo "oriented trace   ''   -2 " ;;
        ie_off)   echo "oriented trace   --ee 0  " ;;
        vogels)   echo "oriented trace   --ee -2 --use-vogels" ;;
        *) echo "FATAL unknown cond $1" >&2; return 1 ;;
    esac
}

run_one() {
    local ds="$1" cond="$2" seed="$3"
    local tag="${ds}_${cond}_s${seed}"
    local out="$BASE/$tag"
    mkdir -p "$out"
    if is_complete "$out/results.json"; then echo "skip  $tag (already complete)"; return 0; fi

    # per-dataset dedicated splits (same setup as phase 2)
    local TR=59000 VA=1000 TE=10000
    case "$ds" in
        svhn)     TR=59000 VA=1000 TE=26032 ;;
        notmnist) TR=15000 VA=1000 TE=2724  ;;
    esac

    # decode condition flags. EE/VOGELS are flag words ('' = omit); PEAK_IE is a number.
    read -r PRIOR RULE EE PEAK_IE VOGELS <<< "$(cond_flags "$cond")"
    [ "$EE" = "''" ] && EE=""
    [ "${VOGELS:-}" = "''" ] && VOGELS=""

    echo "START $tag  $(date +%H:%M:%S)  [prior=$PRIOR rule=$RULE ee='$EE' peak_ie=$PEAK_IE vogels='${VOGELS:-}']"
    python -u experiments/RF_article/interp/interp_harness.py \
        --tag "$tag" --dataset "$ds" --seed "$seed" \
        --prior "$PRIOR" --rule "$RULE" $EE --peak-ie "$PEAK_IE" ${VOGELS:-} \
        --epochs "$EPOCHS" \
        --train-all "$TR" --val-all "$VA" --test-all "$TE" \
        --probe-fit-all 5000 \
        --no-plots --output-dir "$out" > "$BASE/logs/${tag}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && is_complete "$out/results.json"; then
        echo "DONE  $tag  $(date +%H:%M:%S)"
    else
        echo "FAIL  $tag  rc=$rc  (see $BASE/logs/${tag}.log)"
    fi
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 15; done; }

# Two passes: pass 1 runs the whole grid; pass 2 re-attempts any gaps left by a crash.
for pass in 1 2; do
    echo "---- PASS $pass ----"
    for ((seed=0; seed<N_SEEDS; seed++)); do
        for ds in "${DATASETS[@]}"; do
            for cond in "${CONDS[@]}"; do
                throttle
                run_one "$ds" "$cond" "$seed" &
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
total=$(( ${#DATASETS[@]} * ${#CONDS[@]} * N_SEEDS ))
echo "COMPLETE: ${done_n}/${total} runs finished (results.json with finite test_acc)"
echo "finished: $(date)"
echo "========================================================"
