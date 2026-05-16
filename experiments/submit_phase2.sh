#!/bin/bash
# Submit all Phase 2 jobs (sleep duration × noise grid).
#
# 8 durations × 8 noise levels × 5 seeds = 320 jobs.
# Each job writes to experiments/results/phase2/sd<dur>_vn<noise>_s<seed>/
#
# Usage (run from repo root on the login node):
#   bash experiments/submit_phase2.sh
#
# By default uses sleep/static as the regularizer.  Override after Phase 1:
#   BEST_REG_TYPE=normalize BEST_REG_MODE=adaptive bash experiments/submit_phase2.sh
#
# The --array%20 cap limits concurrent jobs to 20 at a time to avoid
# overwhelming the queue. Remove or increase it if the cluster allows more.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/results/phase2/logs"
mkdir -p "${LOG_DIR}"

# Pass best reg type/mode through to the job script
export BEST_REG_TYPE=${BEST_REG_TYPE:-sleep}
export BEST_REG_MODE=${BEST_REG_MODE:-static}

JOB_ID=$(sbatch \
    --array=0-319%20 \
    --output="${LOG_DIR}/slurm_%A_%a.out" \
    --error="${LOG_DIR}/slurm_%A_%a.err" \
    --export=ALL \
    "${SCRIPT_DIR}/slurm_phase2.sh" \
    | awk '{print $NF}')

echo "Phase 2 submitted — job array ID: ${JOB_ID}"
echo "320 jobs (tasks 0-319, max 20 concurrent), logs -> ${LOG_DIR}"
echo "Regularizer: ${BEST_REG_TYPE}/${BEST_REG_MODE}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${SCRIPT_DIR}/results/phase2/sd<dur>_vn<noise>_s<seed>/job.log"
