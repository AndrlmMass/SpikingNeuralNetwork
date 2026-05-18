#!/bin/bash
# Submit all Phase 1 jobs (regularizer comparison).
#
# 5 conditions × 5 seeds = 25 jobs.
# Each job writes to experiments/results/phase1/<condition>_s<seed>/
#
# Usage (run from repo root on the login node):
#   bash experiments/submit_phase1.sh
#
# To limit concurrent jobs (e.g. max 10 at a time), add %10:
#   sbatch --array=0-24%10 experiments/slurm_phase1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/results/phase1/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch \
    --array=0-24 \
    --output="${LOG_DIR}/slurm_%A_%a.out" \
    --error="${LOG_DIR}/slurm_%A_%a.err" \
    "${SCRIPT_DIR}/slurm_phase1.sh" \
    | awk '{print $NF}')

echo "Phase 1 submitted — job array ID: ${JOB_ID}"
echo "25 jobs (tasks 0-24), logs -> ${LOG_DIR}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${SCRIPT_DIR}/results/phase1/<condition>_s<seed>/job.log"
