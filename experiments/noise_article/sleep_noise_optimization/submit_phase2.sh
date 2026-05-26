#!/bin/bash
# Submit all Phase 2 jobs (sleep duration x noise x reg-mode grid).
#
# 6 durations x 6 noise levels x 3 reg-modes x 5 seeds = 540 jobs.
# Each job writes to experiments/noise_article/sleep_noise_optimization/results/phase2/
#
# Usage (run from repo root on the login node):
#   bash experiments/noise_article/sleep_noise_optimization/submit_phase2.sh
#
# The --array%20 cap limits concurrent jobs to 20 at a time.
# Remove or increase it if the cluster allows more.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/results/phase2/logs"
mkdir -p "${LOG_DIR}"

JOB_ID=$(sbatch \
    --output="${LOG_DIR}/slurm_%A_%a.out" \
    --error="${LOG_DIR}/slurm_%A_%a.err" \
    "${SCRIPT_DIR}/slurm_phase2.sh" \
    | awk '{print $NF}')

echo "Phase 2 submitted — job array ID: ${JOB_ID}"
echo "540 jobs (tasks 0-539, 6 durations x 6 noise levels x 3 reg-modes x 5 epochs)"
echo ""
echo "Conditions:"
echo "  mode_idx 0 = static"
echo "  mode_idx 1 = layer"
echo "  mode_idx 2 = neuron"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Elapsed"
echo "  tail -f ${LOG_DIR}/slurm_${JOB_ID}_<task>.out"
