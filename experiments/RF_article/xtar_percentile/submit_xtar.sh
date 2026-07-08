#!/bin/bash
# Submit the x_tar percentile sweep so the whole array lands in ONE
# date+time-stamped results folder.
#
# The timestamp is computed HERE, once, and exported to every array task via
# --export. That guarantees all 354 tasks agree on a single run folder — if
# each task computed its own `date`, they'd start at different moments and
# scatter into different folders.
#
#   bash experiments/RF_article/xtar_percentile/submit_xtar.sh
#
# Results land in:
#   results/<RUN_ID>/<config>/results.json   (the JSON you actually want)
#   results/<RUN_ID>/slurm_logs/<jobid>_<task>.out
#
# Grab the most recent run with:  ls -dt results/run_* | head -1
set -euo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn2
cd "${PROJECT_ROOT}"

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="experiments/RF_article/xtar_percentile/results/${RUN_ID}"
mkdir -p "${RUN_DIR}/slurm_logs"

sbatch \
    --export=ALL,RUN_ID="${RUN_ID}" \
    --output="${RUN_DIR}/slurm_logs/%A_%a.out" \
    --error="${RUN_DIR}/slurm_logs/%A_%a.out" \
    experiments/RF_article/xtar_percentile/slurm_xtar.sh

echo "Submitted run ${RUN_ID}"
echo "Results -> ${PROJECT_ROOT}/${RUN_DIR}"
