#!/bin/bash
#SBATCH --job-name=rf_ood
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=orion
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andreas.lie.massey@nmbu.no
#SBATCH --output=experiments/RF_article/interp/mnist_family_sweep/results/slurm_logs/%A_%a.out
#SBATCH --error=experiments/RF_article/interp/mnist_family_sweep/results/slurm_logs/%A_%a.out
#
# OOD REJECTION, 5 seeds -- replaces the single-seed results/ood_mnist/run1. An MNIST-trained
# RF + R-Hebb network (the phase-2 `oriented` cell, identical flags) is held fixed and probed
# with KMNIST, notMNIST and FMNIST. SVHN is dropped, as in the article.
#
# Differs from run_slurm.sh's mnist_oriented cell in three places, all inherited from run1:
#   --train-all 55000 --val-all 5000  the calibration split (X_cal) is 5000 items, so the
#                                     Mahalanobis model can be fitted on one half and its
#                                     threshold placed on the other (plot_ood_auroc.py)
#   --ood-dataset x3                  dumps X_ood_<name> into uncertainty_features.npz
#   --save-weights                    W_dense is needed to replay the readout on OOD features
#
# After the harness, each task runs the two post-hoc scorers on its own run dir:
#   ood.py            -> ood.json            predictive entropy etc. of the learned readout
#   input_density.py  -> input_density.json  pixel baselines (PCA residual, mean intensity)
#                                            with the pixel fit subsample seeded by SEED
#
# Task ID = seed (0..4). One results folder per submission:
#   results/ood_mnist/<RUN_ID>/mnist_ood_s<seed>/
# Then:  python experiments/RF_article/interp/plot_ood_auroc.py --run results/ood_mnist/<RUN_ID>
#
# Pre-cache KMNIST, notMNIST and FMNIST on a networked login node first (see run_slurm.sh).

set -uo pipefail

PROJECT_ROOT=/mnt/users/andreama/projects/biosnn4
cd "${PROJECT_ROOT}"

OOD_SETS=(kmnist notmnist fmnist)
SEED=${SLURM_ARRAY_TASK_ID}
TAG="mnist_ood_s${SEED}"

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
RUN_DIR="${PROJECT_ROOT}/results/ood_mnist/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/${TAG}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Run  : ${RUN_ID}"
echo "Job  : ${SLURM_JOB_ID}  Task : ${SLURM_ARRAY_TASK_ID}"
echo "Node : $(hostname)  Started : $(date)"
echo "mnist  prior=oriented  seed=${SEED}  ood=${OOD_SETS[*]}"
echo "output -> ${OUTPUT_DIR}"
echo "========================================"

SIF="${PROJECT_ROOT}/noise_env.sif"
if [ ! -f "${SIF}" ]; then
    echo "FATAL: container image not found at ${SIF}" >&2
    exit 1
fi
PY=(singularity exec "${SIF}" conda run --no-capture-output -n noise_env python -u)

# ---- training is skipped only if results.json is complete AND the OOD features exist ----
is_complete() {
    conda run --no-capture-output -n noise_env python - "$1" <<'PY' 2>/dev/null
import json,sys,math,os
import numpy as np
try:
    d=json.load(open(os.path.join(sys.argv[1],"results.json")))
    a=d.get("test_acc")
    ok=isinstance(a,(int,float)) and math.isfinite(a)
    z=np.load(os.path.join(sys.argv[1],"uncertainty_features.npz"))
    ok=ok and all(f"X_ood_{o}" in z.files for o in ("kmnist","notmnist","fmnist"))
    ok=ok and os.path.exists(os.path.join(sys.argv[1],"weights","checkpoint.npz"))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
}

OOD_ARGS=()
for o in "${OOD_SETS[@]}"; do OOD_ARGS+=(--ood-dataset "${o}"); done

if singularity exec "${SIF}" bash -c "$(declare -f is_complete); is_complete '${OUTPUT_DIR}'"; then
    echo "trained run already complete — skipping to the scorers."
else
    "${PY[@]}" experiments/RF_article/interp/interp_harness.py \
        --tag           "${TAG}" \
        --dataset       mnist \
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
        --epochs        3 \
        --train-all     55000 \
        --val-all       5000 \
        --test-all      10000 \
        --probe-fit-all 5000 \
        "${OOD_ARGS[@]}" \
        --ood-all       10000 \
        --save-weights \
        --output-dir    "${OUTPUT_DIR}" || { echo "FATAL: harness failed" >&2; exit 1; }
fi

# ---- post-hoc scorers (cheap; rerun whenever their output is missing) -----------------
if [ ! -s "${OUTPUT_DIR}/ood.json" ]; then
    "${PY[@]}" experiments/RF_article/interp/ood.py --run "${OUTPUT_DIR}" --seed "${SEED}" \
        || echo "WARN: ood.py failed" >&2
fi
if [ ! -s "${OUTPUT_DIR}/input_density.json" ]; then
    DENS_ARGS=()
    for o in "${OOD_SETS[@]}"; do DENS_ARGS+=(--ood "${o}"); done
    "${PY[@]}" experiments/RF_article/interp/input_density.py --dataset mnist \
        "${DENS_ARGS[@]}" --run "${OUTPUT_DIR}" --seed "${SEED}" \
        || echo "WARN: input_density.py failed" >&2
fi

echo "Finished : $(date)"
