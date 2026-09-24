#!/bin/bash
# Run a grid locally across several processes, as an alternative to SLURM.
#
#   ./run_local.sh sweep            [CONCURRENCY]   # 220-cell sleep-ratio sweep
#   ./run_local.sh experiment       [CONCURRENCY]   # 100-cell baseline grid
#
# Concurrency defaults to a RAM-derived estimate rather than the core count:
# each cell peaks around 1.5 GB at num_steps=100, so memory binds well before
# the 24 cores do. Override by passing a number.
#
# Resumable: cells that already have a result file are skipped, so an
# interrupted run continues where it stopped. Progress goes to run_local.log.
#
# Before a long unattended run, close the memory-hungry desktop apps —
# an earlier session sat at load 25 on 24 cores with ollama alone holding 6 GB.

set -uo pipefail

DRIVER="${1:-}"
case "${DRIVER}" in
    sweep)      SCRIPT="src/sweep.py";      OUT="results/sweep" ;;
    experiment) SCRIPT="src/experiment.py"; OUT="results/baselines" ;;
    ablation)   SCRIPT="src/ablation.py";   OUT="results/ablation" ;;
    *) echo "usage: $0 {sweep|experiment|ablation} [concurrency]" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

# --- concurrency ------------------------------------------------------------
CORES=$(nproc)
AVAIL_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
PER_CELL_MB=1600                      # measured peak RSS ~1.5 GB
BY_MEM=$(( AVAIL_MB / PER_CELL_MB ))
BY_MEM=$(( BY_MEM > 1 ? BY_MEM - 1 : 1 ))   # leave one cell's headroom
DEFAULT=$(( BY_MEM < CORES ? BY_MEM : CORES ))
CONC="${2:-${DEFAULT}}"

# --- environment ------------------------------------------------------------
# One BLAS thread per cell: the matrices are too small to thread profitably, and
# many single-threaded cells beat few multi-threaded ones for total throughput.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg

if [ -z "${CONDA_PREFIX:-}" ] || [[ "${CONDA_PREFIX}" != *noise_env* ]]; then
    echo "WARNING: noise_env does not look active. librosa is missing from base," >&2
    echo "         which makes the imports fail. Activate it first:" >&2
    echo "  source ~/anaconda3/etc/profile.d/conda.sh && conda activate noise_env" >&2
    echo >&2
fi

N=$(python "${SCRIPT}" --list | head -1)
mkdir -p "${OUT}"
DONE=$(ls "${OUT}"/*.json 2>/dev/null | grep -cv summary || true)
TOTAL=$(python "${SCRIPT}" --list | grep -cE "^\s+[0-9]+\s")

echo "================================================================"
echo "driver       ${SCRIPT}"
echo "grid         ${N}"
echo "already done ${DONE} / ${TOTAL}"
echo "concurrency  ${CONC}   (cores ${CORES}, ${AVAIL_MB} MB avail"
echo "                        -> ${BY_MEM} by memory at ~${PER_CELL_MB} MB/cell)"
echo "started      $(date)"
echo "================================================================"

# Only dispatch cells that have no result file yet, so this is resumable.
PENDING=$(python - "${SCRIPT}" "${OUT}" <<'PY'
import importlib.util, os, sys
# The driver's own directory must be importable: experiment.py and ablation.py
# import shared configuration from sweep.py, and loading a module by file path
# does not put its directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
spec = importlib.util.spec_from_file_location("drv", sys.argv[1])
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
print(" ".join(str(c["cell_id"]) for c in drv.build_grid()
                if not os.path.exists(drv.cell_path(c))))
PY
)

if [ -z "${PENDING// }" ]; then
    echo "Nothing pending — all cells already have results."
    exit 0
fi

echo "${PENDING}" | tr ' ' '\n' | grep -v '^$' \
  | xargs -P "${CONC}" -I{} python "${SCRIPT}" --task-id {}

STATUS=$?
echo "================================================================"
echo "finished     $(date)  (exit ${STATUS})"
python "${SCRIPT}" --collect
exit ${STATUS}
