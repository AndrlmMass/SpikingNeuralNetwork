"""
Local driver for the frozen-SE supervised grid: 5 datasets x N seeds, run concurrently.

Replaces mnist_family_sweep/run_local.sh, which hardcodes bash, a Linux conda path and
systemd-inhibit. Same two properties that script got right:

  skip-on-complete   a cell counts as done only when results.json parses AND carries a
                     finite test_acc. The harness rewrites results.json at EVERY
                     checkpoint with a partial {config, trajectory} and only stamps
                     test_acc in the final write, so checking mere existence would skip
                     a run killed mid-training and it would never finish.
  bounded concurrency each cell holds ~2.5-3 GB against this box's 17 GB, and the gate is
                     re-checked BEFORE EVERY CELL rather than once at launch. A launch-time
                     cap is wrong within the hour on a desktop: the first attempt at this
                     grid capped itself to 1 concurrent because an unrelated smoke run was
                     still holding memory, which would have turned a ~10 h job into ~30 h.
                     A cell OOM-killed mid-grid is recoverable (skip-on-complete re-runs it);
                     stranded capacity is not recovered at all.

WHY THIS RUNS THE HARNESS AND NOT A CACHED-FEATURE REPLAY
---------------------------------------------------------
The obvious optimization for a frozen network is to run the spiking forward pass once per
image, cache the exc-rate features, and replay the readout's delta rule offline. It does
not work here, and the reason is worth recording: the input encoding is POISSON, so every
presentation of the same image draws a fresh spike train. Featurizing the same 300 test
images twice, in one process, with byte-identical frozen weights, gives features that
correlate 0.968 -- not 1.0. The online readout therefore sees a fresh noisy view of each
image every epoch (implicit augmentation) while a replay would see one frozen draw
repeated. Measured on a matched 3000-image/2-epoch pair: the replayed readout reached
test 0.807 against the online 0.830, with W_dense correlating only 0.54 between them.

So the cells below run the real harness at phase-2 flags with --reward-lr 0. That is exact
by construction and byte-comparable to the 50-run phase-2 grid. It is affordable because
the earlier cost estimate was wrong: this machine featurizes at ~0.026 s per image
presentation, so a cell is ~260k presentations ~= 2 h, not the 10-17 h a plastic phase-2
run took under 6-way contention on a different box.

    val-every 10 (not phase 2's 1): validation costs as much as training at --val-every 1
    with a 5000-image calibration set (165 checkpoints x 5000 = 825k presentations, more
    than the 165k of actual training). 17 checkpoints still resolve the trajectory, and
    the headline numbers do not depend on val cadence at all.

LAUNCHING: use a background mechanism that outlives the launching shell. `nohup ... &`
from a short-lived shell is NOT enough here -- the first attempt at this grid had all
three cells killed together after 1h45, mid-line and with no traceback, which is the
signature of process-group teardown rather than OOM. Each cell only stamps its test_acc
at the very end, so a kill at 1h50 loses the whole cell; skip-on-complete then re-runs it
from scratch.

Usage:
    python experiments/RF_article/frozen_supervised/run_all.py                # full grid
    python experiments/RF_article/frozen_supervised/run_all.py --seeds 1 --smoke
    python experiments/RF_article/frozen_supervised/run_all.py --datasets mnist --seeds 3
"""
import argparse, json, math, os, subprocess, sys, threading, time
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN_FROZEN = os.path.join(REPO, "experiments", "RF_article", "frozen_supervised",
                          "run_frozen.py")

# SVHN last: dense natural images fire far more spikes per epoch than the sparse MNIST
# family, and it carries the full 26032-item test set. notMNIST is cheapest.
DATASETS = ["notmnist", "mnist", "fmnist", "kmnist", "svhn"]

# (train, val, test). val is the CALIBRATION set for the abstention threshold and comes
# out of the dedicated TRAIN split, so train shrinks to keep each total exact -- see
# neurosnn/_data/_partition_indices, which raises before the first image if they do not fit.
SPLITS = {
    "mnist":    (55000, 5000, 10000),   # 60000 train split, exact fill
    "fmnist":   (55000, 5000, 10000),
    "kmnist":   (55000, 5000, 10000),
    "svhn":     (68257, 5000, 26032),   # 73257 train split, exact fill
    "notmnist": (11000, 5000,  2724),   # 18724 merged pool, exact fill
}
SMOKE = (2000, 500, 800)

GB = 1024 ** 3
# MEASURED, not assumed: peak RSS sampled every 6 s across a running MNIST/FMNIST/notMNIST
# trio was 1.74 / 1.74 / 1.65 GB. The dominant term is the per-batch spike buffer
# (batch 1000 x 350 steps x 2784 neurons x int8 ~= 0.97 GB), which is why cell size barely
# depends on the dataset: SVHN caches more images (~0.31 GB vs 0.22 GB) but uses the same
# 1000-image batches. 2.0 leaves margin over the 1.74 without stranding a slot.
MEM_PER_CELL_GB = 2.0

# Reservation state for the memory gate. Without it the gate is a TOCTOU race: several
# waiting cells all observe the same free-memory figure and all start at once, each
# believing it was the only one admitted. Observed exactly that on the first launch --
# three cells started in the same second on 4.5 GB of headroom meant for one.
_mem_lock = threading.Lock()
_reserved = [0.0]


def is_complete(path):
    """Done means results.json parses AND has a finite test_acc -- not merely exists."""
    try:
        with open(path) as f:
            d = json.load(f)
        a = d.get("test_acc")
        return isinstance(d, dict) and isinstance(a, (int, float)) and math.isfinite(a)
    except Exception:
        return False


def free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / GB
    except Exception:
        return float("inf")


def wait_for_memory(tag, need=MEM_PER_CELL_GB, headroom=1.5, poll=60, max_wait=7200):
    """Block until there is room for one more cell, then return.

    Checked PER CELL rather than once at launch. A grid that runs for hours on a desktop
    shares the machine with whatever else is open, so a cap computed at t=0 is wrong by
    t=2h in both directions: it strands capacity when memory frees up (the first attempt
    at this grid capped itself to 1 concurrent because an unrelated smoke run was still
    holding 3 GB, turning a ~10 h job into a ~30 h one), and it oversubscribes when
    something else starts later.

    Admission is RESERVED under a lock, so a cell that has been let through but has not
    yet allocated still counts against the next caller's budget. Cells ramp to peak over
    a couple of minutes, so without the reservation the free-memory reading lags reality
    by exactly the window in which the damage is done.

    After max_wait it starts anyway rather than deadlocking: a cell that gets OOM-killed
    is recoverable, because skip-on-complete will re-run it. A grid that never starts is not.
    """
    t0 = time.time()
    while True:
        with _mem_lock:
            avail = free_gb() - _reserved[0]
            if avail >= need + headroom:
                _reserved[0] += need
                return
        if time.time() - t0 > max_wait:
            print(f"[grid] {tag}: waited {max_wait / 60:.0f} min for memory "
                  f"({avail:.1f} GB free after reservations); starting anyway", flush=True)
            with _mem_lock:
                _reserved[0] += need
            return
        print(f"[grid] {tag}: waiting for memory ({avail:.1f} GB free after "
              f"reservations, need {need + headroom:.1f})", flush=True)
        time.sleep(poll)


def release_memory(need=MEM_PER_CELL_GB):
    """Give the reservation back once a cell has exited (and its RSS is really gone)."""
    with _mem_lock:
        _reserved[0] = max(0.0, _reserved[0] - need)


def cell_cmd(ds, seed, out, epochs, smoke, threads, draws):
    """One frozen cell, via run_frozen.py (the featurize route), not the harness.

    The harness re-presents the training set once per epoch and re-runs a full validation
    pass at every checkpoint: 260k image presentations for a 55k MNIST cell, against the
    70k a frozen measurement actually needs. Freezing W_se does not make the forward pass
    cheaper -- the network still has to be simulated for 350 timesteps per image, and
    measured, that only buys 2.2x (0.1355 s/img training vs 0.0617 s/img in test mode).
    The saving comes from not doing the work 3.7x over.
    """
    tr, va, te = SMOKE if smoke else SPLITS[ds]
    return [sys.executable, "-u", RUN_FROZEN,
            "--dataset", ds, "--seed", str(seed),
            "--epochs", str(epochs), "--draws", str(draws),
            "--train-all", str(tr), "--val-all", str(va), "--test-all", str(te),
            "--output-dir", out]


def run_cell(ds, seed, base, epochs, smoke, threads, draws, results):
    tag = f"{ds}_frozen_s{seed}"
    out = os.path.join(base, tag)
    os.makedirs(out, exist_ok=True)
    rj = os.path.join(out, "results.json")
    if is_complete(rj):
        print(f"[grid] skip  {tag} (already complete)", flush=True)
        results[tag] = "skipped"
        return
    wait_for_memory(tag)
    env = dict(os.environ, PYTHONUTF8="1", KMP_DUPLICATE_LIB_OK="TRUE",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
               NUMBA_NUM_THREADS=str(threads))
    t0 = time.time()
    print(f"[grid] START {tag}  {datetime.now():%H:%M:%S}", flush=True)
    with open(os.path.join(base, "logs", f"{tag}.log"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cell_cmd(ds, seed, out, epochs, smoke, threads, draws),
                            stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    release_memory()
    dt = time.time() - t0
    ok = rc == 0 and is_complete(rj)
    results[tag] = "ok" if ok else f"FAIL(rc={rc})"
    # Write status after EVERY cell, not only at the end. A grid that is killed
    # externally (the first attempt lost 1h45 of work when three cells were terminated
    # together, mid-line, with no traceback) otherwise leaves no record of what finished.
    try:
        with open(os.path.join(base, "grid_status.json"), "w") as sf:
            json.dump(results, sf, indent=2)
    except Exception:
        pass
    print(f"[grid] {'DONE ' if ok else 'FAIL '} {tag}  {dt / 60:.1f} min"
          + ("" if ok else f"  (see {base}/logs/{tag}.log)"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-par", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="BLAS/numba threads per cell")
    ap.add_argument("--draws", type=int, default=1,
                    help="independent Poisson featurizations of the train set "
                         "(1 = cheap route; see run_frozen.py --draws)")
    ap.add_argument("--smoke", action="store_true", help="tiny splits, to prove the wiring")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    run_id = a.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base = os.path.join(REPO, "results", "frozen_supervised", run_id)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)
    par = a.max_par     # hard cap; the per-cell memory gate does the adapting

    # SEED-OUTER: the whole single-seed grid finishes before seed 1 starts, so even a
    # grid that gets cut short leaves a complete five-dataset comparison rather than
    # three seeds of MNIST and nothing else.
    cells = [(ds, s) for s in range(a.seeds) for ds in a.datasets]
    print(f"[grid] run_id={run_id}  cells={len(cells)}  max concurrency={par}  "
          f"epochs={a.epochs}{'  SMOKE' if a.smoke else ''}", flush=True)
    print(f"[grid] {free_gb():.1f} GB free; each cell waits for "
          f"{MEM_PER_CELL_GB + 1.5:.1f} GB before starting", flush=True)
    print(f"[grid] -> {base}", flush=True)

    results, sem = {}, threading.Semaphore(par)
    def worker(ds, seed):
        with sem:
            run_cell(ds, seed, base, a.epochs, a.smoke, a.threads, a.draws, results)
    threads = [threading.Thread(target=worker, args=c) for c in cells]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = sum(1 for v in results.values() if v in ("ok", "skipped"))
    print(f"\n[grid] {done}/{len(cells)} complete in {(time.time() - t0) / 3600:.2f} h")
    for k in sorted(results):
        print(f"  {k:<24} {results[k]}")
    with open(os.path.join(base, "grid_status.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
