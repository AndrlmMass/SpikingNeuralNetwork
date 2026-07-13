"""
Mechanism-harness ladder: D&C regime -> our regime, tracking selectivity,
orientation coherence, readout drift, and recurrent-vs-feedforward drive.

Cells (all one-to-one WTA, PCA+LR readout):
  A0 random   frozen   FF     random FF reference
  A1 random   trace    FF     D&C reproduction (STDP should build selectivity)
  A2 random   trace    +EE    add recurrency to D&C
  B0 oriented frozen   FF     oriented FF reference
  B1 oriented trace    FF     oriented feedforward
  B2 oriented trace    +EE    our full regime (degradation case)
  B3 oriented triplet  +EE    rule variant
  R1 oriented reward   FF     supervised reward-STDP (V1) — compare vs B1/B2

  python experiments/RF_article/interp/run_interp_sweep.py --parallel 6
"""
import argparse, concurrent.futures as cf, datetime as dt, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(HERE, "interp_harness.py")

CELLS = [
    ("A0_rand_frozen_ff",  ["--prior", "random",   "--rule", "frozen"]),
    ("A1_rand_trace_ff",   ["--prior", "random",   "--rule", "trace"]),
    ("A2_rand_trace_ee",   ["--prior", "random",   "--rule", "trace",   "--ee"]),
    ("B0_ori_frozen_ff",   ["--prior", "oriented", "--rule", "frozen"]),
    ("B1_ori_trace_ff",    ["--prior", "oriented", "--rule", "trace"]),
    ("B2_ori_trace_ee",    ["--prior", "oriented", "--rule", "trace",   "--ee"]),
    ("B3_ori_triplet_ee",  ["--prior", "oriented", "--rule", "triplet", "--ee"]),
    ("R1_ori_reward_ff",   ["--prior", "oriented", "--rule", "reward"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(tag, flags, run_dir, common):
    out = os.path.join(run_dir, tag); os.makedirs(out, exist_ok=True)
    if os.path.isfile(os.path.join(out, "results.json")):
        return tag, "skipped", 0.0
    cmd = [sys.executable, HARNESS, "--tag", tag, *flags, *common, "--output-dir", out]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    ok = pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json"))
    return tag, ("OK" if ok else f"FAIL(rc={pr.returncode})"), time.time() - t0


def main():
    a = parse_args()
    run_id = a.run_id or f"run_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "interp", run_id); os.makedirs(run_dir, exist_ok=True)
    common = ["--train-all", str(a.train_all), "--seed", str(a.seed)]
    print(f"Interp ladder -> {run_dir}\n  {len(CELLS)} cells, parallel {a.parallel}\n")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = {ex.submit(run_one, tag, fl, run_dir, common): tag for tag, fl in CELLS}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            tag, status, dt_s = fut.result()
            print(f"  [{i}/{len(CELLS)}] {tag:<20} {status:<12} {dt_s/60:5.1f} min", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Plot: "
          f"python experiments/RF_article/interp/plot_interp.py --results-dir {run_dir}")


if __name__ == "__main__":
    main()
