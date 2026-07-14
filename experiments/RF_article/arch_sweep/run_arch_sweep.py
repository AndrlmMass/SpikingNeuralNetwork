"""
Architecture sweep on ORIENTED RFs — establish the effect of excitatory STDP,
Vogels plastic inhibition, and one-to-one WTA, all on the RF substrate that is the
paper's contribution. Runs the config matrix in parallel and drops each into one
timestamped run folder under the top-level results/ tree.

Matrix (all oriented RFs, MNIST):
  1 frozen            exc=off vogels=off wta=off
  2 exc               exc=on  vogels=off wta=off
  3 exc_vogels        exc=on  vogels=on  wta=off
  4 wta_frozen        exc=off vogels=off wta=on
  5 wta_exc     (T1)  exc=on  vogels=off wta=on  EE=on
  6 wta_exc_noEE (T2) exc=on  vogels=off wta=on  EE=off
  7 wta_exc_vogels    exc=on  vogels=on  wta=on

Answers: STDP-on-RF (1v2), Vogels (2v3), WTA frozen/learning (1v4, 2v5),
EE under WTA (5v6), WTA+Vogels (5v7).

  python experiments/RF_article/arch_sweep/run_arch_sweep.py --parallel 6
Then compare with aggregate_arch.py on the printed run folder.
"""
import argparse, concurrent.futures as cf, datetime as dt, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN_ARCH = os.path.join(HERE, "run_arch.py")

CONFIGS = [
    ("frozen",         []),
    ("exc",            ["--exc"]),
    ("exc_vogels",     ["--exc", "--vogels"]),
    ("wta_frozen",     ["--wta"]),
    ("wta_exc",        ["--exc", "--wta"]),
    ("wta_exc_noEE",   ["--exc", "--wta", "--ablate-ee"]),
    ("wta_exc_vogels", ["--exc", "--wta", "--vogels"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(tag, flags, run_dir, common):
    out_dir = os.path.join(run_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "results.json")):
        return tag, "skipped", 0.0
    log = os.path.join(out_dir, "run.log")
    cmd = [sys.executable, RUN_ARCH, "--tag", tag, *flags, *common, "--output-dir", out_dir]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    ok = pr.returncode == 0 and os.path.isfile(os.path.join(out_dir, "results.json"))
    return tag, ("OK" if ok else f"FAIL(rc={pr.returncode})"), time.time() - t0


def main():
    a = parse_args()
    run_id = a.run_id or f"run_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "arch_sweep", run_id)
    os.makedirs(run_dir, exist_ok=True)
    common = ["--train-all", str(a.train_all), "--seed", str(a.seed)]

    print(f"Arch sweep -> {run_dir}\n  {len(CONFIGS)} configs, parallel {a.parallel}, "
          f"train_all {a.train_all}\n")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = {ex.submit(run_one, tag, fl, run_dir, common): tag for tag, fl in CONFIGS}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            tag, status, dt_s = fut.result()
            print(f"  [{i}/{len(CONFIGS)}] {tag:<18} {status:<12} {dt_s/60:5.1f} min", flush=True)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min wall.")
    print(f"Aggregate: python experiments/RF_article/arch_sweep/aggregate_arch.py --results-dir {run_dir}")


if __name__ == "__main__":
    main()
