"""
Reward-STDP control suite (sequential): does the reward signal actually drive the
readout gain, or is it an artifact of Normalize/homeostasis?

Runs 3 tiled conditions on a SHORT budget and overlays their readout-accuracy /
online-accuracy trajectories:

  reward     : --rule reward --tiled  --reward-lr <lr>          (the real rule)
  reward_off : --rule reward --tiled  --reward-lr 0             (control: reward off, everything else identical)
  shuffle    : --rule reward --tiled  --reward-lr <lr> --shuffle-labels  (control: reward on random targets)

Verdict: reward WORKS if its readout accuracy rises clearly ABOVE reward_off and
shuffle (which should both stay near the init/baseline level). If all three track
together, the "gain" is not the reward.

  python experiments/RF_article/rstdp/run_controls.py
  python experiments/RF_article/rstdp/run_controls.py --reward-lr 2e-5 --train-all 5000
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HARNESS = os.path.join(REPO, "experiments", "RF_article", "interp", "interp_harness.py")

CONDS = {
    "reward":     [],
    "reward_off": ["--reward-lr", "0"],
    "shuffle":    ["--shuffle-labels"],
}
COLORS = {"reward": "#2ca02c", "reward_off": "#999999", "shuffle": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-lr", type=float, default=2e-5)
    p.add_argument("--train-all", type=int, default=5000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def run_one(cond, extra, out, a):
    if os.path.isfile(os.path.join(out, "results.json")):
        return "skipped"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, HARNESS, "--tag", cond, "--prior", "isotropic", "--rule", "reward",
           "--tiled", "--reward-lr", str(a.reward_lr), "--seed", str(a.seed),
           "--train-all", str(a.train_all), "--val-all", str(a.val_all),
           "--val-every", "1", "--test-all", str(a.test_all), "--output-dir", out, *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUTF8="1")
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        pr = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return "OK" if pr.returncode == 0 and os.path.isfile(os.path.join(out, "results.json")) else f"FAIL(rc={pr.returncode})"


def traj_series(run_dir, cond, key):
    rp = os.path.join(run_dir, cond, "results.json")
    if not os.path.isfile(rp):
        return [], []
    tr = json.load(open(rp)).get("trajectory", [])
    xs = [t["batch"] + 1 for t in tr]
    ys = [t.get(key, np.nan) for t in tr]
    return xs, ys


def main():
    a = parse_args()
    run_id = a.run_id or f"controls_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = os.path.join(REPO, "results", "rstdp_controls", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"reward-STDP controls (sequential) -> {run_dir}\n  lr={a.reward_lr} train_all={a.train_all}\n")
    t0 = time.time()
    for cond, extra in CONDS.items():
        t1 = time.time()
        status = run_one(cond, extra, os.path.join(run_dir, cond), a)
        print(f"  {cond:<11} {status:<12} {(time.time()-t1)/60:5.1f} min", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (key, title) in zip(axes, [("softmax_acc", "Readout accuracy (softmax) — held-out val"),
                                       ("online_acc", "Online train decisions (net's own pooled argmax)")]):
        for cond in CONDS:
            xs, ys = traj_series(run_dir, cond, key)
            if xs:
                ax.plot(xs, ys, "-o", ms=3, color=COLORS[cond], label=cond)
        ax.axhline(0.1, ls=":", lw=0.8, color="k", alpha=0.5)
        ax.set_ylim(0, 1.02); ax.set_xlabel("checkpoint (batch+1)"); ax.set_ylabel("accuracy")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("RSTDP controls: reward vs reward-off vs shuffled-labels", fontsize=13)
    fig.tight_layout()
    out = os.path.join(run_dir, "controls.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"\n{'cond':>11} {'soft_init':>10} {'soft_final':>11} {'delta':>7} {'online_final':>13}")
    for cond in CONDS:
        xs, s = traj_series(run_dir, cond, "softmax_acc")
        _, o = traj_series(run_dir, cond, "online_acc")
        if xs:
            print(f"{cond:>11} {s[0]:>10.3f} {s[-1]:>11.3f} {s[-1]-s[0]:>+7.3f} {o[-1]:>13.3f}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out}")
    print("Verdict: reward WORKS if its softmax curve rises clearly above reward_off AND shuffle.")


if __name__ == "__main__":
    main()
