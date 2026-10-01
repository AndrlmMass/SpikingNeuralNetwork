"""Decay-rate calibration for the continuous-decay baseline (post-2026-09-30 fixes).

Reproduces the documented procedure (experiment.py, DECAY_RATE comment): run on
MNIST, record the total excitatory |w| over the first 70k training timesteps, and
choose the per-timestep decay rate whose weight scale at T=70k matches the one
the sleep arm reaches. Records W(t)/W0 every 5k steps for one arm per call.

    python decay_calibration.py none
    python decay_calibration.py decay 2e-5
    python decay_calibration.py sleep <noise_sd>

Writes results/decay_calibration/<arm>_<value>_<dataset>_s<seed>.json
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import train  # noqa: E402
from sweep import lambda_for_ratio  # noqa: E402

T_STOP, EVERY = 70_000, 5_000
arm = sys.argv[1]
value = sys.argv[2] if len(sys.argv) > 2 else "na"
dataset = os.environ.get("CAL_DATASET", "mnist")
seed = os.environ.get("CAL_SEED", "100")
out = os.path.join(REPO, "results", "decay_calibration",
                   f"{arm}_{value}_{dataset}_s{seed}.json")

orig = train.update_weights
trace = {}


def wrapped(**kw):
    # Sleep-phase calls pass virtual time and sleep_now_exc=True; skip them.
    if kw["sleep_now_exc"]:
        return orig(**kw)
    t = int(kw["t"])
    w, n_x, n_inh = kw["weights"], kw["N_x"], kw["N_inh"]
    ih = w.shape[1]
    ex = ih - n_inh
    if not trace or (t % EVERY == 0 and t > 0):
        trace[t] = float(abs(w[:ex, n_x:ih]).sum())
    if t >= T_STOP and t % EVERY == 0:
        w0 = trace[min(trace)]
        rec = {"arm": arm, "value": value, "dataset": dataset, "seed": seed,
               "W0": w0, "ratio_by_t": {k: v / w0 for k, v in sorted(trace.items())}}
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"CAL {arm} {value} {dataset}: W(70k)/W0 = {trace[max(trace)] / w0:.3f}", flush=True)
        raw = os.path.join(REPO, "results", f"results__cal_{arm}_{value}_{dataset}_s{seed}.json")
        if os.path.exists(raw):
            os.remove(raw)
        os._exit(0)
    return orig(**kw)


train.update_weights = wrapped
os.chdir(REPO)
argv = ["main.py", "--dataset", dataset, "--seed", seed, "--runs", "1",
        "--out-tag", f"_cal_{arm}_{value}_{dataset}_s{seed}", "--num-steps", "100",
        "--check-sleep-interval", "3500", "--clip-always", "--no-plots"]
if arm == "none":
    argv += ["--reg-method", "none", "--sleep-rate", "0.0"]
elif arm == "decay":
    argv += ["--reg-method", "decay", "--decay-rate", value, "--sleep-rate", "0.0"]
elif arm == "sleep":
    ratio = float(os.environ.get("CAL_RATIO", "0.1"))
    argv += ["--reg-method", "sleep", "--sleep-rate", str(ratio),
             "--sleep-decay-rate", str(lambda_for_ratio(ratio)), "--sleep-max-iters", "35000",
             "--on-timeout", "give_up", "--sleep-termination", "band",
             "--sleep-components", "downscale,noise,stdp,suppress",
             "--sleep-noise-var", value]
    out = out.replace(".json", f"_r{ratio:g}.json")
else:
    raise SystemExit(f"unknown arm {arm}")
sys.argv = argv
try:
    runpy.run_path(os.path.join(HERE, "main.py"), run_name="__main__")
finally:
    raw = os.path.join(REPO, "results", f"results__cal_{arm}_{value}_{dataset}_s{seed}.json")
    if os.path.exists(raw):
        os.remove(raw)
