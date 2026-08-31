"""
The frozen control for the SUPERVISED model: what does the RF prior support on its own?

Phase 2 (mnist_family_sweep) has NO frozen cell, and phase 1 differs from it in six
config parameters at once, so "frozen beats every plastic rule" currently cannot be said
about the supervised model at all. This script supplies the missing cell: the byte-
identical phase-2 architecture with the excitatory weights never updated, on all five
datasets.

WHAT "FROZEN" MEANS HERE, and why it cannot mean everything
-----------------------------------------------------------
W_se is fixed at its oriented-RF initialization (reward_lr = 0) while the dense readout
still learns. Freezing the readout TOO would be a degenerate experiment, not a stricter
one: `group_tiled_centers` gives every class group the SAME k x k centre grid, and the
tiled path assigns orientations as `arange(N_exc) % 4`, which has period 4 into groups of
100 -- so all ten class groups start byte-identical (verified: max|W_g0 - W_g9| = 0.0).
A pooled readout over identical groups is at exactly chance by construction. Only the
dense readout, which reads all 1000 neurons rather than one group, can decode anything
from this network, so it is the only readout that makes the frozen condition a measurement
rather than a tautology.

NOT THE HEADLINE PATH -- a fast diagnostic. Use run_all.py for the article numbers.
--------------------------------------------------------------------------------
This script featurizes each split ONCE and then replays the readout's delta rule offline
on the cached features. That was designed as an exact ~10x shortcut for a frozen network,
on the reasoning that fixed W_se means an image always evokes the same exc-rate vector.

THAT REASONING IS WRONG, and the measurement is recorded here so nobody re-derives it:
the input encoding is POISSON, so every presentation of the same image draws a fresh
spike train. Featurizing the same 300 test images twice, in one process, with byte-
identical frozen weights, gives features correlating 0.968 -- not 1.0. The online readout
therefore sees a fresh noisy view of each image every epoch (implicit augmentation) while
this replay sees one frozen draw repeated. On a matched 3000-image / 2-epoch pair the
replayed readout reached test 0.807 against the online 0.830, and the two W_dense matrices
correlated only 0.54.

So the replay is a legitimate but DIFFERENT experiment ("readout trained on one fixed
noise draw per image"), and it must not be reported as the frozen cell. It remains useful
for what it is genuinely good at: the offline readout-size sweep, which costs nothing once
the features are cached and shows how sample-starved the decoder is (the linear probe on
this representation is known to be starved below ~7000 samples, 07-25).

The freeze itself is asserted at run time, not assumed: `reward_lr = 0` still leaves
train_weights=True, so the Normalize regularizer runs, and a previous bug in this repo
meant a nominally frozen condition was not frozen. drift_check compares W_se before and
after entering training AND after featurization, and this script refuses to report a
number if either moved beyond float rounding.

Usage:
    python experiments/RF_article/frozen_supervised/run_frozen.py --dataset mnist --seed 0
    python experiments/RF_article/frozen_supervised/run_frozen.py --dataset mnist --seed 0 \
        --train-all 800 --val-all 400 --test-all 300     # smoke
"""
import argparse, json, math, os, sys, time
from datetime import datetime

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "RF_article", "interp"))

import neurosnn as snn  # noqa: E402
from neurosnn._network.init_weights import make_group_assignment  # noqa: E402

# Phase-2 canonical config (mnist_family_sweep/run_slurm.sh). Changing any of these
# breaks comparability with the 50-run phase-2 grid, which is the entire point of the cell.
N_EXC = N_INH = 1000
N_GROUPS = 10
PEAK_EI, PEAK_IE = 50.0, -2.0
RF_LENGTH, RF_THICKNESS, CENTER_MARGIN = 3.0, 1.2, 4.0
READOUT_LR = 0.1
NUM_STEPS, MAX_RATE_HZ = 350, 90.0

# Per-dataset split. val is the CALIBRATION set for the abstention threshold and is
# carved out of the dedicated TRAIN split, so train shrinks to keep the totals exact.
# Phase 2 used 59000/1000; this cell uses 55000/5000 -- a difference that touches only
# the readout's fit, which the offline size-sweep below measures directly.
SPLITS = {
    "mnist":    (55000, 5000, 10000),   # 60000 train split, exact fill
    "fmnist":   (55000, 5000, 10000),
    "kmnist":   (55000, 5000, 10000),
    "svhn":     (68257, 5000, 26032),   # 73257 train split, exact fill
    "notmnist": (11000, 5000,  2724),   # 18724 merged pool, exact fill
}
READOUT_FIT_SIZES = (1000, 5000, 15000, 55000)
PROBE_FIT_N = 5000   # match phase 2's --probe-fit-all


FREEZE_TOL = 1e-6      # relative; see drift_check


def drift_check(W0, W1, tol=FREEZE_TOL):
    """Did W_se actually stay put? Tolerance-based, and the tolerance is not arbitrary.

    Exact byte equality is the WRONG test here. `reward_lr = 0` gates the STDP update off,
    but train_weights is still True so Normalize(mode="neuron") runs, rescaling each
    neuron's weights back to its INITIAL column sum. On unmoved weights that multiplier is
    exactly 1.0, so the operation is a no-op in intent -- but it is still a float multiply,
    and it leaves rounding behind. Measured on this architecture: max|dW| = 4.8e-10 on
    weights with max 1.49, per-neuron sum ratio 1.000000 (std 1.5e-16), within-column ratio
    std 4.3e-16, and zero entries entering or leaving the support. That is rounding.

    Real plasticity is orders of magnitude larger: phase 2's reward_lr is 5e-6 applied with
    a count-product eligibility over tens of thousands of samples. A 1e-6 relative tolerance
    sits far above the rounding floor (~3e-10 relative) and far below any genuine update, so
    it catches the [[project_frozen_not_frozen_bug]] failure without tripping on arithmetic.

    Support change is reported separately and must be exactly zero: a rescale cannot create
    or destroy a synapse, so any change there is structural, not numerical.
    """
    d = float(np.abs(np.asarray(W1) - np.asarray(W0)).max())
    scale = float(np.abs(W0).max()) or 1.0
    rel = d / scale
    support_changed = int(((np.asarray(W0) != 0) != (np.asarray(W1) != 0)).sum())
    return dict(abs=d, rel=rel, support_changed=support_changed,
                ok=bool(rel <= tol and support_changed == 0))


# --------------------------------------------------------------------------- model

def build(dataset, seed, train_all, val_all, test_all, val_batch=1000, prior="oriented"):
    """The phase-2 network with reward_lr = 0. Returns (model, learner, assignment, st, ex).

    prior="random" gives the fourth cell of the prior x plasticity design, which the
    oriented-only version left missing: without it, the frozen column cannot distinguish
    what the ORIENTED prior contributes from what any fixed random projection feeding a
    trained linear readout would contribute. It uses snn.weights.random with the same
    density and peak parameters phase 2's random arm uses (interp_harness.py:337), so the
    two random arms differ only in whether the excitatory weights subsequently move.

    Note the tiled-group degeneracy does NOT apply here: random weights are drawn
    independently per neuron, so the ten class groups are not identical and the pooled
    readout is not pinned to chance the way it is under the tiled oriented prior.
    """
    st, ex = 784, 784 + N_EXC
    wkw = dict(density_se=0.01, density_ee=0.0, density_ei=0.03, density_ie=0.05,
               peak_se=4.0, peak_ee=1.0, peak_ei=PEAK_EI, peak_ie=PEAK_IE)
    if prior == "random":
        weights = snn.weights.random(wta_inhibition=True, **wkw)
    else:
        weights = snn.weights.grouped_excitatory(
            n_groups=N_GROUPS, group_layout="block", tiled=True, oriented=True,
            n_orientations=4, orientation_mode="block",
            sigma_x=RF_LENGTH, gamma=RF_THICKNESS / RF_LENGTH,
            tiled_center_margin=CENTER_MARGIN, **wkw)
    layer = snn.Layer(N_exc=N_EXC, N_inh=N_INH, weights=weights, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5))
    # reward_lr = 0 freezes W_se; readout_lr is irrelevant to the ONLINE pass here
    # (we only use this learner to build the runner) but is kept at the phase-2 value
    # so the config recorded in results.json is the one that was actually replayed.
    learner = snn.learner.RewardSTDP(learning_rate=0.0, class_assignment="block",
                                     seed=seed, readout_lr=READOUT_LR, dense_readout=True)
    model = snn.Model(
        input_size=784, classes=list(range(10)), random_state=seed, num_steps=NUM_STEPS,
        image_dataset=dataset, max_rate_hz=MAX_RATE_HZ, gain=1.0, gabor=False,
        all_images_train=train_all, batch_image_train=1000,
        all_images_val=val_all, batch_image_val=min(val_batch, val_all),
        all_images_test=test_all, batch_image_test=min(1000, test_all))
    return model, layer, learner, make_group_assignment(N_EXC, N_GROUPS, "block"), st, ex


# ------------------------------------------------------------------ offline readout

def replay_dense_readout_draws(draws, epochs=3, lr=READOUT_LR, n_classes=10, seed=0):
    """Replay the readout over a LIST of independent Poisson draws of the same train set.

    draws[e] is the feature matrix the network produced on pass e. The online rule sees a
    fresh spike train every epoch, because the input encoding is Poisson; replaying one
    cached draw `epochs` times instead is a different (noise-free) regime. Passing several
    draws reproduces the online regime exactly, at the cost of featurizing more than once.

    With one draw this cycles it, which is the cheap route -- justified only if the
    single-draw and multi-draw readouts agree on a properly powered test set. Falls back
    to cycling whenever fewer draws than epochs are supplied.
    """
    W = None
    for e in range(epochs):
        X, y = draws[e % len(draws)]
        W = replay_dense_readout(X, y, epochs=1, lr=lr, n_classes=n_classes,
                                 seed=seed, W_init=W)
    return W


def replay_dense_readout(X, y, epochs=3, lr=READOUT_LR, n_classes=10, seed=0,
                         shuffle_each_epoch=False, W_init=None):
    """The dense softmax-delta readout, trained offline on cached features.

    A line-for-line transcription of RewardLearner.step's dense branch
    (neurosnn/_core/synapses.py:493-501):

        er = exc / (exc.max() + 1e-9)
        scores = er @ W_dense
        p = softmax(scores)
        W_dense -= lr * outer(er, p - onehot)

    Sample-by-sample, in presentation order -- NOT a batched or vectorized
    approximation, because the delta rule is order-dependent and a batched version
    would be a different learner wearing the same name.

    shuffle_each_epoch=False matches the harness, which re-serves the same order.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    W = np.zeros((X.shape[1], n_classes), dtype=np.float64) if W_init is None else W_init
    rng = np.random.default_rng(seed)
    order = np.arange(len(y))
    for _ in range(epochs):
        if shuffle_each_epoch:
            rng.shuffle(order)
        for i in order:
            exc = X[i]
            m = exc.max()
            if m <= 0:
                continue                       # silent item: the online rule skips it too
            er = exc / (m + 1e-9)
            sc = er @ W
            p = np.exp(sc - sc.max())
            p /= p.sum() + 1e-12
            p[y[i]] -= 1.0                     # p - onehot
            W -= lr * np.outer(er, p)
    return W


def readout_scores(X, W):
    Xr = np.asarray(X, dtype=np.float64)
    Xr = Xr / (Xr.max(axis=1, keepdims=True) + 1e-9)
    return Xr @ W


def featurize_all(runner, n, part, batch=1000):
    """featurize exactly `n` items, not `n - (n % batch)`.

    runner.featurize walks `n // batch` whole batches, so any split that is not a multiple
    of the batch size is SILENTLY TRUNCATED -- notMNIST's 2724-item test set becomes 2000,
    SVHN's 26032 becomes 26000. That is not a rounding detail: it scores the cheap route on
    a different test set from the harness it is being validated against, which is exactly
    the comparison this script exists to make. Asking for a whole number of batches fixes
    it, because get_batch returns the short final batch (or None, which featurize breaks on).
    """
    want = int(math.ceil(n / batch) * batch)
    X, y = runner.featurize(want, batch, partition=part)
    if X is not None and len(y) > n:
        X, y = X[:n], y[:n]
    return X, y


# --------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", default="oriented", choices=("oriented", "random"),
                    help="frozen W_se init. 'random' supplies the missing fourth cell of "
                         "the prior x plasticity design.")
    ap.add_argument("--epochs", type=int, default=3, help="readout passes over the cached features")
    ap.add_argument("--train-all", type=int, default=None)
    ap.add_argument("--val-all", type=int, default=None)
    ap.add_argument("--test-all", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--draws", type=int, default=1,
                    help="independent Poisson featurizations of the TRAIN set. 1 (default) "
                         "is the cheap route: one forward pass per image, then replay the "
                         "readout over it. --draws N == epochs reproduces the online "
                         "regime, where the network sees fresh input noise each epoch, at "
                         "N times the forward-pass cost.")
    ap.add_argument("--keep-features", action="store_true",
                    help="also write features.npz (the full train cache; large)")
    a = ap.parse_args()

    tr, va, te = SPLITS[a.dataset]
    train_all = a.train_all if a.train_all is not None else tr
    val_all = a.val_all if a.val_all is not None else va
    test_all = a.test_all if a.test_all is not None else te
    out_dir = a.output_dir or os.path.join(
        REPO, "results", "frozen_supervised",
        datetime.now().strftime("run_%Y%m%d"), f"{a.dataset}_frozen_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[frozen] dataset={a.dataset} seed={a.seed} "
          f"train={train_all} val={val_all} test={test_all} epochs={a.epochs}", flush=True)

    model, layer, learner, assignment, st, ex = build(
        a.dataset, a.seed, train_all, val_all, test_all, prior=a.prior)
    reg = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    # The runner (and the initial weight matrix) are built eagerly when the generator is
    # created, but featurize needs a live _trainer, which only exists once training has
    # actually been entered. One batch is enough, and with reward_lr = 0 it changes nothing.
    gen = model.train(layers=[layer], learner=learner, regularizer=reg, epochs=1,
                      train_weights=True, save_model=False, accuracy_method="pca_lr",
                      use_LR=True, use_phi=True, use_pca=False, track_stats=False)
    W0 = np.array(model._runner.model.weights[:st, st:ex], copy=True)
    for _ in gen:
        break
    W_after_touch = np.array(model._runner.model.weights[:st, st:ex], copy=True)
    drift = drift_check(W0, W_after_touch)
    # Expected True for the tiled oriented prior (identical centres and orientations per
    # group) and False for a random init, where each neuron is drawn independently.
    groups_identical = bool(np.array_equal(W0[:, :N_EXC // N_GROUPS],
                                           W0[:, N_EXC // N_GROUPS: 2 * (N_EXC // N_GROUPS)]))
    print(f"[frozen] W_se relative drift after one batch: {drift['rel']:.2e} "
          f"({'FROZEN' if drift['ok'] else 'MOVED'})   support changed: "
          f"{drift['support_changed']}   class groups byte-identical at init: "
          f"{groups_identical}", flush=True)
    if not drift["ok"]:
        raise SystemExit("W_se MOVED with reward_lr=0 -- the frozen condition is not "
                         "frozen. Do not trust any number from this run.")

    # ---- the one expensive step: one forward pass per image, ever -------------
    feats = {}
    train_draws = []
    for d in range(a.draws):
        ts = time.time()
        X, y = featurize_all(model._runner, train_all, "train")
        if X is None:
            raise SystemExit("featurize returned nothing for partition=train")
        train_draws.append((X.astype(np.float32), y.astype(np.int16)))
        print(f"[frozen]   featurized train draw {d + 1}/{a.draws}: {X.shape} "
              f"in {time.time() - ts:.0f}s", flush=True)
    feats["train"] = train_draws[0]
    for part, n in (("val", val_all), ("test", test_all)):
        ts = time.time()
        X, y = featurize_all(model._runner, n, part)
        if X is None:
            raise SystemExit(f"featurize returned nothing for partition={part}")
        feats[part] = (X.astype(np.float32), y.astype(np.int16))
        print(f"[frozen]   featurized {part}: {X.shape} in {time.time() - ts:.0f}s", flush=True)

    drift_end = drift_check(W0, np.asarray(model._runner.model.weights[:st, st:ex]))
    if not drift_end["ok"]:
        raise SystemExit("W_se moved DURING featurization -- featurize is not read-only.")
    print(f"[frozen] W_se relative drift across the whole run: {drift_end['rel']:.2e}",
          flush=True)

    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    Xte, yte = feats["test"]

    # ---- offline: the readout, and its sample-efficiency curve, for free ------
    curve = []
    best = None
    sizes = [z for z in READOUT_FIT_SIZES if z <= len(ytr)]
    if len(ytr) not in sizes:
        sizes.append(len(ytr))
    for n in sizes:
        W = replay_dense_readout_draws([(Xd[:n], yd[:n]) for Xd, yd in train_draws],
                                       epochs=a.epochs, seed=a.seed)
        acc = float((readout_scores(Xte, W).argmax(1) == yte).mean())
        curve.append(dict(n_train=int(n), test_acc=acc))
        print(f"[frozen]   readout on {n:>6} train features -> test {acc:.4f}", flush=True)
        best = (n, W, acc)
    n_best, W_best, acc_best = best

    # The other two decoders, so aggregate_frozen can name every column and the
    # frozen-vs-plastic comparison is never made against an unlabelled accuracy.
    # Both are free here: the features are already in memory.
    from sklearn.linear_model import LogisticRegression        # noqa: E402
    from sklearn.preprocessing import StandardScaler           # noqa: E402
    # Fit on PROBE_FIT_N features, matching phase 2's --probe-fit-all 5000. Fitting on all
    # of train instead would make the probe column incomparable: on notMNIST an 11000-feature
    # fit scored 0.8175 against the harness's 5000-feature 0.7999, a 1.8pp difference that is
    # purely fit-set size (the probe on this representation is data-starved below ~7000, 07-25).
    nprobe = min(PROBE_FIT_N, len(ytr))
    sc = StandardScaler().fit(Xtr[:nprobe])
    clf = LogisticRegression(max_iter=2000, n_jobs=1).fit(
        np.nan_to_num(sc.transform(Xtr[:nprobe])), ytr[:nprobe])
    probe_test = clf.predict_proba(np.nan_to_num(sc.transform(Xte)))
    probe_cal = clf.predict_proba(np.nan_to_num(sc.transform(Xva)))
    lin_acc = float((probe_test.argmax(1) == yte).mean())
    # pooled readout: at chance by construction on a frozen tiled net (all ten class
    # groups are byte-identical at init), reported so the degeneracy stays visible
    Rt = np.stack([Xte[:, np.asarray(assignment) == g].mean(1) for g in range(N_GROUPS)], 1)
    pooled_acc = float((Rt.argmax(1) == yte).mean())
    print(f"[frozen]   linear probe (fit on {nprobe} train feats) -> test {lin_acc:.4f}"
          f"   pooled readout -> {pooled_acc:.4f} (chance = 0.10 by construction)",
          flush=True)

    score_cal = readout_scores(Xva, W_best)
    score_test = readout_scores(Xte, W_best)

    cfg = dict(tag=f"{a.dataset}_frozen{'' if a.prior == 'oriented' else '_rnd'}_s{a.seed}",
               dataset=a.dataset, prior=a.prior,
               rule="frozen_se_plastic_readout", reward_lr=0.0, readout_lr=READOUT_LR,
               dense_readout=True, grouped=True, n_groups=N_GROUPS, group_layout="block",
               tiled=True, n_exc=N_EXC, n_inh=N_INH, peak_ei=PEAK_EI, peak_ie=PEAK_IE,
               rf_length=RF_LENGTH, rf_thickness=RF_THICKNESS, center_margin=CENTER_MARGIN,
               train_all=train_all, val_all=val_all, test_all=test_all,
               epochs=a.epochs, seed=a.seed)
    cfg["draws"] = a.draws
    out = dict(config=cfg,
               test_acc=acc_best,                      # the learned dense readout
               test_lin_acc=lin_acc,                   # L1-style probe on train features
               test_pooled_acc=pooled_acc,             # chance by construction; see above
               readout_fit_curve=curve, readout_fit_n=int(n_best),
               draws=a.draws,
               frozen_verified=drift_end["ok"], frozen_rel_drift=drift_end["rel"],
               frozen_support_changed=drift_end["support_changed"],
               groups_identical_at_init=groups_identical,
               wall_seconds=round(time.time() - t0, 1))

    np.savez_compressed(
        os.path.join(out_dir, "uncertainty_features.npz"),
        X_cal=Xva, y_cal=yva, X_test=Xte, y_test=yte,
        assignment=np.asarray(assignment),
        score_cal=score_cal.astype(np.float32), score_test=score_test.astype(np.float32),
        probe_cal=probe_cal.astype(np.float32), probe_test=probe_test.astype(np.float32))
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "weights", "checkpoint.npz"),
                        W_se=W0.astype(np.float32), W_dense=W_best.astype(np.float32),
                        assignment=np.asarray(assignment), config=json.dumps(cfg))
    if a.keep_features:
        np.savez_compressed(os.path.join(out_dir, "features.npz"),
                            X_train=Xtr, y_train=ytr)

    try:
        sys.path.insert(0, os.path.join(REPO, "experiments", "RF_article", "interp"))
        from uncertainty import run_from_features, format_report
        reports = run_from_features(Xva, yva, Xte, yte, assignment=np.asarray(assignment),
                                    n_groups=N_GROUPS, probe_cal=probe_cal,
                                    probe_test=probe_test, score_cal=score_cal,
                                    score_test=score_test)
        out["uncertainty"] = reports
        for r in reports:
            print(format_report(r), flush=True)
        from plot_risk_coverage import make_risk_coverage_plots
        make_risk_coverage_plots(out_dir)
    except Exception as e:
        print(f"  [uncertainty] skipped: {type(e).__name__}: {e}", flush=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[frozen] DONE {a.dataset} s{a.seed}  test_acc={acc_best:.4f} "
          f"({time.time() - t0:.0f}s) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
