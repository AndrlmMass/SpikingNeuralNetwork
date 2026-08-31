"""
Mechanism harness (single config): track WHAT learning does to the representation
on the oriented-RF prior vs a random prior, feedforward vs recurrent, under
frozen / trace-STDP / triplet-STDP. All configs use one-to-one WTA (inhibition
held constant for this run).

At every val checkpoint we snapshot the live weights (TrainResult.weights) and the
val features (per-item exc firing rates, captured from the Evaluator) and compute:

  1. class_selectivity  — per-neuron peakedness of the class-response (1 - H/logK);
     high = each neuron specialises to a class (the D&C objective).
  2. orientation_coherence — structure-tensor coherence of each RF (W_se column);
     high = oriented / edge-like. Tracks whether STDP erodes the oriented prior.
  3. readout drift — accuracy of a readout FROZEN at init vs a readout REFIT each
     checkpoint. A growing gap = the representation is drifting off the class-useful
     directions it started on.
  4. current decomposition — mean |SE| (feedforward) vs |EE| (recurrent) vs |IE|
     drive per exc neuron; tests whether recurrence is even influential.

Also saves RF snapshot grids (first / mid / last) and a weight-floor fraction.

Usage:
  python experiments/RF_article/interp/interp_harness.py --prior oriented --rule trace --ee --tag B2 --output-dir <dir>
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import neurosnn as snn
from neurosnn._evaluation import evaluation as evalmod
from neurosnn._evaluation.analysis import (
    class_selectivity, orientation_coherence, current_decomp, w_floor_frac,
    pool_by_label, coverage_stats, softmax_readout, spike_share_metrics,
    pool_by_label_pred, softmax_readout_pred, confusion_matrix, group_rf_diversity,
    class_eta_squared, response_correlation, participation_ratio, active_mask,
    rf_gaussian_moments,
)
from neurosnn._plot.weights import save_rf_grid, plot_group_rfs

import matplotlib
matplotlib.use("Agg")

CAP = {"rows": []}
_orig_score = evalmod.Evaluator.score
def _cap(self, X, Y):
    CAP["rows"].append((np.asarray(X).copy(), np.asarray(Y).copy()))
    return _orig_score(self, X, Y)
evalmod.Evaluator.score = _cap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ---------------- readout drift ----------------
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def fit_clf(X, y, seed=0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=400, n_jobs=-1,
                             random_state=seed).fit(np.nan_to_num(sc.transform(X)), y)
    return sc, clf

def score_clf(sc, clf, X, y):
    return accuracy_score(y, clf.predict(np.nan_to_num(sc.transform(X))))


# ---------------- weight block slicing ----------------
def blocks(weights, st, ex, ih):
    return (weights[:st, st:ex], weights[st:ex, st:ex], weights[ex:ih, st:ex])


# The extended MNIST family. All 10-class; the streamer grayscales + resizes every
# dataset to 28x28 (see ImageDataStreamer's transform), so N_x=784 and the oriented-RF
# architecture is byte-identical across them -- only the task changes. CIFAR-10 and SVHN
# are natively 32x32x3 and are collapsed to 28x28 grayscale by that same transform.
_DATASETS = ["mnist", "kmnist", "fmnist", "fashionmnist", "notmnist", "cifar10", "svhn"]


def load_input_rate(dataset, pixel_size=28):
    """Mean per-pixel input intensity (784,) ~ mean input rate.

    Read off the same grayscaled+resized train images the model trains on, via the
    model's own ImageDataStreamer. Correct for every dataset -- including 32x32x3
    CIFAR/SVHN and deeplake notMNIST -- because that streamer applies the same
    Grayscale + Resize(28) transform uniformly. (The model builds its streamer lazily
    inside fit(), and input_rate is needed before fit, so we build a throwaway one.)
    """
    from neurosnn._data.get_data import ImageDataStreamer
    s = ImageDataStreamer(
        data_dir=os.path.join(REPO, "data"), pixel_size=pixel_size, dataset=dataset
    )
    imgs = np.asarray(s.train_images)
    return imgs.reshape(imgs.shape[0], -1).mean(0, dtype=np.float64)  # (784,)


def featurize_ood(model, dataset, n_images, batch=1000):
    """Exc-rate features for `dataset`'s TEST split, through the ALREADY-TRAINED net.

    The out-of-distribution probe. We swap the runner's ImageDataStreamer for one built
    on `dataset`, run runner.featurize, and put the original streamer back. Safe because
    featurize runs trainer.step with training_mode="test", and the reward rule + its
    delta readout are both gated on `training_mode == "train"` (trainer.py:620, :390) --
    so no weight can move no matter what the OOD images do to the network.

    Every dataset is grayscaled and resized to 28x28 by the streamer's own transform, so
    N_x=784 holds and the same network can be fed all of them without reshaping.

    test_count=None asks the streamer for the ENTIRE canonical test split (or, for
    notMNIST, everything left over from its merged pool); featurize then stops early
    when get_batch runs dry, so `n_images` larger than the split is harmless.

    Returns (X, n_requested_shortfall) -- X is (n, N_exc), or None if nothing came back.
    """
    from neurosnn._data.get_data import ImageDataStreamer
    runner = model._runner
    inner = runner.model                       # the _network.model.Model that owns the streamer
    original = inner.image_streamer
    try:
        inner.image_streamer = ImageDataStreamer(
            data_dir="data", pixel_size=inner.pixel_size, num_steps=inner.num_steps,
            max_rate_hz=original.max_rate_hz, gain=original.gain, gabor=original.gabor,
            train_count=1, val_count=0, test_count=None,
            dataset=dataset, random_seed=inner.random_state)
        # featurize walks whole batches, so an n_images below one batch would still
        # pull a full batch; clamp so `--ood-all` means what it says on small probes.
        X, _ = runner.featurize(n_images, min(batch, max(1, n_images)), partition="test")
    finally:
        inner.image_streamer = original        # restore even if the OOD load blew up
    return X


def save_weight_checkpoint(path, weights, st, ex, reward_learner, assignment, cfg):
    """Persist W_se + the readout so this model can be re-probed without retraining.

    The harness runs with save_model=False and never wrote weights anywhere, which is
    exactly why the 60k/5ep run cannot be fed OOD data today. Small file: W_se is
    (784, N_exc) and the dense readout (N_exc, 10).
    """
    blob = dict(W_se=np.asarray(weights[:st, st:ex], dtype=np.float32),
                config=json.dumps(cfg))
    if assignment is not None:
        blob["assignment"] = np.asarray(assignment)
    rl = reward_learner
    if rl is not None:
        if getattr(rl, "dense_readout", False) and getattr(rl, "W_dense", None) is not None:
            blob["W_dense"] = np.asarray(rl.W_dense, dtype=np.float32)
        if getattr(rl, "w_readout", None) is not None:
            blob["w_readout"] = np.asarray(rl.w_readout, dtype=np.float32)
        if getattr(rl, "_A", None) is not None:
            blob["readout_A"] = np.asarray(rl._A, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **blob)
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--prior", choices=["oriented", "isotropic", "random"], required=True)
    p.add_argument("--rule", choices=["frozen", "trace", "triplet", "reward"], required=True)
    p.add_argument("--reward-lr", type=float, default=2e-5, help="reward-STDP learning rate (rule=reward); set 0 for the reward-off control")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="CONTROL: reward on random targets (signal=noise); readout still evaluated on true labels")
    p.add_argument("--readout-lr", type=float, default=0.0,
                   help="plastic cluster->class readout learning rate (0 = fixed uniform pooling)")
    p.add_argument("--dense-readout", action="store_true",
                   help="full (N_exc x n_classes) readout: every neuron votes on every class, signs "
                        "free (own-class +, competitors -). Default is the block-diagonal readout "
                        "(each neuron -> only its own class). Uses --readout-lr.")
    p.add_argument("--peak-ei", type=float, default=20.0,
                   help="E->I drive INTO interneurons (1:1 WTA; raise if inh barely spike)")
    p.add_argument("--peak-ie", type=float, default=-2.0,
                   help="I->E inhibition strength onto exc (intra-group WTA)")
    p.add_argument("--ee", action="store_true", help="enable E->E recurrence (default off=feedforward)")
    p.add_argument("--grouped", action="store_true", help="grouped excitatory architecture (intra-class WTA)")
    p.add_argument("--n-groups", type=int, default=10, help="number of excitatory groups (default 10)")
    p.add_argument("--group-layout", choices=["interleaved", "block"], default="interleaved")
    p.add_argument("--tiled", action="store_true",
                   help="tiled per-class RF centers: each class = a k x k grid tiling the full "
                        "input (block layout, full coverage). Forces N_exc=1000, grouped, block.")
    p.add_argument("--sigma-se", type=float, default=0.0,
                   help="RF Gaussian sigma (px) = structural footprint of each SE neuron. 0 = keep the "
                        "grouped default (3.0 ~ near-global, whole-digit templates). Lower = local patch "
                        "detectors: ~1.5 -> 6px, ~1.0 -> 3.6px. Sets sigma_se_mean on the weight spec.")
    p.add_argument("--rf-length", type=float, default=3.0,
                   help="ORIENTED RFs: sigma ALONG the bar (major axis) in px = length. "
                        "The oriented builder reads this (sigma_x), NOT --sigma-se.")
    p.add_argument("--rf-thickness", type=float, default=1.2,
                   help="ORIENTED RFs: sigma ACROSS the bar (minor axis) in px = thickness; "
                        "gamma = thickness/length. Domantas' tuned pair is 3.0 / 1.2 "
                        "(thin oriented bars), vs the old near-isotropic 3.0 / 1.2*3.0.")
    p.add_argument("--center-margin", type=float, default=0.0,
                   help="px trimmed per edge when tiling RF centers (0 = full 28x28). "
                        ">0 concentrates centers on the central region — MNIST digits are "
                        "centered, so full tiling wastes the outer tiles on blank border "
                        "pixels and those neurons die. Domantas uses 4.")
    p.add_argument("--ablate-ie", action="store_true",
                   help="zero the I->E block: no intra-group WTA competition at all")
    p.add_argument("--sigma-se-lognormal", type=float, default=0.0,
                   help="lognormal spread of per-neuron RF sigma (0 = uniform). >0 gives HETEROGENEOUS RF "
                        "sizes: a mix of small local-feature and larger holistic detectors.")
    p.add_argument("--spiking-readout", action="store_true",
                   help="ALSO run the spiking R-STDP readout alongside the delta readout. "
                        "Both see the same spikes on the same trial, so the comparison is "
                        "paired within-sample. Does not disable or alter the delta readout.")
    p.add_argument("--spiking-lr", type=float, default=0.005)
    p.add_argument("--spiking-margin", type=float, default=2.0,
                   help="how far the target must lead every competitor before the error "
                        "goes to zero (in output spike counts)")
    p.add_argument("--spiking-wmin", type=float, default=0.0,
                   help="readout weight floor. 0 = non-negative (default; negatives measured "
                        "worth only 0.6 points and w>=0 removes the silent-output failure "
                        "mode). Set <0 for the signed variant.")
    p.add_argument("--spiking-target-count", type=float, default=6.0,
                   help="output spikes/trial the intrinsic homeostat aims for")
    p.add_argument("--spiking-lateral-inh", type=float, default=0.0)
    p.add_argument("--spiking-noise", type=float, default=0.0,
                   help="membrane noise on the output neurons; R-STDP is a policy-gradient "
                        "rule and its theory needs trial-to-trial variability")
    p.add_argument("--theta-tau", type=float, default=0.0,
                   help="adaptive-threshold decay time constant (0 = keep current 200). Diehl-style "
                        "homeostasis uses ~1e7 (persistent: theta accumulates lifetime firing, "
                        "equalizing rates across the population). Large = permanent, small = transient fatigue.")
    p.add_argument("--theta-delta", type=float, default=0.0,
                   help="adaptive-threshold increment per spike (0 = keep current 0.5). Diehl uses ~0.05 "
                        "(small, so it accumulates gradually). Tune DOWN if theta blows up (our neurons "
                        "fire more than Diehl's: no refractory, current-based synapses).")
    p.add_argument("--use-vogels", action="store_true", help="Vogels iSTDP on I->E (plastic intra-group inhibition)")
    p.add_argument("--vogels-lr", type=float, default=0.01,
                   help="Vogels iSTDP learning rate (inhibitory plasticity strength; only used with --use-vogels)")
    p.add_argument("--vogels-rho0", type=float, default=0.1,
                   help="Vogels target postsynaptic rate rho_0 (homeostatic setpoint; only used with --use-vogels)")
    p.add_argument("--track-stats", action="store_true", help="enable weight/spike statistics tracking during training")
    p.add_argument("--live-plot", action="store_true", help="save the live class-tiled spike plot during training (grouped/tiled)")
    p.add_argument("--plot-every", type=int, default=1000,
                   help="timesteps between live spike plots AND stat snapshots (default 1000; "
                        "raise it on full runs, e.g. 35000 = once per 100 images, to limit frame count)")
    p.add_argument("--no-plots", action="store_true",
                   help="log metrics to results.json only; skip ALL in-run rendering (confusion.png, "
                        "group_rfs.png, RF grids). Use for long runs you'll plot yourself afterward.")
    p.add_argument("--plot-rfs", action="store_true", help="save RF grid and (oriented) summary/coverage plots after init")
    p.add_argument("--plot-single-neuron", action="store_true", help="save 2x2 SE/EE/EI/IE panel for one neuron after init")
    p.add_argument("--plot-schematic", action="store_true", help="save force-directed network graph from real weights (grouped only)")
    p.add_argument("--neuron-id", type=int, default=512, help="neuron index for --plot-single-neuron (default 512)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="mnist", choices=sorted(_DATASETS),
                   help="dataset (10-class); all grayscaled+resized to 28x28, N_x=784. "
                        "cifar10/svhn are collapsed from 32x32x3; notmnist via deeplake")
    p.add_argument("--epochs", type=int, default=1,
                   help="number of passes over the --train-all images (data is re-served each epoch)")
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-batch", type=int, default=1000,
                   help="images per validation BATCH (default 1000). Kept separate from "
                        "--val-all because the eval pass allocates a (batch*num_steps, N) "
                        "int8 spike buffer: a 5000-image batch is ~4.9 GB and OOMs, while "
                        "5 batches of 1000 is ~1 GB. Validation still covers all --val-all "
                        "images; only the chunking changes.")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--test-all", type=int, default=3000)
    p.add_argument("--probe-fit-all", type=int, default=0,
                   help="fit the FINAL linear probe (test_lin_acc + uncertainty) on this "
                        "many TRAIN-split features instead of on the ~1k val set. 0 = keep "
                        "the old val-fit behaviour. The 07-25 finding was that the probe is "
                        "data-starved at ~1k (88%%) and recovers to ~the dense readout at "
                        "~7k; set ~5000 so the linear-probe number reflects the representation, "
                        "not the fit-set size. Calibration/drift still use val; only the "
                        "classifier's FIT set changes.")
    p.add_argument("--ood-dataset", action="append", default=[],
                   metavar="NAME", choices=sorted(_DATASETS),
                   help="repeatable: after the in-distribution test, run a forward pass over "
                        "this dataset's TEST split and dump its features as X_ood_<name> in "
                        "uncertainty_features.npz. Nothing is trained on it and its labels are "
                        "discarded -- it is the OUT-OF-DISTRIBUTION probe: the question is "
                        "whether the network's own confidence falls low enough to abstain. "
                        "Uses runner.featurize (training_mode='test'), so no weight can move.")
    p.add_argument("--ood-all", type=int, default=10000,
                   help="images per --ood-dataset (capped by that dataset's test split)")
    p.add_argument("--save-weights", action="store_true",
                   help="save W_se + the readout weights to weights/checkpoint.npz so the "
                        "trained model can be re-probed later without retraining. The harness "
                        "otherwise persists NOTHING (save_model=False), which is why the "
                        "60k/5ep run cannot be reused for OOD.")
    p.add_argument("--output-dir", default=None,
                   help="run dir (default: results/<dataset>/<date>/<tag>_<uid>/); "
                        "the sweep passes results/<dataset>/<date>/<sweep_id>/<tag>/")
    return p.parse_args()


def default_output_dir(tag, dataset="mnist"):
    """Unified run dir: results/<dataset>/<date>/<tag>_<uid>/ (uid keeps standalone
    runs from overwriting each other)."""
    from datetime import datetime
    from uuid import uuid4
    date = datetime.now().strftime("%Y.%m.%d")
    return os.path.join(REPO, "results", dataset, date, f"{tag}_{uuid4().hex[:4]}")


def main():
    a = parse_args()
    if a.output_dir is None:
        a.output_dir = default_output_dir(a.tag)
    os.makedirs(a.output_dir, exist_ok=True)
    # tiled forces the grouped/block architecture at N_exc=1000 (10 classes x 10x10 tile)
    if a.tiled:
        a.grouped = True
        a.group_layout = "block"
    N_exc, N_inh = (1000, 1000) if a.tiled else (1024, 1024)   # WTA: N_inh == N_exc
    st, ex, ih = 784, 784 + N_exc, 784 + N_exc + N_inh
    train_weights = a.rule != "frozen"
    density_ee = 0.01 if a.ee else 0.0

    wkw = dict(density_se=0.01, density_ee=density_ee, density_ei=0.03, density_ie=0.05,
               peak_se=4.0, peak_ee=1.0, peak_ei=a.peak_ei, peak_ie=a.peak_ie,
               wta_inhibition=True)
    if a.prior == "oriented" and not a.grouped:
        weights = snn.weights.oriented_receptive_fields(n_orientations=4, orientation_mode="block", **wkw)
    elif a.prior == "isotropic" and not a.grouped:
        weights = snn.weights.receptive_fields(**wkw)
    elif a.prior in ("oriented", "isotropic") and a.grouped:  # grouped handles both RF shapes
        gkw = {k: v for k, v in wkw.items() if k != "wta_inhibition"}
        # ORIENTED RF geometry comes from sigma_x (length ALONG the bar) and
        # gamma = thickness/length (minor/major axis ratio) — NOT from sigma_se_mean,
        # which the factory reads only on the isotropic branch. That is why --sigma-se
        # is silently a no-op under --prior oriented; use --rf-length/--rf-thickness.
        # tiled_center_margin trims the tiled region so RF centers concentrate on the
        # central area: MNIST digits are centered, so full-input tiling puts the outer
        # tiles on permanently blank border pixels and those neurons die.
        weights = snn.weights.grouped_excitatory(
            n_groups=a.n_groups, group_layout=a.group_layout, tiled=a.tiled,
            oriented=(a.prior == "oriented"),
            n_orientations=4, orientation_mode="block",
            sigma_x=a.rf_length, gamma=(a.rf_thickness / a.rf_length),
            tiled_center_margin=a.center_margin, ablate_ie=a.ablate_ie, **gkw)
    else:
        weights = snn.weights.random(**wkw)
    # RF-size override: shrink the structural footprint (sigma_se_mean) to force local
    # patch detectors instead of near-global whole-digit templates, and/or make RF sizes
    # heterogeneous. Applied to the spec so it flows through _to_factory_kwargs.
    # ISOTROPIC ONLY — see the note above; on oriented priors this field is ignored.
    if a.sigma_se > 0.0 and hasattr(weights, "sigma_se_mean"):
        if a.prior == "oriented":
            print("  [warn] --sigma-se is ignored on --prior oriented; "
                  "use --rf-length / --rf-thickness", flush=True)
        weights.sigma_se_mean = a.sigma_se
    if a.sigma_se_lognormal > 0.0 and hasattr(weights, "sigma_se_lognormal_std"):
        weights.sigma_se_lognormal_std = a.sigma_se_lognormal

    layer = snn.Layer(N_exc=N_exc, N_inh=N_inh, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True,
        tau_adaptation=(a.theta_tau if a.theta_tau > 0 else 200.0),
        delta_adaptation=(a.theta_delta if a.theta_delta > 0 else 0.5)), weights=weights)

    if a.rule == "reward":
        learner = snn.learner.RewardSTDP(learning_rate=a.reward_lr,
            class_assignment=("block" if a.tiled else "mod"), seed=a.seed,
            shuffle_labels=a.shuffle_labels, readout_lr=a.readout_lr,
            dense_readout=a.dense_readout,
            spiking_readout=(dict(
                lr=a.spiking_lr, margin=a.spiking_margin, w_min=a.spiking_wmin,
                target_count=a.spiking_target_count, lateral_inh=a.spiking_lateral_inh,
                noise_std=a.spiking_noise, seed=a.seed,
            ) if a.spiking_readout else None))
    elif a.rule == "triplet":
        learner = snn.learner.TripletSTDP()
    else:
        learner = snn.learner.TraceSTDP(learning_rate=0.0004, tau_trace=20, w_max=10.0,
            mu_weight=0.5, x_tar_mode="mean", update_freq=100, clip_weights=True,
            min_weight_exc=0.01, max_weight_exc=25.0, min_weight_inh=-25.0, max_weight_inh=-0.01)

    inh_learner = snn.learner.VogelsSTDP(learning_rate=a.vogels_lr, rho_0=a.vogels_rho0) if a.use_vogels else None

    # neuron_class: used for pool_by_label readout and softmax_readout
    if a.rule == "reward" and a.grouped:
        from neurosnn._network.init_weights import make_group_assignment
        neuron_class = make_group_assignment(N_exc, a.n_groups, a.group_layout)
    elif a.rule == "reward":
        neuron_class = np.arange(N_exc) % 10
    else:
        neuron_class = None

    # group_assignment for softmax_readout (same as neuron_class when grouped+reward,
    # but kept separate so softmax can run on any rule when grouped)
    if a.grouped:
        from neurosnn._network.init_weights import make_group_assignment
        group_assignment = make_group_assignment(N_exc, a.n_groups, a.group_layout)
    else:
        group_assignment = None

    reg = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    model = snn.Model(input_size=784, classes=list(range(10)), random_state=a.seed, num_steps=350,
        all_images_train=a.train_all, batch_image_train=1000, all_images_val=a.val_all,
        batch_image_val=min(a.val_batch, a.val_all), all_images_test=a.test_all, batch_image_test=a.test_all,
        image_dataset=a.dataset, max_rate_hz=90.0, gain=1.0, gabor=False)

    input_rate = load_input_rate(a.dataset)
    cfg = dict(tag=a.tag, dataset=a.dataset, prior=a.prior, rule=a.rule, ee=a.ee, wta=True,
               grouped=a.grouped, n_groups=a.n_groups, group_layout=a.group_layout,
               use_vogels=a.use_vogels, vogels_lr=a.vogels_lr, vogels_rho0=a.vogels_rho0,
               n_exc=N_exc, n_inh=N_inh,
               sigma_se=a.sigma_se, sigma_se_lognormal=a.sigma_se_lognormal,
               rf_length=a.rf_length, rf_thickness=a.rf_thickness,
               center_margin=a.center_margin, ablate_ie=a.ablate_ie,
               peak_ei=a.peak_ei, peak_ie=a.peak_ie,
               readout_lr=a.readout_lr, dense_readout=a.dense_readout,
               theta_tau=a.theta_tau, theta_delta=a.theta_delta,
               train_all=a.train_all, epochs=a.epochs, seed=a.seed)
    print(f"\n[{a.tag}] prior={a.prior} rule={a.rule} ee={a.ee} grouped={a.grouped} "
          f"vogels={a.use_vogels} train_weights={train_weights}\n", flush=True)

    traj = []
    fixed = {"sc": None, "clf": None}
    last_val = {"X": None, "y": None}   # latest val features, for the test-set readout fit
    rf_saved = {}

    def checkpoint(batch, weights):
        CAP["rows"].clear()
        v = model.validate()
        if not CAP["rows"]:
            return
        X = np.concatenate([x for x, _ in CAP["rows"]], 0)
        y = np.concatenate([yy for _, yy in CAP["rows"]], 0).astype(int)
        W_se, W_ee, W_ie = blocks(weights, st, ex, ih)
        exc_rate = X.mean(0)
        se, ee, ie = current_decomp(W_se, W_ee, W_ie, exc_rate, input_rate)
        rng = np.random.default_rng(0); idx = rng.permutation(len(y))
        cut = int(0.7 * len(y)); tr, te = idx[:cut], idx[cut:]
        sc, clf = fit_clf(X[tr], y[tr], a.seed)
        refit = score_clf(sc, clf, X[te], y[te])
        if fixed["clf"] is None:
            fixed["sc"], fixed["clf"] = sc, clf
        fixed_acc = score_clf(fixed["sc"], fixed["clf"], X[te], y[te])
        # `selectivity` is kept only for continuity with old trajectories — it is
        # confounded by firing rate (see analysis.class_selectivity). The metrics
        # that should carry claims are eta2 (per-neuron, noise-aware), resp_corr
        # (population redundancy), and PR (effective dimensionality; needs >= K-1
        # to separate K classes at all).
        _corr_within, _corr_all = response_correlation(X, group_assignment, a.n_groups)
        rec = dict(batch=int(batch),
                   val_acc=float(v.accuracy) if v.accuracy is not None else float("nan"),
                   val_phi=float(v.phi) if v.phi is not None else float("nan"),
                   selectivity=class_selectivity(X, y),
                   eta2=class_eta_squared(X, y),
                   corr_within=_corr_within, corr_all=_corr_all,
                   pr=participation_ratio(X, scale_free=True),
                   pr_cov=participation_ratio(X, scale_free=False),
                   n_active=int(active_mask(X).sum()),
                   orient_coh=orientation_coherence(W_se),
                   refit_acc=float(refit), fixed_acc=float(fixed_acc),
                   cur_se=se, cur_ee=ee, cur_ie=ie, ee_se_ratio=ee / (se + 1e-12),
                   w_floor_frac=w_floor_frac(W_se))
        # Corrected dead fraction: fraction of exc neurons BELOW the relative
        # activity floor (active_mask), computed for EVERY run -- the coverage_stats
        # dead_frac below is only emitted when a class assignment exists (grouped/
        # reward), so the non-grouped phase-1 model would otherwise have no dead
        # measure. This is the "corrected" version (relative floor, not fired>0).
        rec["dead_frac_corrected"] = float(1.0 - rec["n_active"] / X.shape[1])
        # 2D-Gaussian RF geometry: diagonal variance + covariance terms evolving
        # over training (Hubin), reported next to orientation coherence.
        rec.update(rf_gaussian_moments(W_se))
        # online efficacy from the reward learner: the net's own training-time
        # decisions (pooled argmax) vs the reward target, + the baseline R̄.
        _tr = getattr(getattr(model, "_runner", None), "_trainer", None)
        # Spiking readout, when enabled: its own online accuracy over the window
        # since the last checkpoint, reported ALONGSIDE the delta readout's so the
        # two curves can be read against each other.
        _ro = getattr(_tr, "readout", None)
        if _ro is not None:
            rec.update(_ro.pop_stats())
        _rl = getattr(_tr, "reward_learner", None)
        if _rl is not None:
            _os = _rl.pop_online_stats()
            rec["online_acc"] = _os["online_acc"]
            rec["baseline"] = _os["baseline"]
            if getattr(_rl, "readout_lr", 0.0) > 0.0:
                rec["readout_learned_acc"] = float((_rl.readout_predict(X) == y).mean())
                # Uncertainty on the LEARNED readout — the distribution that should
                # carry any abstention claim, since the reward rule's own softmax
                # delta rule computes it internally. The pooled `perplexity` below
                # is near chance (~9 of 10) for a different and uninteresting
                # reason: with thin local RFs the features (strokes, edges) are
                # shared across digits, so every class group ends up with a similar
                # MEAN rate — the least-active group fires at ~32% of the most
                # active one. Uniform pooling averages away the pattern of WHICH
                # neurons fired, which is where the class information actually is.
                # Measured on the same features: pooled perplexity 9.06 vs 1.18
                # here. Subtracting the additive baseline only moves the pooled
                # figure to 7.17, so it is not a normalization artifact.
                try:
                    sys.path.insert(0, os.path.dirname(__file__))
                    from uncertainty import learned_readout_scores, softmax_probs
                    _sc = learned_readout_scores(X, _rl)
                    if _sc is not None:
                        _p = softmax_probs(_sc)
                        _H = -(_p * np.log(_p + 1e-12)).sum(1)
                        _srt = np.sort(_p, axis=1)[:, ::-1]
                        rec["entropy_readout"] = float(_H.mean())
                        rec["perplexity_readout"] = float(np.exp(_H).mean())
                        rec["margin_readout"] = float((_srt[:, 0] - _srt[:, 1]).mean())
                except Exception as _e:
                    print(f"  [readout-uncertainty] skipped: {_e}", flush=True)
        if neuron_class is not None:
            rec["pool_acc"] = pool_by_label(X, y, neuron_class)
            rec["dead_frac"], rec["frac_ever_winner"], rec["winner_entropy"] = coverage_stats(X)
        if group_assignment is not None:
            sm_acc, _ = softmax_readout(X, y, group_assignment, n_groups=a.n_groups)
            rec["softmax_acc"] = sm_acc
            # Default distribution-aware readout metrics (share_ce / perplexity /
            # margin / brier). These replace the old softmax `ce_loss`, which was
            # pinned at ln(10)=2.303 at EVERY checkpoint: spikes_per_item returns
            # mean rates ~0.01, and a T=1 softmax over near-equal tiny numbers is
            # uniform, so it could not tell chance from perfect classification.
            # The share-based version is scale-invariant and needs no temperature.
            rec.update(spike_share_metrics(X, y, group_assignment, n_groups=a.n_groups))
            # within-group RF redundancy: ~1 = neurons in a group learned the same
            # thing (consensus collapse), ~0 = diverse/orthogonal.
            _per, rec["rf_diversity"] = group_rf_diversity(W_se, group_assignment, n_groups=a.n_groups)
        # per-class confusion matrices on the held-out te split: linear classifier
        # (all runs) and the reward/group readout (when a class assignment exists),
        # so we can watch which classes get discriminated / confused over training.
        yte = y[te]
        rec["cm_linear"] = confusion_matrix(
            yte, clf.predict(np.nan_to_num(sc.transform(X[te]))), 10).tolist()
        if group_assignment is not None:
            rec["cm_readout"] = confusion_matrix(
                yte, softmax_readout_pred(X[te], group_assignment, a.n_groups), 10).tolist()
        elif neuron_class is not None:
            rec["cm_readout"] = confusion_matrix(
                yte, pool_by_label_pred(X[te], neuron_class, 10), 10).tolist()
        last_val["X"], last_val["y"] = X, y
        traj.append(rec)
        extra = ""
        if "pool_acc" in rec:
            extra += f" pool {rec['pool_acc']:.3f} dead {rec['dead_frac']:.2f} win_ent {rec['winner_entropy']:.2f}"
        if "softmax_acc" in rec:
            # `perp` reports the LEARNED readout when there is one; the pooled figure
            # sits near chance for reasons unrelated to representation quality (see
            # the note where perplexity_readout is computed) and is kept in the JSON
            # under `perplexity` rather than shown here.
            _pp = rec.get("perplexity_readout", rec.get("perplexity"))
            _src = "perp" if "perplexity_readout" in rec else "perp(pool)"
            extra += (f" softmax {rec['softmax_acc']:.3f} ce {rec['share_ce']:.3f}"
                      f" {_src} {_pp:.2f} margin {rec['margin']:+.3f}")
        print(f"  [{a.tag}] b{batch:>3} val {rec['val_acc']:.3f} eta2 {rec['eta2']:.3f} "
              f"corr {rec['corr_within']:.3f} PR {rec['pr']:.1f}/{rec['n_active']} "
              f"coh {rec['orient_coh']:.3f} refit {refit:.3f} fixed {fixed_acc:.3f}"
              f"{extra}", flush=True)
        if not a.no_plots:
            key = "first" if "first" not in rf_saved else "last"
            save_rf_grid(W_se, os.path.join(a.output_dir, "weights", f"rf_{key}.png"))
            rf_saved[key] = batch
            # per-class RF grid (overwritten each checkpoint) to see whether a group's
            # neurons learn diverse concepts or collapse to one consensus template.
            if group_assignment is not None:
                plot_group_rfs(W_se, group_assignment,
                               os.path.join(a.output_dir, "weights", "group_rfs.png"),
                               n_groups=a.n_groups)
        # always persist the metrics time series (accuracy, ce_loss, val_phi, per-class
        # recall) to results.json for later plotting; only regenerate confusion.png when
        # plotting is enabled.
        try:
            partial = dict(config=cfg, trajectory=traj)
            with open(os.path.join(a.output_dir, "results.json"), "w") as f:
                json.dump(partial, f, indent=2)
            if not a.no_plots:
                sys.path.insert(0, os.path.dirname(__file__))
                # Diagnostic figures live in stats/ alongside stats.png rather than
                # scattered across the run root, which holds only the run's outputs
                # proper (results.json, config.json, weights/, spikes/).
                _stats = os.path.join(a.output_dir, "stats")
                os.makedirs(_stats, exist_ok=True)
                from plot_confusion import make_confusion_plot
                make_confusion_plot(partial, _stats)
                # metrics.png: the response-space dashboard. Kept separate from the
                # library's stats.png, which plots WEIGHT-space diagnostics
                # (rf_participation_ratio / rf_mean_cosine / rf_gini) that are
                # confounded by RF size and so cannot answer whether the
                # representation improved.
                from plot_metrics import make_metrics_plot
                for _t in traj:
                    if _t.get("refit_acc") is not None and _t.get("fixed_acc") is not None:
                        _t["_drift"] = _t["refit_acc"] - _t["fixed_acc"]
                make_metrics_plot(partial, _stats)
        except Exception as e:
            print(f"  [live-plot] skipped: {e}", flush=True)

    train_kwargs = dict(
        layers=[layer], learner=learner, regularizer=reg, epochs=a.epochs,
        train_weights=train_weights, save_model=False, accuracy_method="pca_lr",
        use_LR=True, use_phi=True, use_pca=False, track_stats=a.track_stats,
        stat_tracking_frequency=a.plot_every,  # cadence of live spike plots + stat snapshots
        heatmap_plot=a.live_plot,  # live class-tiled spike plot during training (grouped/tiled)
        output_dir=a.output_dir,   # unify: config.json + stats/ land alongside results.json + weights/
    )
    if inh_learner is not None:
        train_kwargs["inh_learner"] = inh_learner

    train_gen = model.train(**train_kwargs)
    runner = model._runner

    if a.plot_single_neuron or a.plot_rfs or a.plot_schematic:
        from neurosnn._plot.weights import save_init_weight_plots
        weight_type = "oriented_rf" if a.prior == "oriented" else "rf"
        plot_dir = os.path.join(a.output_dir, "weights")
        save_init_weight_plots(
            runner.model,
            plot_dir,
            neuron_id=a.neuron_id,
            n_orientations=4,
            orientation_mode="block",
            weight_type=weight_type,
            plot_single_neuron=a.plot_single_neuron,
            plot_rfs=a.plot_rfs,
        )
        if a.plot_rfs and a.prior != "oriented":
            # save_init_weight_plots only does RF summary for oriented_rf;
            # for isotropic/random use save_rf_grid directly
            save_rf_grid(runner.model.weights[:runner.model.st, runner.model.st:runner.model.ex],
                         os.path.join(plot_dir, "rf_grid_init.png"), n=64)
        if a.plot_rfs and group_assignment is not None:
            # per-class input-space coverage: verify each group tiles the full input
            from neurosnn._plot.weights import plot_group_coverage
            plot_group_coverage(
                runner.model.weights[:runner.model.st, runner.model.st:runner.model.ex],
                group_assignment,
                os.path.join(plot_dir, "group_coverage_init.png"),
                n_groups=a.n_groups,
            )
        if a.plot_schematic and group_assignment is not None:
            # force-directed graph from real weights: WTA groups + W_se input + readout
            from neurosnn._plot.network import plot_network_graph
            m = runner.model
            plot_network_graph(
                m.weights, group_assignment, m.st, m.ex, m.ih,
                os.path.join(plot_dir, "network_graph.png"),
                n_groups=a.n_groups,
            )

    last_w = None
    gstep = 0  # monotonic batch index across epochs (r.batch resets each epoch)
    for r in train_gen:
        last_w = r.weights if r.weights is not None else last_w
        if r.accuracy is not None and last_w is not None:
            if gstep % a.val_every == 0:
                checkpoint(gstep, last_w)
            gstep += 1

    if not traj and last_w is not None:
        checkpoint(0, last_w)

    CAP["rows"].clear()
    _tr = getattr(getattr(model, "_runner", None), "_trainer", None)
    _ro = getattr(_tr, "readout", None)
    if _ro is not None:
        # the buffers accumulate across train/val/test; clearing here leaves exactly
        # the test items behind, decoded from the readout's own spikes
        _tr.readout_preds.clear(); _tr.readout_labels.clear()
    test = model.test()
    out = dict(config=cfg, test_acc=float(test.accuracy) if test.accuracy is not None else float("nan"),
               test_phi=float(test.phi) if test.phi is not None else float("nan"), trajectory=traj)
    if _ro is not None and _tr.readout_preds:
        _p = np.asarray(_tr.readout_preds); _y = np.asarray(_tr.readout_labels)
        out["test_spiking_acc"] = float((_p == _y).mean())
        out["test_spiking_n"] = int(_p.size)
        out["test_spiking_cm"] = confusion_matrix(_y, _p, 10).tolist()
        print(f"  [spiking readout] TEST acc={out['test_spiking_acc']:.4f} "
              f"on {_p.size} items", flush=True)
    _rlw = getattr(_tr, "reward_learner", None)

    # --- weight checkpoint. Runs BEFORE the OOD passes so the saved state is the
    # end-of-training one, and so a crash in an OOD load still leaves the model on disk.
    if a.save_weights and last_w is not None:
        _ckpt = save_weight_checkpoint(
            os.path.join(a.output_dir, "weights", "checkpoint.npz"),
            last_w, st, ex, _rlw, group_assignment if group_assignment is not None else neuron_class, cfg)
        print(f"  [checkpoint] saved -> {_ckpt}", flush=True)

    # --- out-of-distribution probes. The trained network sees a dataset it was never
    # trained on; we keep only its exc-rate features, because the question is whether
    # its own confidence collapses far enough to abstain (scored later in ood.py).
    # Weights are hashed either side: featurize is read-only by construction, and this
    # asserts it rather than trusting it.
    ood_feats = {}
    if a.ood_dataset:
        _w_before = hash(last_w.tobytes()) if last_w is not None else None
        _ro_before = (hash(_rlw.W_dense.tobytes())
                      if _rlw is not None and getattr(_rlw, "dense_readout", False) else None)
        for _ds in a.ood_dataset:
            if _ds == a.dataset:
                print(f"  [ood] skipping {_ds}: it is the in-distribution dataset", flush=True)
                continue
            try:
                _Xo = featurize_ood(model, _ds, a.ood_all, batch=1000)
            except Exception as e:
                print(f"  [ood] {_ds} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            if _Xo is None or not _Xo.size:
                print(f"  [ood] {_ds}: featurize returned nothing", flush=True)
                continue
            ood_feats[_ds] = _Xo.astype(np.float32)
            # A network that goes SILENT on OOD input makes every shape statistic
            # vacuous (uniform p by fiat), so report the silent fraction here rather
            # than discovering it inside the AUROC later.
            _sil = float((_Xo.sum(1) <= 1e-12).mean())
            print(f"  [ood] {_ds}: n={_Xo.shape[0]} mean_rate={_Xo.mean():.4f} "
                  f"silent_frac={_sil:.4f}", flush=True)
            out.setdefault("ood", {})[_ds] = dict(n=int(_Xo.shape[0]),
                                                  mean_rate=float(_Xo.mean()),
                                                  silent_frac=_sil)
        _w_after = hash(last_w.tobytes()) if last_w is not None else None
        _ro_after = (hash(_rlw.W_dense.tobytes())
                     if _rlw is not None and getattr(_rlw, "dense_readout", False) else None)
        assert _w_before == _w_after, "OOD featurize mutated the network weights"
        assert _ro_before == _ro_after, "OOD featurize mutated the readout weights"

    # final test-set confusion matrices: fit the linear readout on the last val
    # features, evaluate on captured test features; readout needs no fit.
    if CAP["rows"] and last_val["X"] is not None:
        Xt = np.concatenate([x for x, _ in CAP["rows"]], 0)
        yt = np.concatenate([yy for _, yy in CAP["rows"]], 0).astype(int)
        # Fit the linear probe on a large TRAIN-feature slice when requested, so
        # test_lin_acc reflects the representation rather than a starved ~1k fit
        # (07-25). Calibration (probe_cal) and drift still use val; only the
        # classifier's FIT set changes. Falls back to val if featurization yields
        # nothing. CAP is untouched by featurize, so Xt (test features) is intact.
        probe_X, probe_y = last_val["X"], last_val["y"]
        if a.probe_fit_all and a.probe_fit_all > 0:
            _Xp, _yp = model.featurize(a.probe_fit_all, 1000, partition="train")
            if _Xp is not None and _Xp.size:
                probe_X, probe_y = _Xp, _yp
                print(f"  [probe] fit on {len(_yp)} train features "
                      f"(--probe-fit-all {a.probe_fit_all})", flush=True)
            else:
                print("  [probe] featurize returned nothing; fell back to val", flush=True)
        sc, clf = fit_clf(probe_X, probe_y, a.seed)
        lin_pred = clf.predict(np.nan_to_num(sc.transform(Xt)))
        out["test_lin_acc"] = float((lin_pred == yt).mean())
        out["test_cm_linear"] = confusion_matrix(yt, lin_pred, 10).tolist()
        if group_assignment is not None:
            out["test_cm_readout"] = confusion_matrix(
                yt, softmax_readout_pred(Xt, group_assignment, a.n_groups), 10).tolist()
        elif neuron_class is not None:
            out["test_cm_readout"] = confusion_matrix(
                yt, pool_by_label_pred(Xt, neuron_class, 10), 10).tolist()
        # --- selective-prediction go/no-go: does uncertainty rank this run's errors?
        # Calibrate on val, evaluate on test (never on train — the net is atypically
        # confident there). Cheap, so it runs unconditionally; see uncertainty.py.
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from uncertainty import run_from_features, format_report, learned_readout_scores
            assign = group_assignment if group_assignment is not None else neuron_class
            probe_cal = clf.predict_proba(np.nan_to_num(sc.transform(last_val["X"])))
            probe_test = clf.predict_proba(np.nan_to_num(sc.transform(Xt)))
            # the reward rule's own plastic readout, when the run trained one. Its
            # weights can be negative (evidence AGAINST a class), which uniform
            # pooling cannot express — so it gets scored as its own readout.
            _rlu = getattr(getattr(getattr(model, "_runner", None), "_trainer", None),
                           "reward_learner", None)
            score_cal = learned_readout_scores(last_val["X"], _rlu)
            score_test = learned_readout_scores(Xt, _rlu)
            reports = run_from_features(
                last_val["X"], last_val["y"], Xt, yt,
                assignment=assign, n_groups=a.n_groups,
                probe_cal=probe_cal, probe_test=probe_test, probe_pred=lin_pred,
                score_cal=score_cal, score_test=score_test)
            out["uncertainty"] = reports
            for r in reports:
                print(format_report(r), flush=True)
            # dump the raw features so statistics can be re-scored without retraining
            np.savez_compressed(
                os.path.join(a.output_dir, "uncertainty_features.npz"),
                X_cal=last_val["X"].astype(np.float32), y_cal=last_val["y"].astype(np.int16),
                X_test=Xt.astype(np.float32), y_test=yt.astype(np.int16),
                probe_cal=probe_cal.astype(np.float32), probe_test=probe_test.astype(np.float32),
                **({"assignment": np.asarray(assign)} if assign is not None else {}),
                **({"score_cal": score_cal.astype(np.float32),
                    "score_test": score_test.astype(np.float32)}
                   if score_cal is not None else {}),
                **{f"X_ood_{k}": v for k, v in ood_feats.items()})
            # full 0->100% risk-coverage figures + the CSV table view. Runs off the
            # features just written, so it is reproducible standalone later via
            # plot_risk_coverage.py --run <output_dir>.
            try:
                from plot_risk_coverage import make_risk_coverage_plots
                make_risk_coverage_plots(a.output_dir)
            except Exception as e:
                print(f"  [risk-coverage] skipped: {e}", flush=True)
        except Exception as e:
            print(f"  [uncertainty] skipped: {e}", flush=True)
    with open(os.path.join(a.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{a.tag}] DONE test_acc={out['test_acc']:.3f} -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
