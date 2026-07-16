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
    pool_by_label, coverage_stats, softmax_readout,
    pool_by_label_pred, softmax_readout_pred, confusion_matrix, group_rf_diversity,
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


def load_input_rate():
    from torchvision import datasets
    ds = datasets.MNIST(root=os.path.join(REPO, "data", "torchvision"), train=False, download=False)
    m = ds.data.numpy().reshape(len(ds), -1).mean(0).astype(np.float64) / 255.0
    return m  # (784,) mean pixel intensity ~ mean input rate


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
    p.add_argument("--sigma-se-lognormal", type=float, default=0.0,
                   help="lognormal spread of per-neuron RF sigma (0 = uniform). >0 gives HETEROGENEOUS RF "
                        "sizes: a mix of small local-feature and larger holistic detectors.")
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
    p.add_argument("--plot-rfs", action="store_true", help="save RF grid and (oriented) summary/coverage plots after init")
    p.add_argument("--plot-single-neuron", action="store_true", help="save 2x2 SE/EE/EI/IE panel for one neuron after init")
    p.add_argument("--plot-schematic", action="store_true", help="save force-directed network graph from real weights (grouped only)")
    p.add_argument("--neuron-id", type=int, default=512, help="neuron index for --plot-single-neuron (default 512)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--test-all", type=int, default=3000)
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
        weights = snn.weights.grouped_excitatory(
            n_groups=a.n_groups, group_layout=a.group_layout, tiled=a.tiled,
            oriented=(a.prior == "oriented"),
            n_orientations=4, orientation_mode="block", **gkw)
    else:
        weights = snn.weights.random(**wkw)
    # RF-size override: shrink the structural footprint (sigma_se_mean) to force local
    # patch detectors instead of near-global whole-digit templates, and/or make RF sizes
    # heterogeneous. Applied to the spec so it flows through _to_factory_kwargs.
    if a.sigma_se > 0.0 and hasattr(weights, "sigma_se_mean"):
        weights.sigma_se_mean = a.sigma_se
    if a.sigma_se_lognormal > 0.0 and hasattr(weights, "sigma_se_lognormal_std"):
        weights.sigma_se_lognormal_std = a.sigma_se_lognormal

    layer = snn.Layer(N_exc=N_exc, N_inh=N_inh, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5), weights=weights)

    if a.rule == "reward":
        learner = snn.learner.RewardSTDP(learning_rate=a.reward_lr,
            class_assignment=("block" if a.tiled else "mod"), seed=a.seed,
            shuffle_labels=a.shuffle_labels, readout_lr=a.readout_lr,
            dense_readout=a.dense_readout)
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
        batch_image_val=a.val_all, all_images_test=a.test_all, batch_image_test=a.test_all,
        image_dataset="mnist", max_rate_hz=90.0, gain=1.0, gabor=False)

    input_rate = load_input_rate()
    cfg = dict(tag=a.tag, prior=a.prior, rule=a.rule, ee=a.ee, wta=True,
               grouped=a.grouped, n_groups=a.n_groups, group_layout=a.group_layout,
               use_vogels=a.use_vogels, vogels_lr=a.vogels_lr, vogels_rho0=a.vogels_rho0,
               n_exc=N_exc, n_inh=N_inh,
               sigma_se=a.sigma_se, sigma_se_lognormal=a.sigma_se_lognormal,
               readout_lr=a.readout_lr, dense_readout=a.dense_readout,
               train_all=a.train_all, seed=a.seed)
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
        rec = dict(batch=int(batch),
                   val_acc=float(v.accuracy) if v.accuracy is not None else float("nan"),
                   val_phi=float(v.phi) if v.phi is not None else float("nan"),
                   selectivity=class_selectivity(X, y),
                   orient_coh=orientation_coherence(W_se),
                   refit_acc=float(refit), fixed_acc=float(fixed_acc),
                   cur_se=se, cur_ee=ee, cur_ie=ie, ee_se_ratio=ee / (se + 1e-12),
                   w_floor_frac=w_floor_frac(W_se))
        # online efficacy from the reward learner: the net's own training-time
        # decisions (pooled argmax) vs the reward target, + the baseline R̄.
        _rl = getattr(getattr(getattr(model, "_runner", None), "_trainer", None), "reward_learner", None)
        if _rl is not None:
            _os = _rl.pop_online_stats()
            rec["online_acc"] = _os["online_acc"]
            rec["baseline"] = _os["baseline"]
            if getattr(_rl, "readout_lr", 0.0) > 0.0:
                rec["readout_learned_acc"] = float((_rl.readout_predict(X) == y).mean())
        if neuron_class is not None:
            rec["pool_acc"] = pool_by_label(X, y, neuron_class)
            rec["dead_frac"], rec["frac_ever_winner"], rec["winner_entropy"] = coverage_stats(X)
        if group_assignment is not None:
            sm_acc, ce = softmax_readout(X, y, group_assignment, n_groups=a.n_groups)
            rec["softmax_acc"] = sm_acc
            rec["ce_loss"] = ce
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
            extra += f" softmax {rec['softmax_acc']:.3f} ce {rec['ce_loss']:.3f}"
        print(f"  [{a.tag}] b{batch:>3} val {rec['val_acc']:.3f} sel {rec['selectivity']:.3f} "
              f"coh {rec['orient_coh']:.3f} refit {refit:.3f} fixed {fixed_acc:.3f} "
              f"EE/SE {rec['ee_se_ratio']:.3f}{extra}", flush=True)
        key = "first" if "first" not in rf_saved else "last"
        save_rf_grid(W_se, os.path.join(a.output_dir, "weights", f"rf_{key}.png"))
        rf_saved[key] = batch
        # per-class RF grid (overwritten each checkpoint) to see whether a group's
        # neurons learn diverse concepts or collapse to one consensus template.
        if group_assignment is not None:
            plot_group_rfs(W_se, group_assignment,
                           os.path.join(a.output_dir, "weights", "group_rfs.png"),
                           n_groups=a.n_groups)
        # live progress: write partial results.json + regenerate confusion.png each
        # checkpoint so readout accuracy / grouped clustering / per-class recall
        # update as the run goes (test matrices fill in at the end).
        try:
            partial = dict(config=cfg, trajectory=traj)
            with open(os.path.join(a.output_dir, "results.json"), "w") as f:
                json.dump(partial, f, indent=2)
            sys.path.insert(0, os.path.dirname(__file__))
            from plot_confusion import make_confusion_plot
            make_confusion_plot(partial, a.output_dir)
        except Exception as e:
            print(f"  [live-plot] skipped: {e}", flush=True)

    train_kwargs = dict(
        layers=[layer], learner=learner, regularizer=reg, epochs=1,
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
    for r in train_gen:
        last_w = r.weights if r.weights is not None else last_w
        if r.accuracy is not None and r.batch % a.val_every == 0 and last_w is not None:
            checkpoint(r.batch, last_w)

    if not traj and last_w is not None:
        checkpoint(0, last_w)

    CAP["rows"].clear()
    test = model.test()
    out = dict(config=cfg, test_acc=float(test.accuracy) if test.accuracy is not None else float("nan"),
               test_phi=float(test.phi) if test.phi is not None else float("nan"), trajectory=traj)
    # final test-set confusion matrices: fit the linear readout on the last val
    # features, evaluate on captured test features; readout needs no fit.
    if CAP["rows"] and last_val["X"] is not None:
        Xt = np.concatenate([x for x, _ in CAP["rows"]], 0)
        yt = np.concatenate([yy for _, yy in CAP["rows"]], 0).astype(int)
        sc, clf = fit_clf(last_val["X"], last_val["y"], a.seed)
        lin_pred = clf.predict(np.nan_to_num(sc.transform(Xt)))
        out["test_lin_acc"] = float((lin_pred == yt).mean())
        out["test_cm_linear"] = confusion_matrix(yt, lin_pred, 10).tolist()
        if group_assignment is not None:
            out["test_cm_readout"] = confusion_matrix(
                yt, softmax_readout_pred(Xt, group_assignment, a.n_groups), 10).tolist()
        elif neuron_class is not None:
            out["test_cm_readout"] = confusion_matrix(
                yt, pool_by_label_pred(Xt, neuron_class, 10), 10).tolist()
    with open(os.path.join(a.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{a.tag}] DONE test_acc={out['test_acc']:.3f} -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
