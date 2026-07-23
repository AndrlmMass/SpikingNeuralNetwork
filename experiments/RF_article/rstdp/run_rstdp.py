"""
run_rstdp.py — one clean entry point for the SUPERVISED reward-STDP model (tiled).

A deliberately small, readable driver for tuning the supervised reward-modulated
STDP model. It fixes the architecture to the one we care about and exposes only
the knobs worth tuning. No frozen/trace/triplet branches, no plotting. Just:

    build the tiled reward-STDP network -> train on N samples -> print metrics.

>>> To tune: edit the CONFIG block right below, then run:
        python experiments/RF_article/rstdp/run_rstdp.py

Architecture (fixed here; see PLAN.md for the rationale):
  - Tiled per-class excitatory layout: N_exc neurons split into 10 class groups;
    each group's receptive-field centers tile the FULL input (k x k grid/class).
  - Oriented (Gabor-style) receptive fields at each tile center.
  - Only feedforward input->exc (SE) weights learn, via reward-STDP. Recurrent
    (EE=0) and inhibition (I->E, fixed WTA scaffold) are frozen.
  - Readout = pool exc firing rate by class group, argmax (`pool_acc`).

Reward-STDP in one line: per sample, accumulate a count-product eligibility
(#pre x #post spikes), then at the sample boundary nudge each synapse by
lr * (R_i - baseline) * eligibility, with R_i = +1 if the post neuron's class
== the label else -1. Correct-class coincidences potentiate, wrong-class depress.
"""

# ==========================================================================  #
#  CONFIG — edit these, then run the file. Nothing else needs changing.       #
# ==========================================================================  #

# ---- the knobs you actually tune ----
REWARD_LR         = 2e-5   # reward-STDP step size (the main dial)
BASELINE_DECAY    = 0.01   # EMA rate for the reward baseline (0 = no centering)
NUMBER_OF_SAMPLES = 3000#3000   # training images
VAL_SAMPLES       = 1000 #1000   # images per validation checkpoint
VALIDATE_EVERY    = 1      # validate every N training batches (1 batch = 1000 images)
TEST_SAMPLES      = 1000   # images for the final test
SEED              = 0

# ---- architecture (change with care; see docstring) ----
N_GROUPS   = 10      # one group per MNIST class
N_EXC      = 1000    # excitatory neurons; must be N_GROUPS * k^2 (per-class k x k tile)
                     #   1000 = 10*10^2 | 1440 = 10*12^2 | 2560 = 10*16^2 ...
ORIENTED   = True    # True = oriented (Gabor-style) RFs; False = isotropic Gaussian
# RF shape, in pixels (which ones apply depends on ORIENTED):
ISOTROPIC_SIZE     = 2.0#3.0   # ORIENTED=False: Gaussian RF radius (sigma) — bigger = wider reach
ORIENTED_LENGTH    = 3.0#3.0   # ORIENTED=True : sigma ALONG the bar (major axis) = length
ORIENTED_THICKNESS = 1.2   # ORIENTED=True : sigma ACROSS the bar (minor axis) = thickness
# Center concentration: MNIST digits are centered, so the outer RF tiles land on
# always-blank border pixels and those neurons die. Concentrate the tiled RF
# centers on the central region instead of the full 28x28.
CONCENTRATE_CENTERS = True   # True = tile centers over the central region (skip blank border)
CENTER_MARGIN       = 4      # px trimmed per edge when concentrating (0..13; 4 ~ MNIST digit box)
# WTA competition: intra-class I->E inhibition makes the 100 neurons of a class
# compete, so they specialize to different modes. False ablates it (no competition
# -> neurons in a class tend to converge on the same average). Mostly a diagnostic.
WTA_INHIBITION = False
NUM_STEPS  = 350     # simulation timesteps per image
MAX_RATE_HZ = 90.0   # peak Poisson input rate
NORMALIZE_FREQ = 1050  # SE weight-normalization cadence (timesteps)

# ---- receptive-field evolution plots ----
PLOT_RF_EVOLUTION = True   # save per-class RF heatmap grids at 0/25/50/75/100% of training
RF_COLORMAP       = "OrRd"  # colormap for the RF "what it sees" heatmaps

# ==========================================================================  #
#  Implementation below — you shouldn't need to touch it to tune.             #
# ==========================================================================  #
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib
matplotlib.use("Agg")  # never pop a window
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import neurosnn as snn
from neurosnn._evaluation import evaluation as evalmod
from neurosnn._evaluation.analysis import (
    class_selectivity, w_floor_frac, pool_by_label, coverage_stats, softmax_readout,
)
from neurosnn._network.init_weights import make_group_assignment


# --------------------------------------------------------------------------- #
# Feature capture                                                             #
# --------------------------------------------------------------------------- #
# The evaluator computes per-item excitatory firing-rate features internally
# but does not return them. We wrap Evaluator.score to grab (X, y) on each call
# so we can compute representation metrics (selectivity, pooled readout,
# coverage) from the exact features the accuracy is measured on.
_CAPTURED: list = []
_orig_score = evalmod.Evaluator.score


def _capturing_score(self, X, Y):
    _CAPTURED.append((np.asarray(X).copy(), np.asarray(Y).copy()))
    return _orig_score(self, X, Y)


evalmod.Evaluator.score = _capturing_score


def drain_features():
    """Return (X, y) accumulated since the last drain, then clear. (None, None) if empty."""
    if not _CAPTURED:
        return None, None
    X = np.concatenate([x for x, _ in _CAPTURED], 0)
    y = np.concatenate([yy for _, yy in _CAPTURED], 0).astype(int)
    _CAPTURED.clear()
    return X, y


# --------------------------------------------------------------------------- #
# Build the network                                                           #
# --------------------------------------------------------------------------- #
def tile_k():
    """Per-class tile side k (validates N_EXC = N_GROUPS * k^2)."""
    per_group = N_EXC // N_GROUPS
    k = int(round(per_group ** 0.5))
    if N_EXC % N_GROUPS != 0 or k * k != per_group:
        raise ValueError(
            f"N_EXC ({N_EXC}) must equal N_GROUPS ({N_GROUPS}) * k^2 (a perfect "
            f"square per class). Try {N_GROUPS}*10^2=1000, {N_GROUPS}*12^2=1440, "
            f"{N_GROUPS}*16^2=2560, ..."
        )
    return k


def train_batch_size():
    """Images per training batch. Set to a quarter of the run so the batch
    boundaries fall at 25/50/75/100% — the marks we snapshot RFs at."""
    return max(1, NUMBER_OF_SAMPLES // 4)


def build():
    """Return (model, layer, learner, regularizer, group_assignment, (st, ex, ih))."""
    N_inh = N_EXC  # one-to-one WTA: N_inh == N_exc
    st, ex, ih = 784, 784 + N_EXC, 784 + N_EXC + N_inh

    # RF shape -> builder params. Oriented: sigma_x = length (major axis),
    # sigma_y = gamma*sigma_x = thickness (minor axis), so gamma = thickness/length.
    # Isotropic: grouped_excitatory feeds sigma_x in as the Gaussian radius (gamma unused).
    if ORIENTED:
        sigma_x = ORIENTED_LENGTH
        gamma = ORIENTED_THICKNESS / ORIENTED_LENGTH
    else:
        sigma_x = ISOTROPIC_SIZE
        gamma = 0.4  # ignored when isotropic

    center_margin = CENTER_MARGIN if CONCENTRATE_CENTERS else 0.0  # 0 = full-input tiling

    # Tiled grouped-excitatory weights: per-class RF centers tiling the input,
    # feedforward only (EE=0), 1:1 WTA + intra-group inhibition scaffold.
    # ablate_ie zeros the I->E block -> no WTA competition when WTA_INHIBITION is off.
    weights = snn.weights.grouped_excitatory(
        n_groups=N_GROUPS, group_layout="block", tiled=True,
        oriented=ORIENTED, n_orientations=4, orientation_mode="block",
        sigma_x=sigma_x, gamma=gamma, tiled_center_margin=center_margin,
        density_se=0.01, density_ee=0.0, density_ei=0.03, density_ie=0.05,
        peak_se=4.0, peak_ee=1.0, peak_ei=20.0, peak_ie=-2.0,
        ablate_ie=not WTA_INHIBITION,
    )

    layer = snn.Layer(N_exc=N_EXC, N_inh=N_inh, weights=weights, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5))

    # Reward-STDP on SE only. "block" class assignment matches the tiled block layout.
    learner = snn.learner.RewardSTDP(
        learning_rate=REWARD_LR, baseline_decay=BASELINE_DECAY,
        class_assignment="block", seed=SEED)

    reg = snn.regularizer.Normalize(frequency=NORMALIZE_FREQ, mode="neuron")

    # exc neuron -> class group (block layout), used for the pooled readouts
    group_assignment = make_group_assignment(N_EXC, N_GROUPS, "block")

    model = snn.Model(
        input_size=784, classes=list(range(N_GROUPS)), random_state=SEED,
        num_steps=NUM_STEPS, image_dataset="mnist", max_rate_hz=MAX_RATE_HZ,
        gain=1.0, gabor=False,
        all_images_train=NUMBER_OF_SAMPLES, batch_image_train=train_batch_size(),
        all_images_val=VAL_SAMPLES, batch_image_val=VAL_SAMPLES,
        all_images_test=TEST_SAMPLES, batch_image_test=TEST_SAMPLES)

    return model, layer, learner, reg, group_assignment, (st, ex, ih)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def compute_metrics(X, y, W_se, group_assignment):
    """Representation + readout metrics from captured val/test features."""
    sm_acc, _ = softmax_readout(X, y, group_assignment, n_groups=N_GROUPS)
    dead_frac, frac_winner, win_ent = coverage_stats(X)
    return dict(
        pool_acc=pool_by_label(X, y, group_assignment),
        softmax_acc=sm_acc,
        selectivity=class_selectivity(X, y),
        dead_frac=dead_frac, frac_winner=frac_winner, winner_entropy=win_ent,
        w_floor=w_floor_frac(W_se),
    )


HEADER = (f"{'batch':>6} {'val_acc':>8} {'pool_acc':>9} {'softmax':>8} "
          f"{'select':>7} {'dead':>6} {'win_ent':>8} {'w_floor':>8}")


def fmt_row(batch, val_acc, m):
    return (f"{batch:>6} {val_acc:>8.3f} {m['pool_acc']:>9.3f} {m['softmax_acc']:>8.3f} "
            f"{m['selectivity']:>7.3f} {m['dead_frac']:>6.2f} {m['winner_entropy']:>8.2f} "
            f"{m['w_floor']:>8.3f}")


# --------------------------------------------------------------------------- #
# Receptive-field evolution plots                                             #
# --------------------------------------------------------------------------- #
# A neuron's receptive field ("what it sees") is its input->exc weight column
# reshaped to 28x28: bright = pixels that drive it. For each class we save a
# k x k montage of its neurons' RFs, at 0/25/50/75/100% of training, so you can
# watch reward-STDP reshape them.
def make_run_dir():
    """results/rstdp/<YYYY-MM-DD_HH-MM-SS>/ with a class_NN subdir per group."""
    run_dir = os.path.join(REPO, "results", "rstdp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    for c in range(N_GROUPS):
        os.makedirs(os.path.join(run_dir, f"class_{c:02d}"), exist_ok=True)
    return run_dir


def save_class_rf_grid(W_se, class_idx, per_group, k, out_path, pct, vmax):
    """Save one class's per-neuron RF heatmaps as a k x k montage."""
    cols = W_se[:, class_idx * per_group:(class_idx + 1) * per_group]  # (784, per_group)
    fig, axes = plt.subplots(k, k, figsize=(k, k))
    axes = np.atleast_1d(axes).ravel()
    for n in range(per_group):
        axes[n].imshow(cols[:, n].reshape(28, 28), cmap=RF_COLORMAP, vmin=0.0, vmax=vmax)
        axes[n].set_xticks([]); axes[n].set_yticks([])
    for n in range(per_group, len(axes)):
        axes[n].axis("off")
    fig.suptitle(f"class {class_idx} — {pct}% trained", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def snapshot_rfs(run_dir, W, st, ex, k, pct):
    """Save the RF montage for every class at this training percentage."""
    W_se = W[:st, st:ex]                                   # (784, N_exc)
    per_group = N_EXC // N_GROUPS
    vmax = float(np.percentile(W_se, 99.5))               # shared scale across classes
    if vmax <= 0:
        vmax = float(W_se.max()) or 1.0
    for c in range(N_GROUPS):
        path = os.path.join(run_dir, f"class_{c:02d}", f"rf_{pct:03d}pct.png")
        save_class_rf_grid(W_se, c, per_group, k, path, pct, vmax)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
def main():
    k = tile_k()
    model, layer, learner, reg, group_assignment, (st, ex, ih) = build()
    run_dir = make_run_dir()

    print(f"\nSupervised reward-STDP (tiled)  |  N_exc={N_EXC} "
          f"({N_GROUPS} classes x {k}x{k}/class)  "
          f"prior={'oriented' if ORIENTED else 'isotropic'}")
    print(f"reward_lr={REWARD_LR:g}  baseline_decay={BASELINE_DECAY:g}  "
          f"train={NUMBER_OF_SAMPLES}  val={VAL_SAMPLES}  test={TEST_SAMPLES}  seed={SEED}")
    if PLOT_RF_EVOLUTION:
        print(f"RF evolution plots -> {run_dir}")
    print("\n" + HEADER)

    def checkpoint(batch, W):
        drain_features()                      # discard anything stale
        v = model.validate()                  # re-run val; features captured via the shim
        X, y = drain_features()
        if X is None:
            return
        val_acc = float(v.accuracy) if v.accuracy is not None else float("nan")
        m = compute_metrics(X, y, W[:st, st:ex], group_assignment)
        print(fmt_row(batch, val_acc, m), flush=True)

    # build the training generator (this creates the runner + init weights eagerly,
    # before the first iteration) so we can snapshot the untrained RFs at 0%.
    train_gen = model.train(layers=[layer], learner=learner, regularizer=reg, epochs=1,
                            train_weights=True, save_model=False, accuracy_method="pca_lr",
                            use_LR=True, use_phi=True, use_pca=False, track_stats=False)
    if PLOT_RF_EVOLUTION:
        snapshot_rfs(run_dir, model._runner.model.weights, st, ex, k, 0)

    # train: one yield per batch (a quarter of the run); snapshot RFs at each
    # boundary (25/50/75/100%) and print the metric row.
    bs = train_batch_size()
    last_w = None
    seen_batches = 0
    prev_batch = object()  # sentinel: first yield is always a new batch
    for r in train_gen:
        if r.weights is not None:
            last_w = r.weights
        if r.batch != prev_batch:
            prev_batch = r.batch
            seen_batches += 1
            if PLOT_RF_EVOLUTION and last_w is not None:
                pct = round(100 * min(NUMBER_OF_SAMPLES, seen_batches * bs) / NUMBER_OF_SAMPLES)
                snapshot_rfs(run_dir, last_w, st, ex, k, pct)
        if r.accuracy is not None and r.batch % VALIDATE_EVERY == 0 and last_w is not None:
            checkpoint(r.batch, last_w)

    # final test
    drain_features()
    test = model.test()
    Xt, yt = drain_features()
    print("\n" + "=" * 60)
    print(f"FINAL TEST  ({TEST_SAMPLES} images)")
    print("=" * 60)
    test_acc = float(test.accuracy) if test.accuracy is not None else float("nan")
    print(f"  linear (PCA+LR) accuracy : {test_acc:.3f}")
    if Xt is not None and last_w is not None:
        m = compute_metrics(Xt, yt, last_w[:st, st:ex], group_assignment)
        print(f"  pooled readout accuracy  : {m['pool_acc']:.3f}   <- the supervised metric")
        print(f"  softmax readout accuracy : {m['softmax_acc']:.3f}")
        print(f"  class selectivity        : {m['selectivity']:.3f}")
        print(f"  dead neuron fraction     : {m['dead_frac']:.2f}")
        print(f"  winner entropy           : {m['winner_entropy']:.2f}")
        print(f"  SE weight-floor fraction : {m['w_floor']:.3f}")
    if PLOT_RF_EVOLUTION:
        print(f"\nRF evolution plots saved under: {run_dir}")
    print()


if __name__ == "__main__":
    main()
