"""
Quick training comparison across log-normal configs.
2 epochs on MNIST (10k train, 2k val) — enough to see early learning differences.

Structure of the sweep:
  Part 1 — fix ee_std=0, sweep se_std to find the W_se sweet spot
  Part 2 — fix se_std at best, sweep ee_std to find the W_ee sweet spot
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import neurosnn as snn


def run_config(label, lognorm_se_std=0.0, lognorm_ee_std=0.0,
               lognorm_se_mean=3.0, lognorm_ee_mean=1.0, seed=42):

    weights = snn.weights.oriented_receptive_fields(
        density_se=0.05, density_ee=0.02, density_ei=0.03, density_ie=0.05,
        peak_se=1.0, peak_ee=0.5, peak_ei=1.0, peak_ie=-0.7,
        sigma_x=lognorm_se_mean,
        sigma_x_lognormal_std=lognorm_se_std,
        sigma_ee_mean=lognorm_ee_mean,
        sigma_ee_lognormal_std=lognorm_ee_std,
    )

    layer = snn.Layer(
        N_exc=1024, N_inh=225,
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
            membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
            resting_potential=-70.0, reset_potential=-80.0,
            spike_threshold=-55.0, min_mp=-100.0, max_mp=40.0,
            mean_noise=0.0, var_noise=0.0,
            spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5,
        ),
        weights=weights,
    )

    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004, tau_trace=20, w_max=10.0, mu_weight=0.6,
        update_freq=100, clip_weights=True,
        min_weight_exc=0.01, max_weight_exc=25.0,
        min_weight_inh=-25.0, max_weight_inh=-0.01,
    )

    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    model = snn.Model(
        input_size=784, classes=list(range(10)), random_state=seed,
        num_steps=350,
        all_images_train=10000, batch_image_train=1000,
        all_images_val=2000,  batch_image_val=2000,
        all_images_test=2000, batch_image_test=2000,
        image_dataset="mnist", max_rate_hz=90.0, gain=1.0,
    )

    t0 = time.time()
    train_accs, train_phis = [], []

    for result in model.train(
        layers=[layer], learner=learner, regularizer=regularizer,
        epochs=2, train_weights=True, save_model=False,
        accuracy_method="pca_lr", use_LR=True, use_phi=True, use_pca=True,
        pca_variance=15, stat_tracking_frequency=999999,
    ):
        if result.accuracy is not None:
            train_accs.append(result.accuracy)
            train_phis.append(result.phi if result.phi else float("nan"))

    val = model.validate()
    val_acc = val.accuracy if val.accuracy is not None else float("nan")
    val_phi = val.phi   if val.phi   is not None else float("nan")
    elapsed = time.time() - t0

    final_train = float(np.nanmean(train_accs[-3:])) if train_accs else float("nan")
    final_phi   = float(np.nanmean(train_phis[-3:])) if train_phis else float("nan")
    print(f"  {label:<38}  train={final_train:.3f}  val={val_acc:.3f}  phi={val_phi:.3f}  ({elapsed:.0f}s)")
    sys.stdout.flush()
    return final_train, val_acc, val_phi


print("\n--- Part 1: sweep se_std (ee_std fixed at 0) ---")
print(f"  {'Config':<38}  train    val      phi     time")
print("  " + "-" * 70)

se_sweep = [
    ("baseline: se=0.0, ee=0.0",          0.0, 0.0),
    ("se_std=0.5, ee=0.0",                 0.5, 0.0),
    ("se_std=1.0, ee=0.0",                 1.0, 0.0),
    ("se_std=1.5, ee=0.0",                 1.5, 0.0),
    ("se_std=2.0, ee=0.0",                 2.0, 0.0),
    ("se_std=2.5, ee=0.0",                 2.5, 0.0),
]

for label, se_std, ee_std in se_sweep:
    run_config(label, lognorm_se_std=se_std, lognorm_ee_std=ee_std)

print("\n--- Part 2: sweep ee_std (se_std fixed at 1.5) ---")
print(f"  {'Config':<38}  train    val      phi     time")
print("  " + "-" * 70)

ee_sweep = [
    ("se=1.5, ee_std=0.0 (repeat)",        1.5, 0.0),
    ("se=1.5, ee_std=0.3",                 1.5, 0.3),
    ("se=1.5, ee_std=0.5",                 1.5, 0.5),
    ("se=1.5, ee_std=0.8",                 1.5, 0.8),
    ("se=1.5, ee_std=1.2",                 1.5, 1.2),
]

for label, se_std, ee_std in ee_sweep:
    run_config(label, lognorm_se_std=se_std, lognorm_ee_std=ee_std)

print("\nDone.")
