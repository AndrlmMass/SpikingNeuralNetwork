"""
Architecture-effect run (single config) on ORIENTED receptive fields.

Establishes, on the RF substrate that is the paper's contribution, the effect of
(a) excitatory STDP vs frozen, (b) static vs plastic (Vogels) inhibition, and
(c) one-to-one WTA competition — with EE recurrence on/off.

Tracks the val-accuracy AND val-phi trajectory plus active-E fraction and RF Gini
per checkpoint, so we can see whether learning improves the *representation*
(phi up) even where the linear-readout accuracy is near its ceiling — decoupling
"is STDP helping" from the accuracy cap.

frozen  := no excitatory STDP and no Vogels (train_weights=False; genuinely frozen
           after the trainer train_weights fix).
exc     := ungated + mean-x_tar excitatory STDP (gate already removed in synapses).
vogels  := plastic I->E via Vogels iSTDP (inhibition plastic instead of static).
wta     := proper one-to-one WTA (N_inh forced to N_exc); strong E->I so each
           interneuron fires from its single driver.

Usage:
  python experiments/RF_article/arch_sweep/run_arch.py --tag exc --exc --output-dir <dir>
  python experiments/RF_article/arch_sweep/run_arch.py --tag wta_exc_vogels --exc --vogels --wta ...
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import neurosnn as snn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--exc", action="store_true", help="excitatory STDP on (else frozen)")
    p.add_argument("--vogels", action="store_true", help="plastic Vogels I->E inhibition")
    p.add_argument("--wta", action="store_true", help="one-to-one WTA (N_inh=N_exc)")
    p.add_argument("--ablate-ee", action="store_true", help="zero E->E recurrence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-exc", type=int, default=1024)
    p.add_argument("--n-inh", type=int, default=225, help="ignored when --wta (forced to N_exc)")
    p.add_argument("--peak-ei", type=float, default=2.0, help="E->I; auto-bumped to 20 for WTA")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--train-all", type=int, default=15000)
    p.add_argument("--train-batch", type=int, default=1000)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--test-all", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    n_inh = a.n_exc if a.wta else a.n_inh
    peak_ei = 20.0 if a.wta else a.peak_ei
    train_weights = bool(a.exc)  # excitatory STDP gates the whole learning block

    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01, density_ee=0.01, density_ei=0.03, density_ie=0.05,
        peak_se=4.0, peak_ee=1.0, peak_ei=peak_ei, peak_ie=-2.0,
        n_orientations=4, orientation_mode="block",
        wta_inhibition=a.wta, ablate_ee=a.ablate_ee)

    layer = snn.Layer(N_exc=a.n_exc, N_inh=n_inh, membrane=snn.membrane.LIF(
        tau_m_exc=20.0, tau_m_inh=15.0, tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0, membrane_resistance_inh=15.0,
        resting_potential=-70.0, reset_potential=-80.0, spike_threshold=-55.0,
        min_mp=-100.0, max_mp=40.0, mean_noise=0.0, var_noise=0.0,
        spike_adaptation=True, tau_adaptation=200.0, delta_adaptation=0.5),
        weights=weights)

    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004, tau_trace=20, w_max=10.0, mu_weight=0.5,
        x_tar_mode="mean", update_freq=100, clip_weights=True,
        min_weight_exc=0.01, max_weight_exc=25.0, min_weight_inh=-25.0, max_weight_inh=-0.01)
    inh_learner = snn.learner.VogelsSTDP(learning_rate=0.01, rho_0=0.1) if a.vogels else None
    reg = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    model = snn.Model(
        input_size=784, classes=list(range(10)), random_state=a.seed, num_steps=350,
        all_images_train=a.train_all, batch_image_train=a.train_batch,
        all_images_val=a.val_all, batch_image_val=a.val_all,
        all_images_test=a.test_all, batch_image_test=a.test_all,
        image_dataset=a.dataset, max_rate_hz=90.0, gain=1.0, gabor=False)

    cfg = dict(tag=a.tag, exc=a.exc, vogels=a.vogels, wta=a.wta, ablate_ee=a.ablate_ee,
               n_exc=a.n_exc, n_inh=n_inh, peak_ei=peak_ei, dataset=a.dataset,
               train_all=a.train_all, seed=a.seed, x_tar_mode="mean")
    print(f"\n[{a.tag}] exc={a.exc} vogels={a.vogels} wta={a.wta} ablate_ee={a.ablate_ee} "
          f"N_inh={n_inh} train_weights={train_weights}\n", flush=True)

    val_hist, stats_hist = [], []
    t0 = time.time()
    for r in model.train(layers=[layer], learner=learner, inh_learner=inh_learner,
                         regularizer=reg, epochs=a.epochs, train_weights=train_weights,
                         save_model=False, accuracy_method="pca_lr", use_LR=True,
                         use_phi=True, use_pca=False, track_stats=True,
                         stat_tracking_frequency=10500):
        if r.stats:
            stats_hist.append({"batch": r.batch, **{k: r.stats.get(k) for k in
                              ("active_frac_exc", "rf_gini", "rf_mean_cosine",
                               "rf_pr_norm", "spikes_exc_mean", "mean_x_tar_se")}})
        if r.accuracy is not None and r.batch % a.val_every == 0:
            v = model.validate()
            va = v.accuracy if v.accuracy is not None else float("nan")
            vp = v.phi if v.phi is not None else float("nan")
            val_hist.append({"batch": r.batch, "val_acc": va, "val_phi": vp})
            print(f"  [{a.tag}] batch {r.batch:>3}  val_acc {va:.4f}  val_phi {vp:.4f}", flush=True)

    test = model.test()
    ta = test.accuracy if test.accuracy is not None else float("nan")
    tp = test.phi if test.phi is not None else float("nan")
    out = dict(config=cfg, test_acc=ta, test_phi=tp, elapsed_s=round(time.time() - t0, 1),
               val_history=val_hist, stats_history=stats_hist)
    with open(os.path.join(a.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{a.tag}] DONE  test_acc={ta:.4f} test_phi={tp:.4f}  -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
