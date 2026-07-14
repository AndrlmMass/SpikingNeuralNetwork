"""
TraceSTDP (E synapses) + VogelsSTDP (I→E synapses) test.

Oriented RFs, WTA inhibition, Normalize homeostasis.

Usage
-----
# Default run (~20-35 min)
python experiments/generic_testing/test_trace_vogels.py

# Quick smoke-test (~2 min)
python experiments/generic_testing/test_trace_vogels.py --train-all 1000 --epochs 1

# Without Vogels iSTDP (fixed I→E weights, for comparison)
python experiments/generic_testing/test_trace_vogels.py --no-vogels

# Tune the E target firing rate (default rho_0=0.1 ≈ 5 Hz at tau_trace=20ms)
python experiments/generic_testing/test_trace_vogels.py --rho0 0.04
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn


def parse_args():
    p = argparse.ArgumentParser(description="TraceSTDP + VogelsSTDP test")

    p.add_argument("--no-vogels", action="store_true",
                   help="Disable Vogels iSTDP (fixed I→E weights)")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "kmnist", "fmnist", "fashionmnist", "geomfig"])
    p.add_argument("--train-all", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--val-all", type=int, default=1000)
    p.add_argument("--test-all", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-every", type=int, default=5)

    # TraceSTDP params
    p.add_argument("--lr", type=float, default=0.0008)
    p.add_argument("--tau-trace", type=int, default=20)
    p.add_argument("--w-max", type=float, default=10.0)
    p.add_argument("--mu-weight", type=float, default=0.6)

    # VogelsSTDP params
    p.add_argument("--lr-inh", type=float, default=0.01,
                   help="Learning rate for I→E Vogels updates (default: 0.01)")
    p.add_argument("--rho0", type=float, default=0.1,
                   help="Target E trace for homeostasis. rho_0 = target_Hz * tau_trace / 1000. "
                        "Default 0.1 ≈ 5 Hz at tau_trace=20ms.")

    return p.parse_args()


def main():
    args = parse_args()
    use_vogels = not args.no_vogels
    label = "trace+vogels" if use_vogels else "trace_only"

    out_dir = os.path.join(
        os.path.dirname(__file__), "results",
        f"{label}_{args.dataset}_s{args.seed}_e{args.epochs}_n{args.train_all}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  TraceSTDP + {'VogelsSTDP' if use_vogels else 'fixed I→E'}")
    print("=" * 60)
    print(f"  Dataset        : {args.dataset}")
    print(f"  Train images   : {args.train_all} / epoch  x {args.epochs} epochs")
    print(f"  Vogels iSTDP   : {use_vogels}")
    if use_vogels:
        print(f"  lr_inh         : {args.lr_inh}")
        print(f"  rho_0          : {args.rho0}  (~{args.rho0 * 1000 / args.tau_trace:.1f} Hz target)")
    print(f"  Seed           : {args.seed}")
    print("=" * 60 + "\n")

    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01,
        density_ee=0.01,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=4.0,
        peak_ee=1.0,
        peak_ei=2.0,
        peak_ie=-2.0,
        sigma_x=3.0,
        sigma_x_lognormal_std=1.0,
        sigma_x_lognormal_max=0.0,
        n_orientations=4,
        orientation_mode="block",
        gamma=0.4,
        r_cut_factor=3.0,
        wta_inhibition=True,
    )

    layer = snn.Layer(
        N_exc=1024,
        N_inh=225,
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0,
            tau_m_inh=15.0,
            tau_syn_exc=10.0,
            tau_syn_inh=9.0,
            membrane_resistance_exc=15.0,
            membrane_resistance_inh=15.0,
            resting_potential=-70.0,
            reset_potential=-80.0,
            spike_threshold=-55.0,
            min_mp=-100.0,
            max_mp=40.0,
            mean_noise=0.0,
            var_noise=0.0,
            spike_adaptation=True,
            tau_adaptation=200.0,
            delta_adaptation=0.5,
        ),
        weights=weights,
    )

    learner = snn.learner.TraceSTDP(
        learning_rate=args.lr,
        tau_trace=args.tau_trace,
        w_max=args.w_max,
        mu_weight=args.mu_weight,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    inh_learner = snn.VogelsSTDP(
        learning_rate=args.lr_inh,
        rho_0=args.rho0,
        mu_weight=args.mu_weight,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    ) if use_vogels else None

    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=args.seed,
        num_steps=350,
        all_images_train=args.train_all,
        batch_image_train=1000,
        all_images_val=args.val_all,
        batch_image_val=1000,
        all_images_test=args.test_all,
        batch_image_test=1000,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []
    t_start = time.time()

    print("Starting training — Numba JIT warmup on first batch (~10-30 s)...\n")

    train_gen = model.train(
        layers=[layer],
        learner=learner,
        inh_learner=inh_learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=False,
        pca_variance=15,
        track_stats=True,
        stat_tracking_frequency=10500,
    )

    for result in train_gen:
        if result.accuracy is None:
            continue

        if result.batch % args.val_every == 0:
            val = model.validate()
            val_acc = val.accuracy if val.accuracy is not None else float("nan")
            val_phi = val.phi if val.phi is not None else float("nan")

            elapsed = time.time() - t_start
            print(
                f"epoch {result.epoch + 1}/{args.epochs}  "
                f"batch {result.batch:>3}  "
                f"train_acc {result.accuracy:.4f}  "
                f"train_phi {result.phi:.3f}  "
                f"val_acc {val_acc:.4f}  "
                f"val_phi {val_phi:.3f}  "
                f"[{elapsed:.0f}s]"
            )

            if result.stats:
                s = result.stats
                print(
                    f"  mp_exc {s['mean_mp_exc']:.2f}  "
                    f"adapt {s['mean_adaptation']:.3f}  "
                    f"thr {s['mean_spike_threshold']:.2f}  "
                    f"delta_w {s['mean_delta_w']:.2e}"
                )

            val_history.append({
                "epoch": result.epoch,
                "batch": result.batch,
                "val_acc": val_acc,
                "val_phi": val_phi,
                "elapsed_s": round(elapsed, 1),
            })
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy
                best_val_phi = val_phi

    elapsed = time.time() - t_start

    print("\nRunning final test set evaluation...")
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Learning rule  : TraceSTDP + {'VogelsSTDP' if use_vogels else 'fixed I→E'}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(f"Test phi       : {test_phi:.4f}")
    print(f"Best val acc   : {best_val_acc:.4f}")
    print(f"Elapsed total  : {elapsed:.1f} s")
    print(f"{'=' * 55}")

    results = {
        "rule": label,
        "config": {
            "dataset": args.dataset,
            "train_all": args.train_all,
            "epochs": args.epochs,
            "effective_samples": args.train_all * args.epochs,
            "seed": args.seed,
            "lr": args.lr,
            "tau_trace": args.tau_trace,
            "w_max": args.w_max,
            "mu_weight": args.mu_weight,
            "use_vogels": use_vogels,
            "lr_inh": args.lr_inh if use_vogels else None,
            "rho_0": args.rho0 if use_vogels else None,
            "wta_inhibition": True,
            "weight_type": "oriented_rf",
            "regularizer": "normalize_neuron",
        },
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "elapsed_s": round(elapsed, 1),
        "val_history": val_history,
    }
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
