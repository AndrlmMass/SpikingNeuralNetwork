"""
WTA inhibition test — oriented RF + TripletSTDP + WTA E/I connectivity.

Replaces the Mexican-hat I→E ring pattern with winner-takes-all:
  - Each E neuron -> exactly one I neuron (round-robin)
  - Each I neuron -> all E neurons (uniform inhibition)

Usage
-----
# Default run
python experiments/generic_testing/test_wta.py

# Compare against Mexican-hat baseline
python experiments/generic_testing/test_wta.py --no-wta

# Quick smoke-test (~2 min)
python experiments/generic_testing/test_wta.py --train-all 1000 --epochs 1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn


def parse_args():
    parser = argparse.ArgumentParser(description="WTA inhibition test run")

    parser.add_argument("--no-wta", action="store_true",
                        help="Disable WTA and use default Mexican-hat inhibition instead")
    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "geomfig"])
    parser.add_argument("--train-all", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--val-all", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=5)

    parser.add_argument("--peak-ei", type=float, default=2.0,
                        help="E->I synapse strength (default: 2.0)")
    parser.add_argument("--peak-ie", type=float, default=-2.0,
                        help="I->E synapse strength (default: -2.0, negative = inhibitory)")

    return parser.parse_args()


def main():
    args = parse_args()
    use_wta = not args.no_wta
    inhibition_label = "wta" if use_wta else "mexican_hat"

    out_dir = os.path.join(
        os.path.dirname(__file__), "results",
        f"wta_{inhibition_label}_{args.dataset}_s{args.seed}_e{args.epochs}_n{args.train_all}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  WTA inhibition test — inhibition: {inhibition_label.upper()}")
    print("=" * 60)
    print(f"  Dataset       : {args.dataset}")
    print(f"  Train images  : {args.train_all} / epoch  x {args.epochs} epochs")
    print(f"  WTA           : {use_wta}")
    print(f"  peak_ei       : {args.peak_ei}")
    print(f"  peak_ie       : {args.peak_ie}")
    print(f"  Seed          : {args.seed}")
    print("=" * 60 + "\n")

    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01,
        density_ee=0.01,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=4.0,
        peak_ee=1.0,
        peak_ei=args.peak_ei,
        peak_ie=args.peak_ie,
        sigma_x=3.0,
        sigma_x_lognormal_std=1.0,
        sigma_x_lognormal_max=0.0,
        n_orientations=4,
        orientation_mode="block",
        gamma=0.4,
        r_cut_factor=3.0,
        wta_inhibition=use_wta,
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

    learner = snn.TripletSTDP(
        learning_rate=0.01,
        tau_trace=20,
        tau_x=101.0,
        tau_y=125.0,
        A2_plus=5e-10,
        A3_plus=6.2e-3,
        A2_minus=7e-3,
        A3_minus=2.3e-4,
        w_max=10.0,
        mu_weight=0.6,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

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
    print(f"Inhibition     : {inhibition_label.upper()}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(f"Test phi       : {test_phi:.4f}")
    print(f"Best val acc   : {best_val_acc:.4f}")
    print(f"Elapsed total  : {elapsed:.1f} s")
    print(f"{'=' * 55}")

    results = {
        "rule": "TripletSTDP",
        "inhibition": inhibition_label,
        "config": {
            "dataset": args.dataset,
            "train_all": args.train_all,
            "epochs": args.epochs,
            "effective_samples": args.train_all * args.epochs,
            "seed": args.seed,
            "peak_ei": args.peak_ei,
            "peak_ie": args.peak_ie,
            "wta_inhibition": use_wta,
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
