"""
Single-run test of the Triplet STDP learning rule on MNIST with oriented RFs.

Expected accuracy: ~85-88% (2 epochs x 20k images = 40k effective samples).
To reach ~90% you need ~50k+ effective samples — set --train-all 25000 --epochs 3
or increase epochs.

Usage
-----
# Default run (~20-35 min)
python experiments/generic_testing/test_triplet_stdp.py

# Longer run for higher accuracy (~50-70 min, ~90% target)
python experiments/generic_testing/test_triplet_stdp.py --train-all 25000 --epochs 3

# Quick smoke-test (geomfig, ~2 min)
python experiments/generic_testing/test_triplet_stdp.py --dataset geomfig --train-all 2000 --epochs 2
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn


def parse_args():
    parser = argparse.ArgumentParser(description="Triplet STDP test run")

    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "geomfig"])
    parser.add_argument("--train-all", type=int, default=20000,
                        help="Training images per epoch (default: 20000)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs (default: 2)")
    parser.add_argument("--val-all", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every N batches (default: 5)")

    # TripletSTDP learning rule parameters
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01; higher than TraceSTDP because "
                             "Pfister amplitudes are small — scales updates to ~3e-4 per event)")
    parser.add_argument("--tau-x", type=float, default=101.0,
                        help="Slow pre-synaptic trace time constant in ms (default: 101)")
    parser.add_argument("--tau-y", type=float, default=125.0,
                        help="Slow post-synaptic trace time constant in ms (default: 125)")
    parser.add_argument("--A2-plus", type=float, default=5e-10,
                        help="Pair LTP amplitude (default: 5e-10, tiny — triplet term dominates)")
    parser.add_argument("--A3-plus", type=float, default=6.2e-3,
                        help="Triplet LTP amplitude (default: 6.2e-3, Pfister 2006 VC fit)")
    parser.add_argument("--A2-minus", type=float, default=7e-3,
                        help="Pair LTD amplitude (default: 7e-3, Pfister 2006 VC fit)")
    parser.add_argument("--A3-minus", type=float, default=2.3e-4,
                        help="Triplet LTD amplitude (default: 2.3e-4, Pfister 2006 VC fit)")
    parser.add_argument("--w-max", type=float, default=10.0)
    parser.add_argument("--mu-weight", type=float, default=0.6)

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "results",
                           f"triplet_{args.dataset}_s{args.seed}_e{args.epochs}_n{args.train_all}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Triplet STDP — oriented RF — MNIST test")
    print("=" * 60)
    print(f"  Dataset       : {args.dataset}")
    print(f"  Train images  : {args.train_all} / epoch  x {args.epochs} epochs")
    print(f"  Effective passes: {args.train_all * args.epochs:,}")
    print(f"  Learning rate : {args.lr}")
    print(f"  tau_x / tau_y : {args.tau_x} / {args.tau_y} ms")
    print(f"  A2+ / A3+     : {args.A2_plus:.2e} / {args.A3_plus:.2e}")
    print(f"  A2- / A3-     : {args.A2_minus:.2e} / {args.A3_minus:.2e}")
    print(f"  Seed          : {args.seed}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Oriented receptive-field weights (same tuned config as run_test.py)
    # ------------------------------------------------------------------
    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01,
        density_ee=0.01,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=4.0,
        peak_ee=1.0,
        peak_ei=2.0,
        peak_ie=-2.0,
        sigma_x=3.0,               # mean RF size in pixels
        sigma_x_lognormal_std=1.0, # log-normal size diversity (V1-like)
        sigma_x_lognormal_max=0.0, # no upper clip
        n_orientations=4,
        orientation_mode="block",
        gamma=0.4,                 # aspect ratio σ_y / σ_x
        r_cut_factor=3.0,
    )

    # ------------------------------------------------------------------
    # Layer: 1024 excitatory, 225 inhibitory
    # LIF params tuned for stable spiking with spike-frequency adaptation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Triplet STDP learner
    #
    # Pfister & Gerstner (2006) visual-cortex amplitudes with a higher
    # learning rate than TraceSTDP because A2+/A3+/A2-/A3- are small
    # (~1e-3). At lr=0.01, typical Δw ~ lr * A3+ * r1 * o2 * bound
    #   ≈ 0.01 * 6.2e-3 * 1 * 1 * 5 ≈ 3.1e-4  — comparable to TraceSTDP.
    # ------------------------------------------------------------------
    learner = snn.TripletSTDP(
        learning_rate=args.lr,
        tau_trace=20,
        tau_x=args.tau_x,
        tau_y=args.tau_y,
        A2_plus=args.A2_plus,
        A3_plus=args.A3_plus,
        A2_minus=args.A2_minus,
        A3_minus=args.A3_minus,
        w_max=args.w_max,
        mu_weight=args.mu_weight,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    # ------------------------------------------------------------------
    # Normalisation homeostasis — deterministic, per-neuron mode
    # Using Normalize (not Sleep) so we test one new variable at a time.
    # Switch to Sleep for the full biologically-inspired pipeline.
    # ------------------------------------------------------------------
    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    # ------------------------------------------------------------------
    # Model (data + bookkeeping)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
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

            val_history.append({"epoch": result.epoch, "batch": result.batch,
                                 "val_acc": val_acc, "val_phi": val_phi,
                                 "elapsed_s": round(elapsed, 1)})
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy
                best_val_phi = val_phi

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------------
    print("\nRunning final test set evaluation...")
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(f"Test phi       : {test_phi:.4f}")
    print(f"Best val acc   : {best_val_acc:.4f}")
    print(f"Elapsed total  : {elapsed:.1f} s")
    print(f"{'=' * 55}")

    if test_acc < 0.85:
        print("\nNote: accuracy below 85% — try --train-all 25000 --epochs 3 "
              "for ~90% target (requires ~50k effective samples).")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "rule": "TripletSTDP",
        "config": {
            "dataset": args.dataset,
            "train_all": args.train_all,
            "epochs": args.epochs,
            "effective_samples": args.train_all * args.epochs,
            "seed": args.seed,
            "lr": args.lr,
            "tau_x": args.tau_x,
            "tau_y": args.tau_y,
            "A2_plus": args.A2_plus,
            "A3_plus": args.A3_plus,
            "A2_minus": args.A2_minus,
            "A3_minus": args.A3_minus,
            "w_max": args.w_max,
            "mu_weight": args.mu_weight,
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
