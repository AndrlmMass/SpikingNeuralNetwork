"""
Single-run experiment script for Phase 2 sleep/noise grid sweep.

Sleep duration, membrane noise, regularizer mode, and random seed are the
free parameters. Regularizer type is fixed to sleep for all Phase 2 runs.

Usage
-----
    python experiments/noise_article/sleep_noise_optimization/run_sleep_tuning.py
    python experiments/noise_article/sleep_noise_optimization/run_sleep_tuning.py --sleep-duration 200 --var-noise 2.0 --reg-mode layer
    python experiments/noise_article/sleep_noise_optimization/run_sleep_tuning.py --dataset kmnist --sleep-duration 100 --var-noise 4.0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import neurosnn as snn


def parse_args():
    parser = argparse.ArgumentParser(description="SNN Phase 2 sleep/noise sweep")
    parser.add_argument("--sleep-duration", type=int, default=100,
                        help="Timesteps per sleep episode (default: 100)")
    parser.add_argument("--var-noise", type=float, default=0.1,
                        help="Membrane noise variance (default: 0.1)")
    parser.add_argument("--reg-type", type=str, default="sleep",
                        choices=["sleep", "normalize", "none"],
                        help="Regularization type (default: sleep)")
    parser.add_argument("--reg-mode", type=str, default="static",
                        choices=["static", "layer", "neuron"],
                        help="Regularization mode (default: static)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "kmnist", "fmnist", "fashionmnist",
                                 "notmnist", "geomfig", "fcx1"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10,
                        help="Validate every N batches (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for results.json (default: auto-generated)")
    return parser.parse_args()


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = (
        f"{args.reg_mode}"
        f"_sd{args.sleep_duration}"
        f"_vn{args.var_noise}"
        f"_s{args.seed}"
    )
    return os.path.join(base, tag)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Layer
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
            var_noise=args.var_noise,
            spike_adaptation=True,
            tau_adaptation=200.0,
            delta_adaptation=0.5,
        ),
        weights=snn.weights.random(
            density_se=0.05,
            density_ee=0.05,
            density_ei=0.05,
            density_ie=0.05,
            peak_se=1.0,
            peak_ee=0.5,
            peak_ei=1.0,
            peak_ie=-0.7,
        ),
    )

    # ------------------------------------------------------------------
    # Learner
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,
        tau_trace=20,
        w_max=10.0,
        mu_weight=0.6,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    # ------------------------------------------------------------------
    # Regularizer
    # ------------------------------------------------------------------
    if args.reg_type == "sleep":
        regularizer = snn.regularizer.Sleep(
            duration=args.sleep_duration,
            frequency=1050,
            mode=args.reg_mode,
        )
    elif args.reg_type == "normalize":
        regularizer = snn.regularizer.Normalize(
            frequency=1050,
            mode=args.reg_mode,
        )
    else:
        regularizer = None

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=args.seed,
        num_steps=350,
        all_images_train=59000,
        batch_image_train=1000,
        all_images_val=1000,
        batch_image_val=1000,
        all_images_test=10000,
        batch_image_test=1000,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
    )

    config = dict(
        dataset=args.dataset,
        reg_type=args.reg_type,
        reg_mode=args.reg_mode,
        sleep_duration=args.sleep_duration,
        var_noise=args.var_noise,
        seed=args.seed,
        epochs=args.epochs,
        val_every=args.val_every,
    )
    print(
        f"\nConfig — dataset: {args.dataset} | reg: {args.reg_type}/{args.reg_mode}"
        f" | sleep_dur: {args.sleep_duration} | var_noise: {args.var_noise}"
        f" | seed: {args.seed} | epochs: {args.epochs}\n"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []
    t_start = time.time()

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=True,
        pca_variance=15,
        stat_tracking_frequency=10500,
    ):
        if result.accuracy is None:
            continue

        if result.batch % args.val_every == 0:
            val = model.validate()
            val_acc = val.accuracy if val.accuracy is not None else float("nan")
            val_phi = val.phi if val.phi is not None else float("nan")
            print(
                f"epoch {result.epoch + 1}  batch {result.batch:>3}"
                f"  train_acc {result.accuracy:.4f}  train_phi {result.phi:.3f}"
                f"  val_acc {val_acc:.4f}  val_phi {val_phi:.3f}"
            )
            val_history.append({"batch": result.batch, "val_acc": val_acc, "val_phi": val_phi})
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy
                best_val_phi = val_phi

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test phi      : {test_phi:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Elapsed (s)   : {elapsed:.1f}")
    print(f"{'=' * 55}")

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    results = {
        "config": config,
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "elapsed_s": round(elapsed, 1),
        "val_history": val_history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()
