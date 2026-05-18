"""
Single-run experiment script.

Base parameters come from big_comb.py defaults, with overrides from the last
working main.py MNIST configuration. Sleep duration and membrane noise are the
two free parameters for sweeping.

Usage
-----
    python experiments/run_experiment.py
    python experiments/run_experiment.py --sleep-duration 200 --var-noise 2.0
    python experiments/run_experiment.py --dataset kmnist --sleep-duration 100 --var-noise 4.0
    python experiments/run_experiment.py --epochs 2 --val-every 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neurosnn as snn


def main():
    parser = argparse.ArgumentParser(description="SNN single-run experiment")
    parser.add_argument("--sleep-duration", type=int, default=100,
                        help="Timesteps per sleep episode (default: 100)")
    parser.add_argument("--var-noise", type=float, default=0.1,
                        help="Membrane noise variance during sleep (default: 0.1)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "kmnist", "fmnist", "fashionmnist",
                                 "notmnist", "geomfig", "fcx1"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10,
                        help="Validate every N batches (default: 10)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Layer — big_comb defaults, main.py overrides applied
    # ------------------------------------------------------------------
    layer = snn.Layer(
        N_exc=1024,           # main.py: 1024 (big_comb default: 200)
        N_inh=225,            # main.py: 225  (big_comb default: 50)
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0,             # main.py override (big_comb: 30)
            tau_m_inh=15.0,             # main.py override (big_comb: 30)
            tau_syn_exc=10.0,           # main.py override (big_comb: 30)
            tau_syn_inh=9.0,            # main.py override (big_comb: 30)
            membrane_resistance_exc=15.0,  # main.py override (big_comb: 30)
            membrane_resistance_inh=15.0,  # main.py override (big_comb: 30)
            resting_potential=-70.0,    # big_comb default
            reset_potential=-80.0,      # big_comb default
            spike_threshold=-55.0,      # big_comb default
            min_mp=-100.0,              # big_comb default
            max_mp=40.0,                # big_comb default
            mean_noise=0.0,             # big_comb default
            var_noise=args.var_noise,   # sweep parameter
            spike_adaptation=True,      # big_comb default
            tau_adaptation=200.0,       # main.py override (big_comb: 100)
            delta_adaptation=0.5,       # main.py override (big_comb: 1.0)
        ),
        weights=snn.weights.random(     # main.py override (big_comb: receptive_fields)
            density_se=0.05,            # big_comb default
            density_ee=0.05,            # main.py override (big_comb: 0.01)
            density_ei=0.05,            # big_comb default
            density_ie=0.05,            # big_comb default
            peak_se=1.0,                # main.py override (big_comb: 0.1)
            peak_ee=0.5,                # main.py override (big_comb: 0.3)
            peak_ei=1.0,                # main.py override (big_comb: 0.3)
            peak_ie=-0.7,               # main.py override (big_comb: -0.2)
        ),
    )

    # ------------------------------------------------------------------
    # Learner — big_comb defaults, main.py overrides applied
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,   # main.py override (big_comb: 0.0008)
        tau_trace=20,           # main.py override (big_comb: 25)
        w_max=10.0,             # big_comb default
        mu_weight=0.6,          # big_comb default
        update_freq=100,        # big_comb default
        clip_weights=True,      # main.py override (big_comb: False)
        min_weight_exc=0.01,    # big_comb default
        max_weight_exc=25.0,    # big_comb default
        min_weight_inh=-25.0,   # big_comb default
        max_weight_inh=-0.01,   # big_comb default
    )

    # ------------------------------------------------------------------
    # Regularizer — sleep only, no normalization
    # ------------------------------------------------------------------
    regularizer = snn.regularizer.Sleep(
        duration=args.sleep_duration,   # sweep parameter (main.py default: 100)
        frequency=1050,                 # big_comb default
        mode="static",                  # big_comb default
    )

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,             # main.py: 784 (big_comb default: 225)
        classes=list(range(10)),
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

    print(
        f"\nConfig — dataset: {args.dataset} | sleep_duration: {args.sleep_duration}"
        f" | var_noise: {args.var_noise} | epochs: {args.epochs}\n"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_acc = 0.0

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",   # main.py override (big_comb: "top")
        use_LR=True,
        use_phi=True,
        use_pca=True,
        pca_variance=15,            # main.py override (big_comb: 0.95)
        stat_tracking_frequency=10500,  # main.py override (big_comb: 1000)
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
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test = model.test()
    print(f"\n{'=' * 55}")
    print(f"Test accuracy : {test.accuracy:.4f}")
    print(f"Test phi      : {test.phi:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
