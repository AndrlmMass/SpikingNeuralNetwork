"""
Single-run experiment script: receptive fields vs random weights.

The regulariser is fixed to neuron-wise normalisation. The only free
parameters are weight initialisation type and random seed.

Usage
-----
# Local smoke-test
python experiments/RF_article/run_experiment.py --weight-type rf --seed 0 --epochs 1 --dataset geomfig

# Full MNIST run
python experiments/RF_article/run_experiment.py --weight-type rf --seed 0
python experiments/RF_article/run_experiment.py --weight-type random --seed 0

# Custom output location (used by SLURM job arrays)
python experiments/RF_article/run_experiment.py --weight-type rf --seed 3 --output-dir results/rf_s3
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import neurosnn as snn


def parse_args():
    parser = argparse.ArgumentParser(description="SNN RF vs random weight experiment")

    # Sweep parameters
    parser.add_argument(
        "--weight-type",
        type=str,
        default="rf",
        choices=["rf", "random", "oriented_rf"],
        help="Weight initialisation type (default: rf)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for full reproducibility (default: 0)",
    )

    # Dataset / training control
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=[
            "mnist",
            "kmnist",
            "fmnist",
            "fashionmnist",
            "notmnist",
            "geomfig",
            "fcx1",
        ],
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Training epochs (default: 5)"
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Validate every N batches (default: 10)",
    )
    parser.add_argument("--train-all", type=int, default=59000)
    parser.add_argument("--train-batch", type=int, default=1000)
    parser.add_argument("--val-all", type=int, default=1000)
    parser.add_argument("--val-batch", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=10000)
    parser.add_argument("--test-batch", type=int, default=10000)

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results.json (default: auto-generated)",
    )

    # Log-normal RF size diversity (Phase 2 sweep parameters)
    parser.add_argument(
        "--lognorm-se-mean",
        type=float,
        default=3.0,
        help="Mean RF size (pixels) for oriented W_se log-normal distribution (default: 3.0 = sigma_x default)",
    )
    parser.add_argument(
        "--lognorm-se-std",
        type=float,
        default=0.0,
        help="Std of RF sizes for W_se log-normal (0 = disabled, try 1.5)",
    )
    parser.add_argument(
        "--lognorm-ee-mean",
        type=float,
        default=1.0,
        help="Mean RF size (E-grid pixels) for W_ee log-normal distribution (default: 1.0 = auto sigma_ee)",
    )
    parser.add_argument(
        "--lognorm-ee-std",
        type=float,
        default=0.0,
        help="Std of RF sizes for W_ee log-normal (0 = disabled, try 0.5)",
    )

    return parser.parse_args()


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = f"{args.weight_type}_s{args.seed}"
    return os.path.join(base, tag)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Weight initialisation — the experimental variable
    # ------------------------------------------------------------------
    weight_kwargs = dict(
        density_se=0.05,
        density_ee=0.02,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=1.0,
        peak_ee=0.5,
        peak_ei=1.0,
        peak_ie=-0.7,
    )
    if args.weight_type == "rf":
        weights = snn.weights.receptive_fields(
            **weight_kwargs,
            sigma_ee_mean=args.lognorm_ee_mean,
            sigma_ee_lognormal_std=args.lognorm_ee_std,
        )
    elif args.weight_type == "oriented_rf":
        weights = snn.weights.oriented_receptive_fields(
            **weight_kwargs,
            sigma_x=args.lognorm_se_mean,
            sigma_x_lognormal_std=args.lognorm_se_std,
            sigma_ee_mean=args.lognorm_ee_mean,
            sigma_ee_lognormal_std=args.lognorm_ee_std,
        )
    else:
        weights = snn.weights.random(**weight_kwargs)

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
            var_noise=0.0,
            spike_adaptation=True,
            tau_adaptation=200.0,
            delta_adaptation=0.5,
        ),
        weights=weights,
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
    # Regulariser — fixed: neuron-wise normalisation
    # ------------------------------------------------------------------
    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=args.seed,
        num_steps=350,
        all_images_train=args.train_all,
        batch_image_train=args.train_batch,
        all_images_val=args.val_all,
        batch_image_val=args.val_batch,
        all_images_test=args.test_all,
        batch_image_test=args.test_batch,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    config = dict(
        dataset=args.dataset,
        weight_type=args.weight_type,
        seed=args.seed,
        epochs=args.epochs,
        val_every=args.val_every,
        lognorm_se_mean=args.lognorm_se_mean,
        lognorm_se_std=args.lognorm_se_std,
        lognorm_ee_mean=args.lognorm_ee_mean,
        lognorm_ee_std=args.lognorm_ee_std,
    )
    print(
        f"\nConfig — dataset: {args.dataset} | weight_type: {args.weight_type}"
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
            val_history.append(
                {"batch": result.batch, "val_acc": val_acc, "val_phi": val_phi}
            )
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
