"""
Generic smoke-test script — runs the SNN with any combination of parameters.

Defaults are tuned for a fast local check (~1 min on geomfig, 1 epoch).
Override any flag to replicate a specific experiment configuration.

Usage
-----
# Fastest sanity check
python experiments/generic_testing/run_test.py

# MNIST, sleep regulariser, RF weights
python experiments/generic_testing/run_test.py \
    --dataset mnist --reg-type sleep --reg-mode neuron --weight-type rf --epochs 5

# Noise sweep point
python experiments/generic_testing/run_test.py \
    --dataset mnist --reg-type sleep --reg-mode static --var-noise 0.3 --sleep-duration 200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn


def parse_args():
    parser = argparse.ArgumentParser(description="Generic SNN test run")

    # --- Weight initialisation ---
    parser.add_argument(
        "--weight-type",
        type=str,
        default="random",
        choices=["rf", "random"],
        help="Weight initialisation type (default: random)",
    )

    # --- Regularisation ---
    parser.add_argument(
        "--reg-type",
        type=str,
        default="normalize",
        choices=["sleep", "normalize", "none"],
        help="Regularisation type (default: normalize)",
    )
    parser.add_argument(
        "--reg-mode",
        type=str,
        default="neuron",
        choices=["static", "layer", "neuron"],
        help="Regularisation schedule mode (default: neuron)",
    )
    parser.add_argument(
        "--sleep-duration",
        type=int,
        default=100,
        help="Timesteps per sleep episode, only used when --reg-type sleep (default: 100)",
    )
    parser.add_argument(
        "--var-noise",
        type=float,
        default=0.0,
        help="Membrane noise variance (default: 0.0)",
    )
    parser.add_argument(
        "--gabor",
        action="store_true",
        default=False,
        help="Whether to use Gabor filters for RF weights (default: False)",
    )

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=0)

    # --- Dataset / training ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="geomfig",
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--train-all", type=int, default=1000)
    parser.add_argument("--train-batch", type=int, default=1000)
    parser.add_argument("--val-all", type=int, default=200)
    parser.add_argument("--val-batch", type=int, default=200)
    parser.add_argument("--test-all", type=int, default=200)
    parser.add_argument("--test-batch", type=int, default=200)

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write results.json (default: experiments/generic_testing/results/<tag>)",
    )

    return parser.parse_args()


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = (
        f"{args.dataset}_{args.weight_type}"
        f"_{args.reg_type}_{args.reg_mode}"
        f"_vn{args.var_noise}_s{args.seed}"
    )
    return os.path.join(base, tag)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"\nConfig — dataset: {args.dataset} | weight: {args.weight_type}"
        f" | reg: {args.reg_type}/{args.reg_mode}"
        f" | var_noise: {args.var_noise} | sleep_dur: {args.sleep_duration}"
        f" | seed: {args.seed} | epochs: {args.epochs}\n"
    )

    # --- Weights ---
    weight_kwargs = dict(
        density_se=0.05,
        density_ee=0.05,
        density_ei=0.05,
        density_ie=0.05,
        peak_se=1.0,
        peak_ee=0.5,
        peak_ei=1.0,
        peak_ie=-0.7,
    )
    if args.weight_type == "rf":
        weights = snn.weights.receptive_fields(**weight_kwargs)
    else:
        weights = snn.weights.random(**weight_kwargs)

    # --- Layer ---
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
        weights=weights,
    )

    # --- Learner ---
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

    # --- Regulariser ---
    if args.reg_type == "sleep":
        regularizer = snn.regularizer.Sleep(
            duration=args.sleep_duration,
            frequency=1050,
            mode=args.reg_mode,
        )
    elif args.reg_type == "normalize":
        regularizer = snn.regularizer.Normalize(frequency=1050, mode=args.reg_mode)
    else:
        regularizer = None

    # --- Model ---
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
        gabor=args.gabor,
    )

    # --- Training ---
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

    # --- Test ---
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test phi      : {test_phi:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Elapsed (s)   : {elapsed:.1f}")
    print(f"{'=' * 55}")

    results = {
        "config": vars(args),
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
