"""
Single-run experiment script for HPC sweep and local testing.

Base parameters come from big_comb.py defaults, with overrides from the last
working main.py MNIST configuration. Sleep duration, membrane noise, regularizer
type/mode, and random seed are the free parameters for Phase 1/2 sweeps.

Usage
-----
# Local smoke-test
python experiments/run_experiment.py

# Phase 1 — regularizer comparison (fixed noise, vary reg type/mode)
python experiments/run_experiment.py --reg-type sleep --reg-mode static --seed 0
python experiments/run_experiment.py --reg-type normalize --reg-mode neuron --seed 1
python experiments/run_experiment.py --reg-type none --seed 2

# Phase 2 — sleep/noise grid
python experiments/run_experiment.py --sleep-duration 200 --var-noise 2.0 --seed 3

# Custom output location (used by SLURM job arrays)
python experiments/run_experiment.py --seed 42 --output-dir results/phase1/sleep_static_s42
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import neurosnn as snn
from neurosnn._utils.weight_sampler import WeightSampler
from neurosnn._plot.weights import plot_weight_sample_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="SNN single-run experiment")

    # Sweep parameters
    parser.add_argument(
        "--sleep-duration",
        type=int,
        default=100,
        help="Timesteps per sleep episode (default: 100)",
    )
    parser.add_argument(
        "--var-noise",
        type=float,
        default=0.1,
        help="Membrane noise variance (default: 0.1)",
    )
    parser.add_argument(
        "--reg-type",
        type=str,
        default="sleep",
        choices=["sleep", "normalize", "none"],
        help="Regularization type (default: sleep)",
    )
    parser.add_argument(
        "--reg-mode",
        type=str,
        default="static",
        choices=["static", "layer", "neuron"],
        help="Regularization schedule mode (default: static)",
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
        "--epochs", type=int, default=1, help="Training epochs (default: 1)"
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Validate every N batches (default: 10)",
    )
    parser.add_argument(
        "--train-all",
        type=int,
        default=59000,
        help="Number of training data samples per epoch",
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=1000,
        help="Number of training data samples per epoch",
    )
    parser.add_argument(
        "--val-all",
        type=int,
        default=1000,
        help="Number of training data samples per epoch",
    )
    parser.add_argument(
        "--val-batch",
        type=int,
        default=1000,
        help="Number of training data samples per epoch",
    )
    parser.add_argument(
        "--test-all",
        type=int,
        default=1000,
        help="Number of training data samples per epoch",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=10000,
        help="Number of training data samples per epoch",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results.json (default: auto-generated)",
    )
    parser.add_argument(
        "--plot-weight-trajectories",
        action="store_true",
        default=False,
        help="Collect and plot w_ee weight trajectories (off by default)",
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=10000,
        help="Only show the first N timesteps in the weight trajectory plot (default: 10000; 0 = full run)",
    )

    return parser.parse_args()


def build_output_dir(args) -> str:
    """Generate a deterministic output path from run parameters."""
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = (
        f"{args.reg_type}_{args.reg_mode}"
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
    # Layer — big_comb defaults, main.py overrides applied
    # ------------------------------------------------------------------
    layer = snn.Layer(
        N_exc=1024,  # main.py: 1024  (big_comb default: 200)
        N_inh=225,  # main.py: 225   (big_comb default: 50)
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0,  # main.py override (big_comb: 30)
            tau_m_inh=15.0,  # main.py override (big_comb: 30)
            tau_syn_exc=10.0,  # main.py override (big_comb: 30)
            tau_syn_inh=9.0,  # main.py override (big_comb: 30)
            membrane_resistance_exc=15.0,  # main.py override (big_comb: 30)
            membrane_resistance_inh=15.0,  # main.py override (big_comb: 30)
            resting_potential=-70.0,  # big_comb default
            reset_potential=-80.0,  # big_comb default
            spike_threshold=-55.0,  # big_comb default
            min_mp=-100.0,  # big_comb default
            max_mp=40.0,  # big_comb default
            mean_noise=0.0,  # big_comb default
            var_noise=args.var_noise,  # sweep parameter
            spike_adaptation=True,  # big_comb default
            tau_adaptation=200.0,  # main.py override (big_comb: 100)
            delta_adaptation=0.5,  # main.py override (big_comb: 1.0)
        ),
        weights=snn.weights.random(  # main.py override (big_comb: receptive_fields)
            density_se=0.05,  # big_comb default
            density_ee=0.05,  # main.py override (big_comb: 0.01)
            density_ei=0.05,  # big_comb default
            density_ie=0.05,  # big_comb default
            peak_se=1.0,  # main.py override (big_comb: 0.1)
            peak_ee=0.5,  # main.py override (big_comb: 0.3)
            peak_ei=1.0,  # main.py override (big_comb: 0.3)
            peak_ie=-0.7,  # main.py override (big_comb: -0.2)
        ),
    )

    # ------------------------------------------------------------------
    # Learner — big_comb defaults, main.py overrides applied
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,  # main.py override (big_comb: 0.0008)
        tau_trace=20,  # main.py override (big_comb: 25)
        w_max=10.0,  # big_comb default
        mu_weight=0.6,  # big_comb default
        update_freq=100,  # big_comb default
        clip_weights=True,  # main.py override (big_comb: False)
        min_weight_exc=0.01,  # big_comb default
        max_weight_exc=25.0,  # big_comb default
        min_weight_inh=-25.0,  # big_comb default
        max_weight_inh=-0.01,  # big_comb default
    )

    # ------------------------------------------------------------------
    # Regularizer — selected by --reg-type
    # ------------------------------------------------------------------
    if args.reg_type == "sleep":
        regularizer = snn.regularizer.Sleep(
            duration=args.sleep_duration,
            frequency=1050,  # big_comb default
            mode=args.reg_mode,
        )
    elif args.reg_type == "normalize":
        regularizer = snn.regularizer.Normalize(
            frequency=1050,
            mode=args.reg_mode,
        )
    else:  # none
        regularizer = None

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,  # main.py: 784  (big_comb default: 225)
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

    # ------------------------------------------------------------------
    # Weight trajectory sampler — only created when --plot-weight-trajectories
    # is passed; zero overhead otherwise.
    # ------------------------------------------------------------------
    _sampler = None
    if args.plot_weight_trajectories:
        _sampler = WeightSampler(N_x=784, N_exc=layer.N_exc, n_samples=30)
        if args.reg_type != "none":
            regularizer.record_fn_se = _sampler.record_se
            regularizer.record_fn_ee = _sampler.record_ee

    t_start = time.time()

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",  # main.py override (big_comb: "top")
        use_LR=True,
        use_phi=True,
        use_pca=True,
        pca_variance=15,  # main.py override (big_comb: 0.95)
        stat_tracking_frequency=10500,  # main.py override (big_comb: 1000)
        record_fn_awake_se=_sampler.record_awake_se if _sampler else None,
        record_fn_awake_ee=_sampler.record_awake_ee if _sampler else None,
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
    # Weight trajectory figure
    # ------------------------------------------------------------------
    if _sampler is not None and (_sampler.awake_ee or _sampler.reg_ee):
        config_label = f"{args.reg_type} – {args.reg_mode}"
        fig_path = os.path.join(
            output_dir,
            f"weight_trajectories_{args.reg_type}_{args.reg_mode}.pdf",
        )
        x_max = args.plot_window if args.plot_window > 0 else None
        plot_weight_sample_trajectories(_sampler, config_label, fig_path, x_max=x_max)
        print(f"Weight trajectory figure -> {fig_path}")

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
