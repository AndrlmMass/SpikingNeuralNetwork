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
        default="oriented_rf",
        choices=["rf", "random", "oriented_rf"],
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

    # --- Log-normal RF size diversity ---
    parser.add_argument(
        "--n-orientations",
        type=int,
        default=4,
        help="Number of orientation groups for oriented_rf (default: 4, try 8 or 16)",
    )
    parser.add_argument(
        "--orientation-mode",
        type=str,
        default="block",
        choices=["block", "interleaved"],
        help="Orientation assignment: block=large spatial blocks, interleaved=cycling stripes (default: block)",
    )
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
        "--max-sigma-x",
        type=float,
        default=0.0,
        help="Upper clip for W_se RF size in pixels (0 = no clip, try 5.0)",
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

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=0)

    # --- Dataset / training ---
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument(
        "--max-rate-hz",
        type=float,
        default=90.0,
        help="Peak Poisson firing rate for input encoding in Hz (default: 90.0)",
    )
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
    parser.add_argument(
        "--plot-rfs",
        action="store_true",
        default=False,
        help="Save an oriented RF summary plot after weight initialisation (oriented_rf only)",
    )
    parser.add_argument(
        "--plot-spikes",
        action="store_true",
        default=False,
        help="Save heatmap spike/weight plots during training (written to plots/spikes/)",
    )
    parser.add_argument(
        "--plot-weights",
        action="store_true",
        default=False,
        help="Save full weight matrix heatmap and print E/I balance stats after init",
    )
    parser.add_argument(
        "--plot-single-neuron",
        action="store_true",
        default=False,
        help="Save a 2×2 weight plot (SE/EE/EI/IE) for a single excitatory neuron before training",
    )
    parser.add_argument(
        "--neuron-id",
        type=int,
        default=512,
        help="Excitatory neuron index to plot when --plot-single-neuron is set (default: 512)",
    )
    parser.add_argument(
        "--plot-pca",
        action="store_true",
        default=False,
        help="Save PCA scatter frames during training",
    )
    parser.add_argument(
        "--gif-pca",
        action="store_true",
        default=False,
        help="Assemble PCA scatter frames into a GIF after training (requires --plot-pca)",
    )
    parser.add_argument(
        "--track-stats",
        action="store_true",
        default=False,
        help="Enable neuron/synapse stat tracking and print diagnostics each val step",
    )
    parser.add_argument(
        "--stat-freq",
        type=int,
        default=10500,
        help="Stat tracking frequency in simulation timesteps (default: 10500)",
    )

    return parser.parse_args()


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = (
        f"{args.dataset}_{args.weight_type}"
        f"_{args.reg_type}_{args.reg_mode}"
        f"_vn{args.var_noise}_hz{args.max_rate_hz}_s{args.seed}"
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
        density_se=0.01,
        density_ee=0.03,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=2.0,
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
            sigma_x_lognormal_max=args.max_sigma_x,
            n_orientations=args.n_orientations,
            orientation_mode=args.orientation_mode,
            sigma_ee_mean=args.lognorm_ee_mean,
            sigma_ee_lognormal_std=args.lognorm_ee_std,
        )
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
        max_rate_hz=args.max_rate_hz,
        gain=1.0,
        gabor=args.gabor,
    )

    # --- Training ---
    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []
    stats_history = []
    t_start = time.time()

    # Capture generator so weights are built before we start iterating
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
        use_pca=True,
        pca_variance=15,
        track_stats=args.track_stats,
        stat_tracking_frequency=args.stat_freq,
        heatmap_plot=args.plot_spikes,
        PCA_plot=args.plot_pca,
        gif_pca_plot=args.gif_pca,
    )

    # Weights are built at this point — plot initial weight structure before training
    if args.plot_single_neuron:
        from neurosnn._plot.weights import plot_single_neuron_weights
        import numpy as np

        snn_model = model._runner.model
        w = snn_model.weights
        st, ex = snn_model.st, snn_model.ex
        H = W = int(np.sqrt(st))
        H_e = W_e = int(np.sqrt(ex - st))
        out_path = os.path.join(output_dir, f"single_neuron_{args.neuron_id}.pdf")
        plot_single_neuron_weights(
            w, st, ex, H, W, H_e, W_e, id_=args.neuron_id, out_path=out_path
        )
        print(f"Single-neuron weight plot saved -> {out_path}")

    if args.plot_rfs:
        from neurosnn._plot.weights import plot_oriented_rf_summary

        snn_model = model._runner.model
        W_se = snn_model.weights[: snn_model.st, snn_model.st : snn_model.ex]
        plot_oriented_rf_summary(
            W_se=W_se,
            input_size=snn_model.pixel_size,
            n_orientations=args.n_orientations,
            out_path=os.path.join(output_dir, "oriented_rf_summary.pdf"),
        )

    for result in train_gen:
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
            if args.track_stats and result.stats:
                s = result.stats
                print(
                    f"  mp_exc {s['mean_mp_exc']:.3f}  mp_inh {s['mean_mp_inh']:.3f}"
                    f"  I_syn_exc {s['mean_I_syn_exc']:.4f}  I_syn_inh {s['mean_I_syn_inh']:.4f}"
                    f"  adapt {s['mean_adaptation']:.4f}  thr {s['mean_spike_threshold']:.3f}"
                    f"  delta_w {s['mean_delta_w']:.6f}  x_pre {s['mean_x_pre']:.4f}"
                    f"  x_tar_se {s['mean_x_tar_se']:.4f}  x_tar_ee {s['mean_x_tar_ee']:.4f}"
                )
                stats_history.append({"batch": result.batch, **s})
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
        "stats_history": stats_history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()
