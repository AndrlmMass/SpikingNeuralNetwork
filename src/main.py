from big_comb import snn_sleepy
import argparse
import numpy as np
import random
import json
from datetime import datetime
import os
import cProfile
import pstats
import pandas as pd


def run_once(
    run_idx: int,
    total_runs: int,
    args,
    disable_plotting: bool = False,
):
    print(f"\n===== RUN {run_idx + 1}/{total_runs} =====")
    # set seeds for run-to-run variance control
    seed = 42 + run_idx
    np.random.seed(seed)
    random.seed(seed)
    run_id = np.random.randint(0, 100000)

    if args.dataset.lower() == "geomfig":
        classes = [0, 1, 2, 3]
        num_input = 225
        use_validation_data = True
        w_dense_se = 0.1
        tau_m = 30
        tau_syn = 30
        Rm = 30
        max_rate_hz = 67.0
    elif args.dataset.lower() == "fcx1":
        classes = [0, 1]
        num_input = 100
        use_validation_data = False
        w_dense_se = 0.5
        tau_m = 1
        tau_syn = 1
        Rm = 30
        max_rate_hz = 67.0
    else:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_input = 784
        use_validation_data = True
        w_dense_se = 0.05
        w_dense_ee = 0.025
        w_dense_ei = 0.05
        w_dense_ie = 0.05
        se_weights = 4.0
        ee_weights = 2.0
        ei_weights = 4.0
        ie_weights = -2.0
        tau_syn_exc = 10
        tau_syn_inh = 9
        learning_rate_exc = 0.0002  # slightly reduced
        tau_m_exc = 20
        tau_m_inh = 15
        Rm_exc = 15
        Rm_inh = 17.5
        max_rate_hz = 250.0
        delta_adaption = 0.5
        tau_trace = 20
        tau_adaption = 200
        w_max = 10
        num_steps = 350
        x_tar = 0.5
        mu_weight = 0.6

    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S")

    # init class
    snn_N = snn_sleepy(
        classes=classes,
        random_state=seed,
        N_x=num_input,
        N_exc=1024,
        N_inh=225,
        ts_spec=ts_spec,
    )

    # acquire data
    use_geomfig = getattr(args, "dataset", None) and args.dataset.lower() == "geomfig"
    is_test_mode = bool(getattr(args, "test_mode", False))
    # Use a lower gain for geomfig to reduce input drive
    gain_for_dataset = float(getattr(args, "geom_gain", 0.5)) if use_geomfig else 1.0
    if is_test_mode and use_geomfig:
        img_tr, img_va, img_te = 4, 4, 4
        b_tr, b_va, b_te = 4, 4, 4
        force_recreate_flag = True
    else:
        img_tr, img_va, img_te = 30000, 250, 5000
        b_tr, b_va, b_te = 1000, 250, 500
        force_recreate_flag = False
    snn_N.prepare_data(
        all_audio_train=22000,
        batch_audio_train=100,
        all_audio_test=7800,
        batch_audio_test=600,
        all_audio_val=200,
        batch_audio_val=200,
        num_steps=num_steps,
        all_images_train=img_tr,
        batch_image_train=b_tr,
        all_images_test=img_te,
        batch_image_test=b_te,
        all_images_val=img_va,
        batch_image_val=b_va,
        use_validation_data=use_validation_data,
        add_breaks=False,
        force_recreate=force_recreate_flag,
        noisy_data=False,
        gain=gain_for_dataset,
        noise_level=0.0,
        max_rate_hz=max_rate_hz,
        audioMNIST=False,
        imageMNIST=(False if use_geomfig else True),
        create_data=False,
        plot_spectrograms=False,
        image_dataset=(
            args.dataset if getattr(args, "dataset", None) else args.image_dataset
        ),
        geom_noise_var=getattr(args, "geom_noise_var", 0.02),
        geom_noise_mean=getattr(args, "geom_noise_mean", 0.0),
        geom_jitter=getattr(args, "geom_jitter", False),
        geom_jitter_amount=getattr(args, "geom_jitter_amount", 0.05),
        geom_workers=int(
            getattr(args, "geom_workers", max(1, (os.cpu_count() or 2) - 1))
        ),
    )

    if run_idx == 0 and getattr(args, "preview_dataset", False):
        if disable_plotting:
            print(
                "Preview requested via --preview-dataset; displaying dataset sample "
                "before batch runs."
            )
        snn_N.preview_loaded_data(num_image_samples=1)

    # set up network for training
    snn_N.prepare_network(
        plot_weights=False,
        w_dense_ee=w_dense_ee,
        w_dense_se=w_dense_se,
        w_dense_ei=w_dense_ei,
        w_dense_ie=w_dense_ie,
        se_weights=se_weights,
        ee_weights=ee_weights,
        ei_weights=ei_weights,
        ie_weights=ie_weights,
        create_network=False,
        random_weights=args.random_weights,
    )

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    snn_N.train_network(
        train_weights=not args.no_train,
        noisy_potential=True,
        compare_decay_rates=False,
        check_sleep_interval=35000,
        weight_decay_rate_exc=[0.99997],
        weight_decay_rate_inh=[0.99997],
        max_weight_exc=25,
        min_weight_inh=-25,
        samples=10,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        tau_m_exc=tau_m_exc,
        tau_m_inh=tau_m_inh,
        membrane_resistance_exc=Rm_exc,
        membrane_resistance_inh=Rm_inh,
        force_train=True,
        plot_spikes_train=False,
        delta_adaption=delta_adaption,
        tau_trace=tau_trace,
        mu_weight=mu_weight,
        plot_weights=False,
        plot_epoch_performance=False,
        plot_weights_per_epoch=args.plot_weights_per_epoch,
        plot_spikes_per_epoch=args.plot_spikes_per_epoch,
        sleep_synchronized=False,
        plot_top_response_test=False,
        plot_top_response_train=False,
        plot_tsne_during_training=False,
        heatmap_plot=args.heatmap_plot,
        tsne_plot_interval=1,
        plot_spectrograms=False,
        use_validation_data=False,
        var_noise=args.noise_level,
        x_tar=x_tar,
        track_weights=args.track_weights,
        sleep=not args.no_sleep,
        sleep_mode=args.sleep_mode,
        narrow_top=0.2,
        track_stats=args.track_stats,
        run=run_id,
        A_minus=0.95,
        A_plus=1.0,
        tau_LTD=20,
        tau_LTP=20,
        tau_adaption=tau_adaption,
        w_max=w_max,
        learning_rate_exc=learning_rate_exc,
        learning_rate_inh=0.0005,
        accuracy_method="pca_lr",
        test_only=False,
        use_QDA=False,
        early_stopping=args.early_stopping,
        early_stopping_patience_pct=0.3,
        sleep_ratio=args.sleep_rate,
        sleep_max_iters=args.sleep_max_iters,
        on_timeout=args.on_timeout,
        normalize_weights=args.normalize_weights,
    )
    if args.profile:
        pr.disable()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        td = datetime.now().strftime("%Y%m%d")
        directory = os.path.join("profile", f"{args.image_dataset}", td)
        os.makedirs(directory, exist_ok=True)
        profile_path = os.path.join(directory, f"profile_{ts}_run{run_idx+1}.prof")
        try:
            pr.dump_stats(profile_path)
            print(f"Profile saved to: {profile_path}")
            # Print top hotspots by cumulative time
            pstats.Stats(pr).sort_stats("cumtime").print_stats(20)
        except Exception as e:
            print(f"WARNING: could not write/print profile stats: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="number of repeated runs")
    parser.add_argument(
        "--no-train", action="store_true", default=False, help="disable training"
    )
    parser.add_argument(
        "--sleep-rate",
        type=float,
        nargs="+",
        default=[0.1],
        help=(
            "sleep rate(s) during training "
            "(can specify multiple, e.g., --sleep-rate 0.5 0.6 0.7)"
        ),
    )
    parser.add_argument(
        "--sleep-max-iters",
        type=int,
        default=10000,
        help="maximum number of iterations to sleep",
    )
    parser.add_argument(
        "--on_timeout", type=str, default="give_up", help="action to take on timeout"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="enable early stopping",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        default=True,
        help="disable sleep during training (default: sleep enabled)",
    )
    parser.add_argument(
        "--no_normalize-weights",
        action="store_true",
        default=False,
        help="enable per-group weight-sum normalization (may slow training)",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        nargs="+",
        default=[0.1],
        help=(
            "noise level during training "
            "(can specify multiple, e.g., --sleep-rate 0.5 0.6 0.7)"
        ),
    )
    parser.add_argument(
        "--preview-dataset",
        action="store_true",
        default=False,
        help="plot a quick sample of the selected dataset before training",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="create minimal dataset (geomfig: 4 per split) and force recreate",
    )
    parser.add_argument(
        "--track-weights",
        action="store_true",
        default=False,
        help="track weights during training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "kmnist",
            "fmnist",
            "fashionmnist",
            "fashion",
            "notmnist",
            "geomfig",
            "fcx1",
        ],
        default="mnist",
        help="dataset to use (image-only or geomfig)",
    )
    parser.add_argument(
        "--image-dataset",
        type=str,
        choices=[
            "mnist",
            "kmnist",
            "fmnist",
            "fashionmnist",
            "fashion",
            "notmnist",
            "fcx1",
        ],
        default="mnist",
        help="image dataset to use for image-only or multimodal modes",
    )
    parser.add_argument(
        "--heatmap-plot",
        action="store_true",
        default=False,
        help="plot the heatmap of the weights",
    )
    parser.add_argument(
        "--get-giffed",
        action="store_true",
        default=False,
        help="create gif from heatmap plots",
    )
    parser.add_argument(
        "--geom-noise-var",
        type=float,
        default=0.02,
        help="per-pixel Gaussian noise variance for geomfig generation",
    )
    parser.add_argument(
        "--geom-noise-mean",
        type=float,
        default=0.0,
        help="noise mean offset for geomfig generation",
    )
    parser.add_argument(
        "--geom-jitter",
        action="store_true",
        default=False,
        help="enable random jitter of size/thickness for geomfig (disabled by default)",
    )
    parser.add_argument(
        "--geom-jitter-amount",
        type=float,
        default=0.05,
        help="relative jitter amount for geomfig size/thickness (0.05 = ±5%)",
    )
    parser.add_argument(
        "--geom-gain",
        type=float,
        default=0.5,
        help="rate-coding gain for geomfig (multiplies pixel intensities before Poisson sampling)",
    )
    parser.add_argument(
        "--geom-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="number of parallel workers for geomfig generation (process-based)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="enable cProfile around training to find hotspots",
    )
    parser.add_argument(
        "--plot-acc-history",
        action="store_true",
        default=False,
        help="plot accuracy history at end (train+val together, test separate)",
    )
    parser.add_argument(
        "--plot-weights-per-epoch",
        action="store_true",
        default=False,
        help="plot and save weights after each epoch (for debugging, saves to plots/weights_epoch_*.png)",
    )
    parser.add_argument(
        "--plot-spikes-per-epoch",
        action="store_true",
        default=False,
        help="plot and save spikes after each epoch (for debugging, saves to plots/spikes_*_epoch_*.png)",
    )
    parser.add_argument(
        "--sleep-mode",
        type=str,
        choices=["static", "group", "post"],
        default="static",
        help="sleep target mode: static uses fixed targets; group uses group-mean magnitudes; post uses per-post mean magnitudes",
    )
    parser.add_argument(
        "--track-excel",
        action="store_true",
        default=False,
        help="track results in GLM/Results_.xlsx file after each run",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        default=False,
        help="use random weights (0.01 for ee, 0.05 for se, 0.05 for ei, 0.05 for ie)",
    )
    parser.add_argument(
        "--track-stats",
        action="store_true",
        default=False,
        help="track statistics during training",
    )
    args, _ = parser.parse_known_args()

    # Ensure sleep_rate is a list
    sleep_rates = (
        args.sleep_rate if isinstance(args.sleep_rate, list) else [args.sleep_rate]
    )

    noise_levels = (
        args.noise_level if isinstance(args.noise_level, list) else [args.noise_level]
    )

    all_results = []  # Store (sleep_rate, run_idx, result)
    disable_plotting = args.runs > 1 or len(sleep_rates) > 1

    # Helper function to safely convert values
    def safe_float(val):
        if val is None:
            return None
        try:
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            return None
        except (ValueError, TypeError):
            return None

    # Excel tracking functions
    excel_path = "GLM/Results_.xlsx"
    model_name = "SNN_sleepy"
    lambda_value = 0.99997

    def get_next_run_number():
        """Get the next run number for the model from existing Excel file."""
        if not os.path.exists(excel_path):
            return 1
        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
            if "Model" not in df.columns or "Run" not in df.columns:
                return 1
            model_runs = df[df["Model"] == model_name]
            if model_runs.empty:
                return 1
            return int(model_runs["Run"].max()) + 1
        except Exception as e:
            print(f"WARNING: Could not read Excel file to determine run number: {e}")
            return 1

    def save_to_excel(sleep_rate, noise_level, run_idx, result, run_number):
        """Append a new row to the Excel file after a run completes."""
        if not getattr(args, "track_excel", False):
            return
        try:
            if isinstance(result, tuple) and len(result) >= 1:
                acc_dict = result[0]
            else:
                acc_dict = result

            test_accuracy = (
                safe_float(acc_dict.get("test")) if isinstance(acc_dict, dict) else None
            )
            if test_accuracy is None:
                print("WARNING: Test accuracy is None, skipping Excel update")
                return

            seed = 1 + run_idx
            sleep_duration = float(sleep_rate)
            dataset_name_val = (
                args.dataset if getattr(args, "dataset", None) else args.image_dataset
            )

            new_row = {
                "Sleep_duration": sleep_duration,
                "Noise_level": noise_level,
                "Model": model_name,
                "Run": run_number,
                "Lambda": lambda_value,
                "Seed": seed,
                "Dataset": dataset_name_val,
                "Accuracy": test_accuracy,
            }

            if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
                try:
                    df = pd.read_excel(excel_path, engine="openpyxl")
                    required_columns = [
                        "Sleep_duration",
                        "Model",
                        "Run",
                        "Lambda",
                        "Seed",
                        "Dataset",
                        "Accuracy",
                    ]
                    if df.empty or not all(
                        col in df.columns for col in required_columns
                    ):
                        df = pd.DataFrame(columns=required_columns)
                except Exception:
                    df = pd.DataFrame(
                        columns=[
                            "Sleep_duration",
                            "Model",
                            "Run",
                            "Lambda",
                            "Seed",
                            "Dataset",
                            "Accuracy",
                        ]
                    )
            else:
                df = pd.DataFrame(
                    columns=[
                        "Sleep_duration",
                        "Model",
                        "Run",
                        "Lambda",
                        "Seed",
                        "Dataset",
                        "Accuracy",
                    ]
                )

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            df.to_excel(excel_path, index=False, engine="openpyxl")
            print(f"Results saved to {excel_path}")
        except Exception as e:
            print(f"WARNING: Could not save to Excel file: {e}")

    # Initial JSON structure
    dataset_name = (
        args.dataset if getattr(args, "dataset", None) else args.image_dataset
    )

    # Run for each sleep rate
    preview_requested = bool(getattr(args, "preview_dataset", False))
    preview_consumed = False

    # Determine run number for Excel tracking (same for all runs in this execution)
    excel_run_number = (
        get_next_run_number() if getattr(args, "track_excel", False) else None
    )

    for sleep_rate_idx, sleep_rate in enumerate(sleep_rates):
        print(f"\n{'='*70}")
        print(f"Sleep Rate: {sleep_rate} ({sleep_rate_idx + 1}/{len(sleep_rates)})")
        print(f"{'='*70}")

        for noise_level_idx, noise_level in enumerate(noise_levels):
            print(f"\n{'='*70}")
            print(
                f"Noise levels: {noise_level} ({noise_level_idx + 1}/{len(noise_levels)})"
            )
            print(f"{'='*70}")

            for run_idx in range(args.runs):
                args_copy = argparse.Namespace(**vars(args))
                args_copy.sleep_rate = float(sleep_rate)
                args_copy.noise_level = float(noise_level)
                preview_this_run = preview_requested and not preview_consumed
                args_copy.preview_dataset = preview_this_run
                result = run_once(
                    run_idx,
                    args.runs,
                    args_copy,
                    disable_plotting=disable_plotting,
                )
                if preview_this_run:
                    preview_consumed = True
                all_results.append((sleep_rate, noise_level, run_idx, result))

        if args.runs > 0 and len(all_results) > 0:
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)

            # Group results by sleep rate
            results_by_sleep_rate = {}
            for sleep_rate, run_idx, result in all_results:
                results_by_sleep_rate.setdefault(sleep_rate, []).append(
                    (run_idx, result)
                )

            # Group results by noise level
            results_by_noise_level = {}
            for noise_level, run_idx, result in all_results:
                results_by_noise_level.setdefault(noise_level, []).append(
                    (run_idx, result)
                )

            all_test_accs = []
            all_test_phis = []

            # Per-sleep-rate summary
            for sleep_rate in sorted(
                results_by_sleep_rate.keys(), key=lambda x: float(x)
            ):
                results = results_by_sleep_rate[sleep_rate]
                valid_results = [
                    (run_idx, r)
                    for run_idx, r in results
                    if (r[0] is not None) or (len(r) > 1 and r[1] is not None)
                ]

                if not valid_results:
                    continue

                print(f"\n{'='*70}")
                print(f"Sleep Rate: {sleep_rate}")
                print(f"{'='*70}")
                print(f"\n{'Run':<6} {'Test Accuracy':<15} {'Final Test Phi':<16}")
                print("-" * 70)

                test_accs = []
                test_phis = []

                for run_idx, (acc_dict, phi) in valid_results:
                    test_acc = (
                        acc_dict.get("test", "N/A")
                        if isinstance(acc_dict, dict)
                        else "N/A"
                    )
                    phi_str = (
                        f"{phi:.4f}"
                        if (phi is not None and isinstance(phi, (int, float)))
                        else "N/A"
                    )

                    if isinstance(test_acc, (int, float)):
                        test_accs.append(test_acc)
                        all_test_accs.append(test_acc)
                        print(f"{run_idx+1:<6} {test_acc:<15.4f} {phi_str:<16}")
                    else:
                        print(f"{run_idx+1:<6} {test_acc:<15} {phi_str:<16}")

                    if phi is not None and isinstance(phi, (int, float)):
                        test_phis.append(phi)
                        all_test_phis.append(phi)
                    else:
                        test_phis.append(float("nan"))

                if test_accs:
                    print("-" * 70)
                    # Best / mean acc
                    try:
                        best_acc_idx = max(
                            range(len(test_accs)), key=lambda i: test_accs[i]
                        )
                        best_run = valid_results[best_acc_idx][0] + 1
                        print(
                            f"Best test accuracy: Run {best_run} ({test_accs[best_acc_idx]:.4f})"
                        )
                    except Exception as e:
                        print(f"Could not determine best accuracy: {e}")

                    try:
                        mean_acc = float(np.mean(test_accs))
                        std_acc = float(np.std(test_accs))
                        print(f"Mean test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                    except Exception as e:
                        print(f"Could not compute accuracy stats: {e}")

                    # Phi stats (only valid numbers)
                    valid_phis_only = [
                        p
                        for p in test_phis
                        if isinstance(p, (int, float)) and not np.isnan(p)
                    ]
                    if valid_phis_only:
                        try:
                            mean_phi = float(np.mean(valid_phis_only))
                            std_phi = float(np.std(valid_phis_only))
                            print(
                                f"Mean final test phi: {mean_phi:.4f} ± {std_phi:.4f}"
                            )
                        except Exception as e:
                            print(f"Could not compute phi stats: {e}")

            # Overall summary
            if all_test_accs:
                print(f"\n{'='*70}")
                print("OVERALL SUMMARY (All Sleep Rates)")
                print(f"{'='*70}")
                print(f"Total runs: {len(all_test_accs)}")

                try:
                    best_overall_acc = max(all_test_accs)
                    found = False
                    for sleep_rate in sorted(
                        results_by_sleep_rate.keys(), key=lambda x: float(x)
                    ):
                        results = results_by_sleep_rate[sleep_rate]
                        valid_results = [
                            (run_idx, r)
                            for run_idx, r in results
                            if (r[0] is not None)
                            and isinstance(r[0], dict)
                            and isinstance(r[0].get("test"), (int, float))
                        ]
                        for run_idx, (acc_dict, phi) in valid_results:
                            test_acc = acc_dict.get("test")
                            if (
                                test_acc is not None
                                and abs(float(test_acc) - float(best_overall_acc))
                                < 1e-6
                            ):
                                print(
                                    f"Best overall test accuracy: Sleep Rate {sleep_rate}, "
                                    f"Run {run_idx+1} ({best_overall_acc:.4f})"
                                )
                                found = True
                                break
                        if found:
                            break
                except Exception as e:
                    print(f"Could not determine best overall accuracy: {e}")

                try:
                    mean_acc = float(np.mean(all_test_accs))
                    std_acc = float(np.std(all_test_accs))
                    print(f"Overall mean test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                except Exception as e:
                    print(f"Could not compute overall accuracy stats: {e}")

                if all_test_phis:
                    try:
                        mean_phi = float(np.mean(all_test_phis))
                        std_phi = float(np.std(all_test_phis))
                        print(
                            f"Overall mean final test phi: {mean_phi:.4f} ± {std_phi:.4f}"
                        )
                    except Exception as e:
                        print(f"Could not compute overall phi stats: {e}")

        print("=" * 70)

        # ---- Write summary back to JSON ----

        def safe_stat_mean(values):
            try:
                return float(np.mean(values)) if values else None
            except (ValueError, TypeError):
                return None

        def safe_stat_std(values):
            try:
                return float(np.std(values)) if values else None
            except (ValueError, TypeError):
                return None

        def safe_stat_max(values):
            try:
                return float(max(values)) if values else None
            except (ValueError, TypeError):
                return None


if __name__ == "__main__":
    main()
