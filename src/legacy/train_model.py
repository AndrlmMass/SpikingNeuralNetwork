from src.models.snn import snn_sleepy
from src.utils.platform import PLATFORM_NAME, IS_WINDOWS, safe_cpu_count
import argparse
import numpy as np
import random
import json
from datetime import datetime
import os
import cProfile
import pstats


def run_once(run_idx: int, total_runs: int, args, disable_plotting: bool = False):
    print(f"\n===== RUN {run_idx + 1}/{total_runs} =====")
    # set seeds for run-to-run variance control
    seed = 1 + run_idx
    np.random.seed(seed)
    random.seed(seed)

    # init class
    snn_N = (
        snn_sleepy(classes=[0, 1, 2, 3])
        if getattr(args, "dataset", None) and args.dataset.lower() == "geomfig"
        else snn_sleepy()
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
        img_tr, img_va, img_te = 6000, 100, 2000
        b_tr, b_va, b_te = 400, 100, 200
        force_recreate_flag = False
    snn_N.prepare_data(
        all_images_train=img_tr,
        batch_image_train=b_tr,
        all_images_test=img_te,
        batch_image_test=b_te,
        all_images_val=img_va,
        batch_image_val=b_va,
        add_breaks=False,
        num_steps=100,
        force_recreate=force_recreate_flag,
        noisy_data=False,
        gain=gain_for_dataset,
        noise_level=0.0,
        imageMNIST=(False if use_geomfig else True),
        dataset=(
            args.dataset if getattr(args, "dataset", None) else args.image_dataset
        ),
        geom_noise_var=getattr(args, "geom_noise_var", 0.02),
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
    snn_N.prepare_network()

    if getattr(args, "profile", False):
        pr = cProfile.Profile()
        pr.enable()
    snn_N.train_network(
        train_weights=True,
        noisy_potential=True,
        check_sleep_interval=35000,
        weight_decay_rate_exc=[0.99997],
        weight_decay_rate_inh=[0.99997],
        force_train=True,
        plot_spikes_train=False,
        plot_weights=False,
        plot_epoch_performance=False,
        plot_weights_per_epoch=bool(getattr(args, "plot_weights_per_epoch", False)),
        plot_spikes_per_epoch=bool(getattr(args, "plot_spikes_per_epoch", False)),
        weight_track_samples=int(getattr(args, "weight_track_samples", 32)),
        weight_track_interval=int(getattr(args, "weight_track_interval", 0)),
        sleep_synchronized=False,
        plot_top_response_test=False,
        plot_top_response_train=False,
        plot_tsne_during_training=False,
        tsne_plot_interval=1,
        use_validation_data=False,
        var_noise=float(getattr(args, "noise", 2.0)),
        max_weight_exc=25,
        min_weight_inh=-25,
        sleep=not args.no_sleep,
        sleep_mode=str(args.sleep_mode),
        tau_syn=30,
        narrow_top=0.2,
        A_minus=0.5,
        A_plus=0.5,
        tau_LTD=10,
        tau_LTP=10,
        learning_rate_exc=0.0005,
        learning_rate_inh=0.0005,
        accuracy_method="pca_lr",
        test_only=False,
        use_QDA=True,
        early_stopping=bool(args.early_stopping),
        early_stopping_patience_pct=0.2,
        sleep_ratio=float(args.sleep_rate),
        sleep_max_iters=int(args.sleep_max_iters),
        on_timeout=str(args.on_timeout),
        normalize_weights=bool(args.normalize_weights),
    )

    if getattr(args, "profile", False):
        pr.disable()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = args.profile_output or os.path.join(
            "results", f"profile_{ts}_sr{args.sleep_rate}_run{run_idx+1}.prof"
        )
        try:
            pr.dump_stats(profile_path)
            print(f"Profile saved to: {profile_path}")
            # Print top hotspots by cumulative time
            pstats.Stats(pr).sort_stats("cumtime").print_stats(20)
        except Exception as e:
            print(f"WARNING: could not write/print profile stats: {e}")

    if disable_plotting:
        result = snn_N.analyze_results(
            t_sne_test=False, t_sne_train=False, pca_train=False, pca_test=False
        )
    else:
        result = snn_N.analyze_results(t_sne_test=True)

    acc_dict = result[0] if isinstance(result, tuple) else result
    final_test_phi = getattr(snn_N, "final_test_phi", None)
    # Optional: plot accuracy histories at end
    try:
        if getattr(args, "plot_acc_history", False):
            # Create filename suffix with run info to avoid overwriting
            dataset_name = getattr(args, "dataset", "unknown")
            sleep_rate = getattr(args, "sleep_rate", "unknown")
            suffix = f"{dataset_name}_sr{sleep_rate}_run{run_idx+1}"
            snn_N.plot_accuracy_history(filename_suffix=suffix)
    except Exception as e:
        print(f"Warning: failed to plot accuracy history ({e})")
    return acc_dict, final_test_phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="number of repeated runs")
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
        "--on-timeout", type=str, default="give_up", help="action to take on timeout"
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
        default=False,
        help="disable sleep during training (default: sleep enabled)",
    )
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        default=False,
        help="enable per-group weight-sum normalization (may slow training)",
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
        "--dataset",
        type=str,
        nargs="+",
        choices=[
            "mnist",
            "kmnist",
            "fmnist",
            "notmnist",
            "geomfig",
        ],
        default=["mnist"],
        help="dataset(s) to use - can specify multiple (e.g., --dataset mnist fmnist geomfig)",
    )
    parser.add_argument(
        "--geom-noise-var",
        type=float,
        default=0.2,
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
        default=safe_cpu_count(),
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
        "--weight-track-samples",
        type=int,
        default=32,
        help="number of excitatory/inhibitory synapses to track in weight trajectory plots",
    )
    parser.add_argument(
        "--weight-track-interval",
        type=int,
        default=1,
        help="training timesteps between tracked weight snapshots (0 = auto based on sleep interval)",
    )
    parser.add_argument(
        "--weight-track-sleep-interval",
        type=int,
        default=0,
        help="sleep-iteration stride between tracked snapshots during hard-pause sleep (0 = auto)",
    )
    parser.add_argument(
        "--sleep-mode",
        type=str,
        choices=["static", "group", "post"],
        default="static",
        help="sleep target mode: static uses fixed targets; group uses group-mean magnitudes; post uses per-post mean magnitudes",
    )
    parser.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[2.0],
        help=(
            "noise variance(s) for Gaussian noise during training "
            "(can specify multiple, e.g., --noise 1.0 2.0 3.0)"
        ),
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows", "auto"],
        default="auto",
        help=f"platform mode for compatibility (auto-detected: {PLATFORM_NAME})",
    )
    args, _ = parser.parse_known_args()

    # Handle platform override
    if args.platform == "auto":
        args.platform = PLATFORM_NAME
    print(f"Running on platform: {args.platform}")

    # Ensure sleep_rate is a list
    sleep_rates = (
        args.sleep_rate if isinstance(args.sleep_rate, list) else [args.sleep_rate]
    )

    # Ensure noise is a list
    noise_levels = args.noise if isinstance(args.noise, list) else [args.noise]

    # Ensure dataset is a list
    datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]

    all_results = []  # Store (dataset, sleep_rate, noise_level, run_idx, result)
    disable_plotting = (
        args.runs > 1
        or len(sleep_rates) > 1
        or len(noise_levels) > 1
        or len(datasets) > 1
    )

    # Initialize results file at the start
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{output_dir}/results_{timestamp_str}.json"

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

    # Initial JSON structure
    initial_results_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "in_progress",
        "datasets": datasets,
        "args": {
            "runs": args.runs,
            "sleep_rates": [float(sr) for sr in sleep_rates],
            "noise_levels": [float(n) for n in noise_levels],
            "datasets": datasets,
            "sleep_max_iters": args.sleep_max_iters,
            "on_timeout": args.on_timeout,
            "early_stopping": args.early_stopping,
            "no_sleep": args.no_sleep,
            "normalize_weights": args.normalize_weights,
        },
        "results_by_dataset": {ds: {} for ds in datasets},
    }

    # Save initial file
    try:
        with open(results_filename, "w") as f:
            json.dump(initial_results_data, f, indent=2)
        print(f"Results will be saved incrementally to: {results_filename}")
    except Exception as e:
        print(f"WARNING: Could not initialize results file: {e}")
        results_filename = None

    # Function to save results incrementally
    def save_results_incremental(
        dataset_name, sleep_rate, noise_level, run_idx, result
    ):
        if results_filename is None:
            return
        try:
            try:
                with open(results_filename, "r") as f:
                    results_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results_data = initial_results_data.copy()

            # Use combined key for sleep_rate and noise_level
            config_key = f"sr{sleep_rate}_n{noise_level}"
            if isinstance(result, tuple) and len(result) >= 2:
                acc_dict, phi = result[0], result[1]
            elif isinstance(result, tuple) and len(result) == 1:
                acc_dict, phi = result[0], None
            else:
                acc_dict, phi = result, None

            result_entry = {
                "run": int(run_idx + 1),
                "sleep_rate": float(sleep_rate),
                "noise_level": float(noise_level),
                "dataset": dataset_name,
                "test_accuracy": (
                    safe_float(acc_dict.get("test"))
                    if isinstance(acc_dict, dict)
                    else None
                ),
                "train_accuracy": (
                    safe_float(acc_dict.get("train"))
                    if isinstance(acc_dict, dict)
                    else None
                ),
                "val_accuracy": (
                    safe_float(acc_dict.get("val"))
                    if isinstance(acc_dict, dict)
                    else None
                ),
                "final_test_phi": safe_float(phi),
            }

            if dataset_name not in results_data["results_by_dataset"]:
                results_data["results_by_dataset"][dataset_name] = {}
            if config_key not in results_data["results_by_dataset"][dataset_name]:
                results_data["results_by_dataset"][dataset_name][config_key] = []

            results_data["results_by_dataset"][dataset_name][config_key].append(
                result_entry
            )

            with open(results_filename, "w") as f:
                json.dump(results_data, f, indent=2)
        except Exception as e:
            print(f"WARNING: Could not save incremental results: {e}")

    # ... (existing code) ...

    # Collect results for DataFrame
    df_rows = []

    # Run for each dataset, then sleep rate, then runs
    preview_requested = bool(getattr(args, "preview_dataset", False))
    preview_consumed = False

    for dataset_idx, current_dataset in enumerate(datasets):
        print(f"\n{'#'*70}")
        print(
            f"# DATASET: {current_dataset.upper()} ({dataset_idx + 1}/{len(datasets)})"
        )
        print(f"{'#'*70}")

        for sleep_rate_idx, sleep_rate in enumerate(sleep_rates):
            print(f"\n{'='*70}")
            print(
                f"Dataset: {current_dataset} | Sleep Rate: {sleep_rate} ({sleep_rate_idx + 1}/{len(sleep_rates)})"
            )
            print(f"{'='*70}")

            for noise_idx, noise_level in enumerate(noise_levels):
                print(f"\n{'-'*70}")
                print(
                    f"Noise Level: {noise_level} ({noise_idx + 1}/{len(noise_levels)})"
                )
                print(f"{'-'*70}")

                for run_idx in range(args.runs):
                    args_copy = argparse.Namespace(**vars(args))
                    args_copy.dataset = (
                        current_dataset  # Set single dataset for this run
                    )
                    args_copy.sleep_rate = float(sleep_rate)
                    args_copy.noise = float(noise_level)
                    preview_this_run = preview_requested and not preview_consumed
                    args_copy.preview_dataset = preview_this_run
                    result = run_once(
                        run_idx, args.runs, args_copy, disable_plotting=disable_plotting
                    )
                    if preview_this_run:
                        preview_consumed = True
                    all_results.append(
                        (current_dataset, sleep_rate, noise_level, run_idx, result)
                    )
                    save_results_incremental(
                        current_dataset, sleep_rate, noise_level, run_idx, result
                    )

                    # Extract accuracy for DataFrame
                    if isinstance(result, tuple) and len(result) >= 1:
                        acc_dict = result[0]
                        test_acc = acc_dict.get("test") if isinstance(acc_dict, dict) else None
                    else:
                        test_acc = None

                    # Append to DataFrame rows
                    df_rows.append({
                        "Sleep_duration": float(sleep_rate) * 100,
                        "Model": "SNN_sleepy",
                        "Run": run_idx + 1,
                        "Seed": run_idx + 1,
                        "Dataset": current_dataset,
                        "Accuracy": test_acc
                    })

    if args.runs > 0 and len(all_results) > 0:
        # ... (existing summary print print logic) ...

        # ---- Save to Excel ----
        try:
            import pandas as pd
            df = pd.DataFrame(df_rows)
            
            # Ensure proper column order
            cols = ["Sleep_duration", "Model", "Run", "Seed", "Dataset", "Accuracy"]
            # Add any extra columns that might be present (none currently, but good practice)
            for c in df.columns:
                if c not in cols:
                    cols.append(c)
            df = df[cols]

            # Save to src/analysis/Results_.xlsx for consistency
            analysis_dir = os.path.join("src", "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            excel_path = os.path.join(analysis_dir, "Results_.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"\nResults saved to Excel: {excel_path}")
        except ImportError:
            print("\nWARNING: pandas/openpyxl not installed. Could not save to Excel.")
        except Exception as e:
            print(f"\nWARNING: Failed to save Excel results: {e}")

        # ... (existing JSON summary logic) ...

    if args.runs > 0 and len(all_results) > 0:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        # Group results by dataset, sleep rate, and noise level
        results_by_dataset = {}
        for dataset, sleep_rate, noise_level, run_idx, result in all_results:
            if dataset not in results_by_dataset:
                results_by_dataset[dataset] = {}
            config_key = (sleep_rate, noise_level)
            results_by_dataset[dataset].setdefault(config_key, []).append(
                (run_idx, result)
            )

        all_test_accs = []
        all_test_phis = []

        # Per-dataset, per-sleep-rate summary
        for dataset in datasets:
            if dataset not in results_by_dataset:
                continue

            print(f"\n{'#'*70}")
            print(f"# DATASET: {dataset.upper()}")
            print(f"{'#'*70}")

            dataset_results = results_by_dataset[dataset]

            for config_key in sorted(
                dataset_results.keys(), key=lambda x: (float(x[0]), float(x[1]))
            ):
                sleep_rate, noise_level = config_key
                results = dataset_results[config_key]
                valid_results = [
                    (run_idx, r)
                    for run_idx, r in results
                    if (r[0] is not None) or (len(r) > 1 and r[1] is not None)
                ]

                if not valid_results:
                    continue

                print(f"\n{'='*70}")
                print(
                    f"Dataset: {dataset} | Sleep Rate: {sleep_rate} | Noise: {noise_level}"
                )
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
            print("OVERALL SUMMARY (All Datasets & Sleep Rates)")
            print(f"{'='*70}")
            print(f"Total runs: {len(all_test_accs)}")

            try:
                best_overall_acc = max(all_test_accs)
                found = False
                for dataset in datasets:
                    if dataset not in results_by_dataset:
                        continue
                    for config_key in sorted(
                        results_by_dataset[dataset].keys(),
                        key=lambda x: (float(x[0]), float(x[1])),
                    ):
                        sleep_rate, noise_level = config_key
                        results = results_by_dataset[dataset][config_key]
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
                                    f"Best overall test accuracy: {dataset}, Sleep Rate {sleep_rate}, "
                                    f"Noise {noise_level}, Run {run_idx+1} ({best_overall_acc:.4f})"
                                )
                                found = True
                                break
                        if found:
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

        if results_filename and os.path.exists(results_filename):
            try:
                with open(results_filename, "r") as f:
                    results_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"WARNING: Could not read existing results file: {e}")
                results_data = initial_results_data.copy()
        else:
            results_data = initial_results_data.copy()

        all_test_accs_from_file = []
        all_test_phis_from_file = []

        # Aggregate stats from all datasets and sleep rates
        for dataset_key, sleep_rate_dict in results_data.get(
            "results_by_dataset", {}
        ).items():
            for sleep_rate_key, results_list in sleep_rate_dict.items():
                for result_entry in results_list:
                    if result_entry.get("test_accuracy") is not None:
                        all_test_accs_from_file.append(result_entry["test_accuracy"])
                    if result_entry.get("final_test_phi") is not None:
                        all_test_phis_from_file.append(result_entry["final_test_phi"])

        results_data["summary"] = {
            "total_runs": len(all_test_accs_from_file),
            "overall_mean_test_accuracy": safe_stat_mean(all_test_accs_from_file),
            "overall_std_test_accuracy": safe_stat_std(all_test_accs_from_file),
            "overall_mean_final_test_phi": safe_stat_mean(all_test_phis_from_file),
            "overall_std_final_test_phi": safe_stat_std(all_test_phis_from_file),
            "per_dataset_summary": {},
        }

        for dataset_key, config_dict in results_data.get(
            "results_by_dataset", {}
        ).items():
            results_data["summary"]["per_dataset_summary"][dataset_key] = {}
            # Sort by extracting sleep_rate and noise from config key (format: "sr{X}_n{Y}")
            for config_key in sorted(
                config_dict.keys(),
                key=lambda x: (float(x.split("_")[0][2:]), float(x.split("_")[1][1:])),
            ):
                results_list = config_dict[config_key]
                test_accs = [
                    r["test_accuracy"]
                    for r in results_list
                    if r.get("test_accuracy") is not None
                ]
                test_phis = [
                    r["final_test_phi"]
                    for r in results_list
                    if r.get("final_test_phi") is not None
                ]

                if test_accs or test_phis:
                    results_data["summary"]["per_dataset_summary"][dataset_key][
                        config_key
                    ] = {
                        "mean_test_accuracy": safe_stat_mean(test_accs),
                        "std_test_accuracy": safe_stat_std(test_accs),
                        "mean_final_test_phi": safe_stat_mean(test_phis),
                        "std_final_test_phi": safe_stat_std(test_phis),
                        "best_test_accuracy": safe_stat_max(test_accs),
                        "best_final_test_phi": safe_stat_max(test_phis),
                    }

        results_data["status"] = "completed"
        results_data["completed_timestamp"] = datetime.now().isoformat()

        try:
            with open(results_filename, "w") as f:
                json.dump(results_data, f, indent=2)
            print(f"\nFinal results with summary saved to: {results_filename}")
        except Exception as e:
            print(f"\nWARNING: Failed to save final results: {e}")
            print("Results were computed but could not be saved.")


if __name__ == "__main__":
    main()
