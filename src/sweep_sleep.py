from big_comb import snn_sleepy


def run_sweep_for_modality(name: str, imageMNIST: bool, audioMNIST: bool):
    print(f"\n=== Running sleep-rate sweep: {name} ===")

    LAMBDAS = [0.1, 0.5, 0.7, 0.9, 0.99, 0.999]

    snn = snn_sleepy()

    # Use generated data path (non-streaming) and no validation to align with current create_data API
    snn.prepare_data(
        imageMNIST=imageMNIST,
        audioMNIST=audioMNIST,
        create_data=True,
        use_validation_data=False,
        plot_spectrograms=False,
    )

    snn.prepare_network(create_network=False)

    results = snn.sweep_sleep_rates(
        sleep_rates=LAMBDAS,
        train_params=dict(
            train_weights=True,
            plot_epoch_performance=False,
            compare_decay_rates=False,
            accuracy_method="top",
        ),
        reinit_network_each_rate=True,
    )

    for r in results:
        print(
            f"{name:12s} lambda={r['lambda']}: acc_top={r['accuracy_top']}, phi={r['phi_test']}"
        )

    return [{**r, "modality": name} for r in results]


def main():
    all_results = []

    # Image only
    all_results += run_sweep_for_modality("imageMNIST", imageMNIST=True, audioMNIST=False)

    # Audio only (requires AudioMNIST dataset at path set in get_data.py)
    all_results += run_sweep_for_modality("audioMNIST", imageMNIST=False, audioMNIST=True)

    # Image + Audio
    all_results += run_sweep_for_modality("image+audio", imageMNIST=True, audioMNIST=True)

    # Optional: write CSV summary next to project root
    try:
        import csv

        with open("sleep_sweep_results.csv", "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["modality", "lambda", "accuracy_top", "phi_test"]
            )
            w.writeheader()
            w.writerows(all_results)
        print("\nSaved results to sleep_sweep_results.csv")
    except Exception as e:
        print(f"Warning: could not save CSV ({e})")


if __name__ == "__main__":
    main()


