import numpy as np
import json
import os

from big_comb import snn_sleepy


def run_sleep_rate_comparison():
    """
    Run real experiments across sleep rates and data modes, collecting accuracy and phi.
    """

    sleep_rates = [0.1, 0.5, 0.7, 0.9, 0.99, 0.999]
    data_modes = [
        {"name": "imageMNIST", "audioMNIST": False, "imageMNIST": True},
        {"name": "audioMNIST", "audioMNIST": True, "imageMNIST": False},
        {"name": "image+audio", "audioMNIST": True, "imageMNIST": True},
    ]

    results = {}
    os.environ.setdefault(
        "AUDIO_MNIST_PATH",
        r"C:\\Users\\Andreas\\Documents\\GitHub\\AudioMNIST\\data",
    )

    for mode in data_modes:
        mode_name = mode["name"]
        results[mode_name] = {}
        for lam in sleep_rates:
            print(f"\n=== {mode_name} | sleep_rate={lam} ===")
            try:
                snn = snn_sleepy()

                snn.prepare_data(
                    all_audio_train=6000,
                    batch_audio_train=500,
                    all_audio_test=1000,
                    batch_audio_test=200,
                    all_audio_val=1000,
                    batch_audio_val=100,
                    all_images_train=6000,
                    batch_image_train=500,
                    all_images_test=1000,
                    batch_image_test=200,
                    all_images_val=1000,
                    batch_image_val=100,
                    add_breaks=False,
                    force_recreate=True,
                    noisy_data=True,
                    gain=1.0,
                    noise_level=0.01,
                    audioMNIST=mode["audioMNIST"],
                    imageMNIST=mode["imageMNIST"],
                    create_data=True,
                    plot_spectrograms=False,
                    use_validation_data=False,
                )

                snn.prepare_network(
                    plot_weights=False,
                    w_dense_ee=0.05,
                    w_dense_se=0.1,
                    w_dense_ei=0.1,
                    w_dense_ie=0.1,
                    se_weights=0.2,
                    ee_weights=0.05,
                    ei_weights=0.4,
                    ie_weights=-0.8,
                    create_network=False,
                )

                snn.train_network(
                    train_weights=True,
                    noisy_potential=True,
                    compare_decay_rates=False,
                    check_sleep_interval=10000,
                    weight_decay=True,
                    weight_decay_rate_exc=[lam],
                    weight_decay_rate_inh=[lam],
                    samples=1,
                    force_train=True,
                    plot_spikes_train=False,
                    plot_weights=False,
                    plot_epoch_performance=False,
                    sleep_synchronized=False,
                    plot_top_response_test=False,
                    plot_top_response_train=False,
                    plot_tsne_during_training=False,
                    plot_spectrograms=False,
                    use_validation_data=False,
                    var_noise=2,
                    tau_syn=30,
                    narrow_top=0.2,
                    A_minus=0.2,
                    A_plus=0.5,
                    tau_LTD=7.5,
                    tau_LTP=10,
                    learning_rate_exc=0.0008,
                    learning_rate_inh=0.005,
                )

                if hasattr(snn, "performance_tracker") and snn.performance_tracker is not None:
                    phi_ = float(snn.performance_tracker[-1, 0])
                    acc_ = float(snn.performance_tracker[-1, 1])
                    results[mode_name][f"lambda_{lam}"] = {"accuracy": acc_, "phi": phi_}
                    print(f"acc={acc_:.3f} phi={phi_:.3f}")
                else:
                    results[mode_name][f"lambda_{lam}"] = {"accuracy": None, "phi": None}
                    print("No performance tracker available")
            except Exception as e:
                results[mode_name][f"lambda_{lam}"] = {
                    "accuracy": None,
                    "phi": None,
                    "error": str(e),
                }
                print(f"Error: {e}")

    return results

def save_results(results, filename="sleep_rate_comparison_results.json"):
    """Save results to a JSON file."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    filepath = os.path.join("results", filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")

def create_production_script():
    """Create a production-ready script for actual experiments."""
    production_code = '''"""
Production Sleep Rate Comparison Script

To run this, you need:
1. Install dependencies: pip install librosa snntorch torch torchvision numpy matplotlib scikit-learn tqdm
2. Download AudioMNIST dataset to /home/andreas/Documents/GitHub/AudioMNIST/data
3. Run: python sleep_rate_comparison_production.py
"""

from big_comb import snn_sleepy
import numpy as np
import json
import os

def run_production_comparison():
    sleep_rates = [0.1, 0.5, 0.7, 0.9, 0.99, 0.999]
    data_configs = [
        {"name": "imageMNIST", "audioMNIST": False, "imageMNIST": True},
        {"name": "audioMNIST", "audioMNIST": True, "imageMNIST": False},
        {"name": "image+audio", "audioMNIST": True, "imageMNIST": True}
    ]

    results = {}

    for config in data_configs:
        results[config["name"]] = {}

        for sleep_rate in sleep_rates:
            print(f"\\n=== Training {config['name']} with sleep_rate = {sleep_rate} ===")

            try:
                # Initialize and configure network
                snn_N = snn_sleepy()

                # Prepare data
                snn_N.prepare_data(
                    all_audio_train=6000, batch_audio_train=500,
                    all_audio_test=1000, batch_audio_test=200,
                    all_images_train=6000, batch_image_train=500,
                    all_images_test=1000, batch_image_test=200,
                    add_breaks=False, force_recreate=True, noisy_data=True,
                    gain=1.0, noise_level=0.01,
                    audioMNIST=config["audioMNIST"], imageMNIST=config["imageMNIST"],
                    create_data=False, plot_spectrograms=False,
                )

                # Prepare network
                snn_N.prepare_network(
                    plot_weights=False, w_dense_ee=0.05, w_dense_se=0.1,
                    w_dense_ei=0.1, w_dense_ie=0.1, se_weights=0.2,
                    ee_weights=0.05, ei_weights=0.4, ie_weights=-0.8,
                    create_network=False,
                )

                # Train with specific sleep rate
                snn_N.train_network(
                    train_weights=True, noisy_potential=True, compare_decay_rates=False,
                    check_sleep_interval=10000,
                    weight_decay_rate_exc=[sleep_rate], weight_decay_rate_inh=[sleep_rate],
                    samples=10, force_train=True, plot_spikes_train=False,
                    plot_weights=False, plot_epoch_performance=False,
                    sleep_synchronized=False, plot_top_response_test=False,
                    plot_top_response_train=False, plot_tsne_during_training=False,
                    plot_spectrograms=False, use_validation_data=False,
                    var_noise=2, tau_syn=30, narrow_top=0.2, A_minus=0.2, A_plus=0.5,
                    tau_LTD=7.5, tau_LTP=10, learning_rate_exc=0.0008, learning_rate_inh=0.005,
                )

                # Extract final metrics
                if hasattr(snn_N, 'performance_tracker') and snn_N.performance_tracker is not None:
                    final_phi = float(snn_N.performance_tracker[-1, 0])
                    final_accuracy = float(snn_N.performance_tracker[-1, 1])

                    results[config["name"]][f"λ_{sleep_rate}"] = {
                        "accuracy": final_accuracy, "phi": final_phi
                    }
                    print(f"Accuracy: {final_accuracy:.3f}, Phi: {final_phi:.3f}")
                else:
                    results[config["name"]][f"λ_{sleep_rate}"] = {"accuracy": None, "phi": None}

            except Exception as e:
                print(f"Error: {e}")
                results[config["name"]][f"λ_{sleep_rate}"] = {"accuracy": None, "phi": None, "error": str(e)}

    return results

if __name__ == "__main__":
    results = run_production_comparison()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/sleep_rate_comparison_production.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\\nProduction experiment completed!")
'''

    with open("sleep_rate_comparison_production.py", "w") as f:
        f.write(production_code)

    print("Production script created: src/sleep_rate_comparison_production.py")

if __name__ == "__main__":
    print("Sleep Rate Comparison - DEMONSTRATION MODE")
    print("=" * 60)
    print("This script shows example output format.")
    print("For actual experiments, use the production script.")
    print()

    # Run demonstration
    results = run_sleep_rate_comparison()

    # Save demonstration results
    save_results(results, "sleep_rate_comparison_demo.json")

    # Create production script (commented out due to Unicode issues)
    # create_production_script()

    print("\\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("To run actual experiments:")
    print("1. Install dependencies: pip install librosa snntorch torch torchvision")
    print("2. Download AudioMNIST dataset")
    print("3. Run: python src/sleep_rate_comparison_production.py")
