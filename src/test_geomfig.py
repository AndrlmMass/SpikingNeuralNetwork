"""
Test script for running geomfig experiment using config setup.
This will test the full training pipeline end-to-end.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import PAPER_GEOMFIG_EXPERIMENT, QUICK_TEST_EXPERIMENT
from models.SNN_sleepy.snn import snn_sleepy

def run_geomfig_test(quick=False):
    """Run geomfig experiment test."""
    
    # Choose experiment config
    if quick:
        print("=" * 60)
        print("Running QUICK TEST experiment (small dataset)")
        print("=" * 60)
        exp_config = QUICK_TEST_EXPERIMENT
    else:
        print("=" * 60)
        print("Running PAPER GEOMFIG experiment")
        print("=" * 60)
        exp_config = PAPER_GEOMFIG_EXPERIMENT
    
    # Extract config sections
    network_params = exp_config["network"]
    training_params = exp_config.get("training", {})
    data_params = exp_config["data"]
    
    print(f"\nExperiment: {exp_config['name']}")
    print(f"Description: {exp_config.get('description', 'N/A')}")
    print(f"\nNetwork params: {network_params}")
    print(f"\nTraining params: {training_params}")
    print(f"\nData params: {data_params}")
    
    # Create SNN instance
    print("\n" + "=" * 60)
    print("Creating SNN instance...")
    print("=" * 60)
    
    snn = snn_sleepy(
        N_exc=network_params["N_exc"],
        N_inh=network_params["N_inh"],
        N_x=network_params["N_x"],
        seed=1,
        which_classes=network_params["classes"],
    )
    
    # Prepare data
    print("\n" + "=" * 60)
    print("Preparing data...")
    print("=" * 60)
    
    # Calculate total_data from train/val/test splits
    all_train = data_params.get("all_images_train", 6000)
    all_val = data_params.get("all_images_val", 100)
    all_test = data_params.get("all_images_test", 1000)
    total_data = all_train + all_val + all_test
    
    # Calculate splits as ratios
    train_split = all_train / total_data
    val_split = all_val / total_data
    test_split = all_test / total_data
    
    snn.prepare_data(
        dataset="geomfig",
        total_data=total_data,
        num_steps=data_params.get("num_steps", 100),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        batch_size=data_params.get("batch_image_train", 400),
        gain=data_params.get("gain", 0.5),
        geom_jitter=data_params.get("jitter", False),
        geom_jitter_amount=data_params.get("jitter_amount", 0.05),
        geom_noise_var=data_params.get("noise_var", 0.2),
        force_recreate=False,
    )
    
    # Prepare network
    print("\n" + "=" * 60)
    print("Preparing network...")
    print("=" * 60)
    
    snn.prepare_network(
        w_dense_ee=network_params.get("w_dense_ee", 0.15),
        w_dense_se=network_params.get("w_dense_se", 0.1),
        w_dense_ei=network_params.get("w_dense_ei", 0.2),
        w_dense_ie=network_params.get("w_dense_ie", 0.25),
        se_weights=network_params.get("se_weights", 0.15),
        ee_weights=network_params.get("ee_weights", 0.3),
        ei_weights=network_params.get("ei_weights", 0.3),
        ie_weights=network_params.get("ie_weights", -0.3),
    )
    
    # Train network
    print("\n" + "=" * 60)
    print("Training network...")
    print("=" * 60)
    
    # Calculate batch sizes and epochs from data params
    batch_size = data_params.get("batch_image_train", 400)
    val_batch_size = data_params.get("batch_image_val", 100)
    test_batch_size = data_params.get("batch_image_test", 200)
    epochs = max(1, data_params.get("all_images_train", 6000) // batch_size)
    
    # Set training parameters
    snn.train_network(
        train_weights=training_params.get("train_weights", True),
        learning_rate_exc=training_params.get("learning_rate_exc", 0.0005),
        learning_rate_inh=training_params.get("learning_rate_inh", 0.0005),
        sleep=training_params.get("sleep", True),
        sleep_ratio=training_params.get("sleep_ratio", 0.2),
        sleep_mode=training_params.get("sleep_mode", "static"),
        accuracy_method=training_params.get("accuracy_method", "pca_lr"),
        pca_variance=training_params.get("pca_variance", 0.95),
        use_validation_data=True,
        test_batch_size=test_batch_size,
        save_checkpoints=True,
        checkpoint_frequency="epoch",
        keep_checkpoints=3,
        force_train=True,  # Force training even if model exists
        save_model=True,
        # Add other training params from config (exclude invalid parameters)
        **{k: v for k, v in training_params.items() 
           if k not in ["train_weights", "learning_rate_exc", "learning_rate_inh", 
                       "sleep", "sleep_ratio", "sleep_mode", "accuracy_method", "pca_variance",
                       "resting_potential", "check_sleep_interval", "timing_update", 
                       "trace_update", "vectorized_trace"]}
    )
    
    # Run training
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60)
    
    snn.train(
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Print final results
    if hasattr(snn, 'performance_tracker'):
        print("\nFinal Performance:")
        if snn.performance_tracker.get("train_accuracy"):
            print(f"  Train Accuracy: {snn.performance_tracker['train_accuracy'][-1]:.4f}")
        if snn.performance_tracker.get("val_accuracy"):
            print(f"  Val Accuracy: {snn.performance_tracker['val_accuracy'][-1]:.4f}")
        if snn.performance_tracker.get("test_accuracy"):
            print(f"  Test Accuracy: {snn.performance_tracker['test_accuracy'][-1]:.4f}")
    
    return snn


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test geomfig experiment")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with small dataset (faster)"
    )
    
    args = parser.parse_args()
    

    snn = run_geomfig_test(quick=args.quick)
    print("\n✅ Test completed successfully!")


