"""
Geomfig experiment script.

Supports two modes:
1. simple_geomfig: Run a single geomfig experiment (with or without sleep)
2. sleep_comparison_geomfig: Compare geomfig classification with and without sleep
"""

import sys
from pathlib import Path

from sympy.logic import false

# Add src to path (go up one level from experiments/ to src/)
src_root = Path(__file__).parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from ..config.experiment_configs import GEOMFIG_EXPERIMENT, GEOMFIG_SLEEP_COMPARISON, QUICK_GEOMFIG
from ..config.defaults import DEFAULT_TRAINING_PARAMS
from ..models.SNN_sleepy.snn import snn_sleepy


def simple_geomfig(quick=False):
    """Run a single geomfig experiment.
    
    Args:
        quick: If True, use quick test configuration (small dataset)
    
    Returns:
        snn_sleepy: Trained SNN model instance
    """
    # Choose experiment config
    if quick:
        print("=" * 60)
        print("Running QUICK TEST experiment (small dataset)")
        print("=" * 60)
        exp_config = QUICK_GEOMFIG
    else:
        print("=" * 60)
        print("Running PAPER GEOMFIG experiment")
        print("=" * 60)
        exp_config = GEOMFIG_EXPERIMENT
    
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
        create_network=True,
        w_dense_ee=network_params.get("w_dense_ee", 0.15),
        w_dense_se=network_params.get("w_dense_se", 0.1),
        w_dense_ei=network_params.get("w_dense_ei", 0.2),
        w_dense_ie=network_params.get("w_dense_ie", 0.25),
        se_weights=network_params.get("se_weights", 0.15),
        ee_weights=network_params.get("ee_weights", 0.3),
        ei_weights=network_params.get("ei_weights", 0.3),
        ie_weights=network_params.get("ie_weights", -0.3),
        spike_threshold_default=network_params.get("spike_threshold_default", -55),
        resting_membrane=network_params.get("resting_potential", -70),
    )
    
    # Train network
    print("\n" + "=" * 60)
    print("Training network...")
    print("=" * 60)
    
    # Calculate batch sizes
    batch_size = data_params.get("batch_image_train", 400)
    val_batch_size = data_params.get("batch_image_val", 100)
    test_batch_size = data_params.get("batch_image_test", 200)
    
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
                       "resting_potential", "spike_threshold_default", "check_sleep_interval", 
                       "timing_update", "trace_update", "vectorized_trace"]}
    )
    
    # Run training
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60)
    
    snn.train(
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


def sleep_comparison_geomfig(quick=False, preview_data=False):
    """Run geomfig sleep comparison experiment.
    
    Compares geomfig classification with and without sleep.
    
    Args:
        quick: If True, use smaller dataset for quick testing
        preview_data: If True, show preview plots of loaded data
    
    Returns:
        dict: Results dictionary with accuracy for each configuration
    """
    exp_config = GEOMFIG_SLEEP_COMPARISON
    
    print("=" * 60)
    print(f"Running {exp_config['name']}")
    print(f"Description: {exp_config.get('description', 'N/A')}")
    print("=" * 60)
    
    # Override data params for quick test if requested
    if quick:
        exp_config = exp_config.copy()
        exp_config["data"] = {
            **exp_config["data"],
            "all_images_train": 100,
            "batch_image_train": 50,
            "all_images_test": 50,
            "batch_image_test": 50,
            "all_images_val": 50,
            "batch_image_val": 50,
        }
        print("\n⚠️  Using quick test configuration (small dataset)")
    
    # Extract config sections
    network_params = exp_config["network"]
    data_params = exp_config["data"]
    sleep_configs = exp_config["sleep_configs"]
    
    print(f"\nNetwork params: {network_params}")
    print(f"\nData params: {data_params}")
    print(f"\nSleep configurations to compare: {len(sleep_configs)}")
    for i, sleep_cfg in enumerate(sleep_configs):
        print(f"  {i+1}. {sleep_cfg['name']}: sleep={sleep_cfg['sleep']}, sleep_ratio={sleep_cfg.get('sleep_ratio', 0.0)}")
    
    results = {}
    
    # Run each sleep configuration
    for sleep_cfg in sleep_configs:
        config_name = sleep_cfg["name"]
        print("\n" + "=" * 60)
        print(f"Running configuration: {config_name}")
        print("=" * 60)
        
        # Create SNN instance
        print("\nCreating SNN instance...")
        snn = snn_sleepy(
            N_exc=network_params["N_exc"],
            N_inh=network_params["N_inh"],
            N_x=network_params["N_x"],
            seed=1,
            which_classes=network_params["classes"],
        )
        
        # Prepare data
        print("\nPreparing data...")
        all_train = data_params.get("all_images_train", 6000)
        all_val = data_params.get("all_images_val", 100)
        all_test = data_params.get("all_images_test", 1000)
        total_data = all_train + all_val + all_test
        
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
            preview_data=preview_data,
        )
        
        # Prepare network
        print("\nPreparing network...")
        snn.prepare_network(
            create_network=True,
            w_dense_ee=network_params.get("w_dense_ee", 0.15),
            w_dense_se=network_params.get("w_dense_se", 0.1),
            w_dense_ei=network_params.get("w_dense_ei", 0.2),
            w_dense_ie=network_params.get("w_dense_ie", 0.25),
            se_weights=network_params.get("se_weights", 0.15),
            ee_weights=network_params.get("ee_weights", 0.3),
            ei_weights=network_params.get("ei_weights", 0.3),
            ie_weights=network_params.get("ie_weights", -0.3),
            spike_threshold_default=network_params.get("spike_threshold_default", -55),
            resting_membrane=network_params.get("resting_potential", -70),
        )
        
        # Train network with this sleep configuration
        print("\nTraining network...")
        batch_size = data_params.get("batch_image_train", 400)
        val_batch_size = data_params.get("batch_image_val", 100)
        test_batch_size = data_params.get("batch_image_test", 200)
        
        # Use default training params but override sleep settings
        training_params = DEFAULT_TRAINING_PARAMS.copy()
        training_params["sleep"] = sleep_cfg["sleep"]
        training_params["sleep_ratio"] = sleep_cfg.get("sleep_ratio", 0.0)
        
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
            force_train=True,
            epochs=1,
            save_model=True,
            **{k: v for k, v in training_params.items() 
               if k not in ["train_weights", "learning_rate_exc", "learning_rate_inh", 
                           "sleep", "sleep_ratio", "sleep_mode", "accuracy_method", "pca_variance",
                           "resting_potential", "spike_threshold_default", "check_sleep_interval", 
                           "timing_update", "trace_update", "vectorized_trace", "epochs"]}
        )
        
        # Run training
        print(f"\nStarting training for {config_name}...")
        snn.train(
            batch_size=batch_size,
            val_batch_size=val_batch_size,
        )
        
        # Store results
        results[config_name] = {}
        if hasattr(snn, 'performance_tracker'):
            if snn.performance_tracker.get("train_accuracy"):
                results[config_name]["train_accuracy"] = snn.performance_tracker['train_accuracy'][-1]
            if snn.performance_tracker.get("val_accuracy"):
                results[config_name]["val_accuracy"] = snn.performance_tracker['val_accuracy'][-1]
            if snn.performance_tracker.get("test_accuracy"):
                results[config_name]["test_accuracy"] = snn.performance_tracker['test_accuracy'][-1]
        
        print(f"\n{config_name} complete!")
    
    # Print comparison results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        if "train_accuracy" in result:
            print(f"  Train Accuracy: {result['train_accuracy']:.4f}")
        if "val_accuracy" in result:
            print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
        if "test_accuracy" in result:
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run geomfig experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simple geomfig experiment
  python -m experiments.geomfig simple_geomfig
  
  # Run sleep comparison experiment
  python -m experiments.geomfig sleep_comparison_geomfig
  
  # Run quick test (small dataset)
  python -m experiments.geomfig simple_geomfig --quick
  python -m experiments.geomfig sleep_comparison_geomfig --quick
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["simple_geomfig", "sleep_comparison_geomfig"],
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick test configuration (small dataset, faster)"
    )
    parser.add_argument(
        "--preview-data",
        action="store_true",
        help="Show preview plots of loaded data"
    )
    
    args = parser.parse_args()
    
    if args.mode == "simple_geomfig":
        snn = simple_geomfig(quick=args.quick)
        print("\n✅ Simple geomfig experiment completed successfully!")
    elif args.mode == "sleep_comparison_geomfig":
        results = sleep_comparison_geomfig(quick=args.quick, preview_data=args.preview_data)
        print("\n✅ Sleep comparison experiment completed successfully!")
