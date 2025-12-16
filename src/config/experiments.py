"""
Pre-defined experiment configurations.

These configurations can be used to reproduce specific experiments
or as starting points for new experiments.
"""

from .defaults import (
    DEFAULT_NETWORK_PARAMS,
    DEFAULT_TRAINING_PARAMS,
    DEFAULT_DATA_PARAMS,
    GEOMFIG_PARAMS,
)


# Paper reproduction experiment
PAPER_GEOMFIG_EXPERIMENT = {
    "name": "paper_geomfig",
    "description": "Geometric figures classification as described in the paper",
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"],  # Override: 4 classes for geomfig
    },
    "training": {
        **DEFAULT_TRAINING_PARAMS,
        # Only override what's different (defaults already have sleep=True, sleep_ratio=0.2)
    },
    "data": {
        **DEFAULT_DATA_PARAMS,
        # Override only geomfig-specific params (defaults handle the rest)
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
        "noise_mean": GEOMFIG_PARAMS["noise_mean"],
        "jitter": GEOMFIG_PARAMS["jitter"],
        "jitter_amount": GEOMFIG_PARAMS["jitter_amount"],
    },
}


# MNIST baseline experiment
MNIST_BASELINE_EXPERIMENT = {
    "name": "mnist_baseline",
    "description": "Standard MNIST classification with sleep",
    "network": DEFAULT_NETWORK_PARAMS.copy(),  # Use all defaults (10 classes)
    "training": DEFAULT_TRAINING_PARAMS.copy(),  # Use all defaults
    "data": DEFAULT_DATA_PARAMS.copy(),  # Use all defaults
}


# No-sleep baseline for comparison
NO_SLEEP_EXPERIMENT = {
    "name": "no_sleep_baseline",
    "description": "Baseline without sleep mechanism for comparison",
    "network": DEFAULT_NETWORK_PARAMS.copy(),
    "training": {
        **DEFAULT_TRAINING_PARAMS,
        "sleep": False,
        "sleep_ratio": 0.0,
    },
    "data": DEFAULT_DATA_PARAMS.copy(),
}


# Sleep rate comparison experiment
SLEEP_RATE_COMPARISON = {
    "name": "sleep_rate_comparison",
    "description": "Compare different sleep rates",
    "sleep_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "network": DEFAULT_NETWORK_PARAMS.copy(),
    "training_base": DEFAULT_TRAINING_PARAMS.copy(),
    "data": DEFAULT_DATA_PARAMS.copy(),
}


# Quick test configuration (minimal data for debugging)
QUICK_TEST_EXPERIMENT = {
    "name": "quick_test",
    "description": "Minimal configuration for quick testing (geomfig, small dataset)",
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"],  # 4 classes for geomfig
    },
    "training": DEFAULT_TRAINING_PARAMS.copy(),  # Use defaults
    "data": {
        **DEFAULT_DATA_PARAMS,
        # Override only what's different for quick test
        "all_images_train": 100,
        "batch_image_train": 50,
        "all_images_test": 50,
        "batch_image_test": 50,
        "all_images_val": 50,
        "batch_image_val": 50,
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
    },
}


# MNIST Family experiment (canonical for paper)
MNIST_FAMILY_EXPERIMENT = {
    "name": "mnist_family",
    "description": "Compare sleep rates across MNIST-family datasets",
    "datasets": ["mnist", "kmnist", "fmnist", "notmnist"],
    "sleep_rates": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "sleep_mode": "static",
    "seeds": [1, 2, 3, 4, 5],
    "network": DEFAULT_NETWORK_PARAMS.copy(),
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "data": DEFAULT_DATA_PARAMS.copy(),
}


# All experiments for easy access
ALL_EXPERIMENTS = {
    "paper_geomfig": PAPER_GEOMFIG_EXPERIMENT,
    "mnist_baseline": MNIST_BASELINE_EXPERIMENT,
    "mnist_family": MNIST_FAMILY_EXPERIMENT,
    "no_sleep": NO_SLEEP_EXPERIMENT,
    "sleep_comparison": SLEEP_RATE_COMPARISON,
    "quick_test": QUICK_TEST_EXPERIMENT,
}


def get_experiment(name: str) -> dict:
    """
    Get experiment configuration by name.
    
    Parameters
    ----------
    name : str
        Experiment name (one of: paper_geomfig, mnist_baseline, 
        no_sleep, sleep_comparison, quick_test)
        
    Returns
    -------
    dict
        Experiment configuration dictionary
    """
    if name not in ALL_EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available: {list(ALL_EXPERIMENTS.keys())}"
        )
    return ALL_EXPERIMENTS[name]

