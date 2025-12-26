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


# Geomfig sleep comparison experiment
GEOMFIG_SLEEP_COMPARISON = {
    "name": "geomfig_sleep_comparison",
    "description": "Compare geomfig classification with and without sleep",
    "sleep_configs": [
        {
            "name": "no_sleep",
            "sleep": False,
            "sleep_ratio": 0.0,
        },
        {
            "name": "with_sleep",
            "sleep": True,
            "sleep_ratio": DEFAULT_TRAINING_PARAMS["sleep_ratio"],
        }
    ],
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"],
    },
    "data": {
        **DEFAULT_DATA_PARAMS,
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
        "noise_mean": GEOMFIG_PARAMS["noise_mean"],
        "jitter": GEOMFIG_PARAMS["jitter"],
        "jitter_amount": GEOMFIG_PARAMS["jitter_amount"],
    },
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


# Paper geomfig experiment (canonical for paper)
GEOMFIG_EXPERIMENT = {
    "name": "paper_geomfig",
    "description": "Geomfig classification experiment for paper",
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"],
    },
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "data": {
        **DEFAULT_DATA_PARAMS,
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
        "noise_mean": GEOMFIG_PARAMS["noise_mean"],
        "jitter": GEOMFIG_PARAMS["jitter"],
        "jitter_amount": GEOMFIG_PARAMS["jitter_amount"],
    },
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
    "paper_geomfig": GEOMFIG_EXPERIMENT,
    "geomfig_sleep_comparison": GEOMFIG_SLEEP_COMPARISON,
    "mnist_family": MNIST_FAMILY_EXPERIMENT,
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

