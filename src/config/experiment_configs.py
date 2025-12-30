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
    MNIST_DATASETS,
    SLEEP_RATES,
    SEEDS,
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
        "classes": GEOMFIG_PARAMS["classes"].copy(),  # Copy list to prevent mutation
    },
    "data": {
        **DEFAULT_DATA_PARAMS,
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
        "noise_mean": GEOMFIG_PARAMS["noise_mean"],
        "jitter": GEOMFIG_PARAMS["jitter"],
        "jitter_amount": GEOMFIG_PARAMS["jitter_amount"],
    },
    "plot": {
        "boxplot": True,
        "preview_data": True,
    },
}

# Paper geomfig experiment (canonical for paper)
GEOMFIG_EXPERIMENT = {
    "name": "paper_geomfig",
    "description": "Geomfig classification experiment for paper",
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"].copy(),  # Copy list to prevent mutation
    },
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "datasets": ["geomfig"],
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
QUICK_GEOMFIG = {
    "name": "paper_geomfig",
    "description": "Geomfig classification experiment for paper",
    "network": {
        **DEFAULT_NETWORK_PARAMS,
        "classes": GEOMFIG_PARAMS["classes"].copy(),  # Copy list to prevent mutation
    },
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "datasets": ["geomfig"],
    "data": {
        **DEFAULT_DATA_PARAMS,
        "gain": GEOMFIG_PARAMS["gain"],
        "noise_var": GEOMFIG_PARAMS["noise_var"],
        "noise_mean": GEOMFIG_PARAMS["noise_mean"],
        "jitter": GEOMFIG_PARAMS["jitter"],
        "jitter_amount": GEOMFIG_PARAMS["jitter_amount"],
    },
}

QUICK_MNIST = {
    "name": "quick_mnist",
    "description": "Minimal configuration for quick testing (mnist, small dataset)",
    "datasets": MNIST_DATASETS.copy(),  # Copy list to prevent mutation
    "network": DEFAULT_NETWORK_PARAMS.copy(),
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "data": DEFAULT_DATA_PARAMS.copy(),
}


# MNIST Family experiment (canonical for paper)
MNIST_FAMILY_EXPERIMENT = {
    "name": "mnist_family",
    "description": "Compare sleep rates across MNIST-family datasets",
    "datasets": MNIST_DATASETS.copy(),  # Copy list to prevent mutation
    "sleep_rates": SLEEP_RATES.copy(),  # Copy list to prevent mutation
    "sleep_mode": "static",
    "seeds": SEEDS.copy(),  # Copy list to prevent mutation
    "network": DEFAULT_NETWORK_PARAMS.copy(),
    "training": DEFAULT_TRAINING_PARAMS.copy(),
    "data": DEFAULT_DATA_PARAMS.copy(),
}


# All experiments for easy access
ALL_EXPERIMENTS = {
    "paper_geomfig": GEOMFIG_EXPERIMENT,
    "geomfig_sleep_comparison": GEOMFIG_SLEEP_COMPARISON,
    "mnist_family": MNIST_FAMILY_EXPERIMENT,
    "quick_geomfig": QUICK_GEOMFIG,
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

