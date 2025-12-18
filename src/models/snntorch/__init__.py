"""
SNNTorch comparison model for benchmarking against SNN-sleepy.

This module contains:
- main.py: Training script for snntorch model
- orchestrate_to_excel.py: Orchestration script for running experiments
- reform_data.py: Data reformatting utilities

Usage:
    python -m src.snntorch_comparison.main --dataset MNIST --runs 5
    python -m src.snntorch_comparison.orchestrate_to_excel --datasets MNIST KMNIST FMNIST
"""

from pathlib import Path

# Module directory
SNNTORCH_DIR = Path(__file__).parent

# Standard file names
RESULTS_FILE = "Results_.xlsx"
RESULTS_JSON = "results/orchestrate_runs.json"


def get_results_path() -> Path:
    """Get path to Results_.xlsx."""
    return SNNTORCH_DIR / RESULTS_FILE


def get_results_json_path() -> Path:
    """Get path to results JSON."""
    return SNNTORCH_DIR / RESULTS_JSON



