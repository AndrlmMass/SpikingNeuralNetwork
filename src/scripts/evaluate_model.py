#!/usr/bin/env python
"""
Evaluate a trained model.

This script loads saved weights and runs evaluation on test data.

Usage:
    python -m src.scripts.evaluate_model --model-dir model/[1 2 2 1 0]
"""

import sys
import os
import argparse
import numpy as np

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing weights.npy",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["geomfig", "mnist", "kmnist", "fmnist" "notmnist"],
        default="mnist",
        help="Dataset to evaluate on",
    )
    args = parser.parse_args()
    
    # Check if model exists
    weights_path = os.path.join(args.model_dir, "weights.npy")
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        sys.exit(1)
    
    print(f"Loading weights from {weights_path}")
    weights = np.load(weights_path)
    print(f"Weights shape: {weights.shape}")
    
    # TODO: Implement full evaluation pipeline
    print("\nEvaluation functionality to be implemented.")
    print("For now, use the main.py with --test-only flag.")


if __name__ == "__main__":
    main()





