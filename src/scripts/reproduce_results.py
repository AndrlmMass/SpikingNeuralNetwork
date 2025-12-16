#!/usr/bin/env python
"""
Reproduce paper results.

This script runs the experiments described in the paper with the exact
parameters used to generate the reported results.

Usage:
    python -m src.scripts.reproduce_results --experiment all
    python -m src.scripts.reproduce_results --experiment geomfig
"""

import sys
import os
import argparse

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from models import snn_sleepy
from config.defaults import DEFAULT_TRAINING_PARAMS, GEOMFIG_PARAMS


def run_geomfig_experiment():
    """Run the geometric figures classification experiment."""
    print("=" * 70)
    print("Running Geomfig Classification Experiment")
    print("=" * 70)
    
    # Initialize network with 4 classes
    snn = snn_sleepy(classes=[0, 1, 2, 3])
    
    # Prepare data
    snn.prepare_data(
        all_images_train=6000,
        batch_image_train=400,
        all_images_test=1000,
        batch_image_test=200,
        all_images_val=100,
        batch_image_val=100,
        image_dataset="geomfig",
        gain=GEOMFIG_PARAMS["gain"],
        geom_noise_var=GEOMFIG_PARAMS["noise_var"],
    )
    
    # Prepare network
    snn.prepare_network(
        plot_weights=False,
        w_dense_ee=0.15,
        w_dense_se=0.1,
        w_dense_ei=0.2,
        w_dense_ie=0.25,
    )
    
    # Train
    snn.train_network(
        train_weights=True,
        sleep=True,
        sleep_ratio=0.1,
        accuracy_method="pca_lr",
        **{k: v for k, v in DEFAULT_TRAINING_PARAMS.items() 
           if k not in ['sleep', 'sleep_ratio', 'accuracy_method']}
    )
    
    # Analyze
    results = snn.analyze_results(t_sne_test=True)
    
    print("\n" + "=" * 70)
    print("Geomfig Experiment Complete")
    print("=" * 70)
    
    return results


def run_mnist_experiment():
    """Run the MNIST classification experiment."""
    print("=" * 70)
    print("Running MNIST Classification Experiment")
    print("=" * 70)
    
    snn = snn_sleepy()
    
    snn.prepare_data(
        all_images_train=6000,
        batch_image_train=400,
        all_images_test=1000,
        batch_image_test=200,
        all_images_val=100,
        batch_image_val=100,
        image_dataset="mnist",
    )
    
    snn.prepare_network(
        plot_weights=False,
        w_dense_ee=0.15,
        w_dense_se=0.1,
        w_dense_ei=0.2,
        w_dense_ie=0.25,
    )
    
    snn.train_network(
        train_weights=True,
        sleep=True,
        sleep_ratio=0.1,
        accuracy_method="pca_lr",
    )
    
    results = snn.analyze_results(t_sne_test=True)
    
    print("\n" + "=" * 70)
    print("MNIST Experiment Complete")
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "geomfig", "mnist"],
        default="geomfig",
        help="Which experiment to run",
    )
    args = parser.parse_args()
    
    if args.experiment == "all":
        run_geomfig_experiment()
        run_mnist_experiment()
    elif args.experiment == "geomfig":
        run_geomfig_experiment()
    elif args.experiment == "mnist":
        run_mnist_experiment()


if __name__ == "__main__":
    main()





