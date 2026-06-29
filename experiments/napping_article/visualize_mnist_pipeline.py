"""
Visualize one MNIST digit through the three pipeline stages.

Usage
-----
    python experiments/visualize_mnist_pipeline.py [--digit 0] [--out-dir results/data/MNIST]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurosnn._data.get_data import ImageDataStreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join("results", "data", "MNIST"))
    args = parser.parse_args()

    ds = ImageDataStreamer(data_dir="data", dataset="mnist", pixel_size=15)
    ds.preview_pipeline(digit_idx=args.digit, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
