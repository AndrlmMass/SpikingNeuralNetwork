"""
Standalone RF inspection script — builds oriented SE weights and saves diagnostic plots.

No model or dataset required; runs in a few seconds.

Usage
-----
# Default: oriented RFs, 4 orientations, save to results/inspect_rfs/
python experiments/generic_testing/inspect_rfs.py

# Tune elongation and orientations
python experiments/generic_testing/inspect_rfs.py \
    --sigma-x 4.0 --gamma 0.3 --n-orientations 8

# Compare isotropic (gamma=1) vs elongated
python experiments/generic_testing/inspect_rfs.py --gamma 1.0 --out inspect_isotropic

# Log-normal size diversity
python experiments/generic_testing/inspect_rfs.py --lognormal-std 0.4
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from neurosnn._network.init_weights import oriented_gaussian_se_weights
from neurosnn._plot.weights import plot_oriented_rf_summary


def parse_args():
    p = argparse.ArgumentParser(description="Inspect oriented SE receptive fields")

    # RF geometry
    p.add_argument("--sigma-x", type=float, default=3.0,
                   help="Major-axis sigma of each RF Gaussian (default: 3.0)")
    p.add_argument("--gamma", type=float, default=0.4,
                   help="Aspect ratio sigma_y/sigma_x; 1.0 = isotropic (default: 0.4)")
    p.add_argument("--n-orientations", type=int, default=4,
                   help="Number of orientation groups cycling across E neurons (default: 4)")
    p.add_argument("--r-cut-factor", type=float, default=3.0,
                   help="Hard cutoff at r_cut_factor * sigma_x (default: 3.0)")
    p.add_argument("--lognormal-std", type=float, default=0.0,
                   help="Log-normal sigma_x diversity; 0 = uniform size (default: 0.0)")
    p.add_argument("--peak", type=float, default=1.0,
                   help="Per-neuron weight peak after normalisation (default: 1.0)")

    # Network size
    p.add_argument("--n-exc", type=int, default=1024,
                   help="Number of excitatory neurons (default: 1024)")
    p.add_argument("--input-size", type=int, default=28,
                   help="Square input side length in pixels (default: 28)")

    # Display
    p.add_argument("--n-show", type=int, default=256,
                   help="Neurons shown in tile grid (default: 256)")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="Output directory (default: results/inspect_rfs/<tag>)")

    return p.parse_args()


def build_output_dir(args) -> str:
    if args.out is not None:
        return args.out
    base = os.path.join(os.path.dirname(__file__), "results", "inspect_rfs")
    tag = (
        f"sx{args.sigma_x}_g{args.gamma}"
        f"_ori{args.n_orientations}_rc{args.r_cut_factor}"
        f"_ln{args.lognormal_std}_s{args.seed}"
    )
    return os.path.join(base, tag)


def main():
    args = parse_args()
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    N_x = args.input_size ** 2

    print(
        f"\nBuilding oriented RFs — "
        f"sigma_x={args.sigma_x}  gamma={args.gamma}  "
        f"n_orientations={args.n_orientations}  "
        f"r_cut={args.r_cut_factor}  lognormal_std={args.lognormal_std}\n"
    )

    W_se = oriented_gaussian_se_weights(
        N_x=N_x,
        N_exc=args.n_exc,
        input_size=args.input_size,
        sigma_x=args.sigma_x,
        gamma=args.gamma,
        n_orientations=args.n_orientations,
        r_cut_factor=args.r_cut_factor,
        sigma_x_lognormal_std=args.lognormal_std,
        peak=args.peak,
        rng=rng,
    )

    out_path = os.path.join(output_dir, "oriented_rf_summary.pdf")
    plot_oriented_rf_summary(
        W_se=W_se,
        input_size=args.input_size,
        n_orientations=args.n_orientations,
        out_path=out_path,
        n_show=args.n_show,
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
