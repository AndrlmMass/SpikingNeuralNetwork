"""
Network-size sweep: scale the excitatory/inhibitory populations together while keeping
the *influence of inhibition proportional* across sizes.

The network config is the generic_testing/run_test.py default (oriented_rf weights,
engaged-inhibition regime). The only swept variables are the layer sizes and the
random seed:

    N_exc : 1024 (32^2)  ->  2116 (46^2)  ->  4096 (64^2)
    N_inh :  225         ->   450         ->   900

oriented_rf requires N_exc to be an even perfect square (perfect-square assertion in
oriented_rf_assignment, and block orientation_mode with n_orientations=4 -> a 2x2 grid
needs an even side), so the 2x cell is 46^2=2116 (closest even-square to 2048) rather
than 2048. N_inh has no square constraint (the inhibitory grid is laid out flexibly).

Proportional inhibition
-----------------------
Both inhibitory pathways are normalised per row by ``weight_compliance`` to a total of
``frac * N_exc * peak`` (see neurosnn/_network/init_weights.py). That makes the raw
inhibitory drive grow linearly with population size, so a fixed peak would make
inhibition progressively stronger in the larger nets. To hold the per-neuron inhibitory
influence constant we scale both inhibitory peaks down with size:

    inh_scale  = N_INH_BASE / N_inh        # 225 / N_inh -> 1.0, 0.5, 0.25
    peak_ei_eff = PEAK_EI_BASE * inh_scale  # E->I  (drive INTO inhibition)
    peak_ie_eff = PEAK_IE_BASE * inh_scale  # I->E  (inhibition BACK onto E)

inh_scale is keyed to the inhibitory population (exactly 1x/2x/4x), so it gives the
clean 1/2 and 1/4 the rule specifies. The I->E pathway (the actual influence of
inhibition onto E) scales with N_inh, so this is the principled denominator. At the
maximum 4x size (4096/900) this is exactly 1/4, matching the hand-tuned rule.

Usage
-----
# Local smoke-test (small/fast)
python experiments/RF_article/RF_netw_size_sweep/run_experiment.py \
    --n-exc 1024 --n-inh 225 --seed 0 --train-all 1000 --val-all 200 --test-all 500

# Full MNIST cell (used by the SLURM array)
python experiments/RF_article/RF_netw_size_sweep/run_experiment.py \
    --n-exc 4096 --n-inh 900 --seed 1 \
    --output-dir results/exc4096_inh900_s1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import neurosnn as snn

# Inhibition scaling is referenced to this base inhibitory population.
N_INH_BASE = 225

# Base peaks / densities — generic_testing/run_test.py engaged-inhibition regime.
PEAK_SE_BASE = 4.0
PEAK_EE_BASE = 1.0
PEAK_EI_BASE = 2.0
PEAK_IE_BASE = -2.0
DENSITY_SE = 0.01
DENSITY_EE = 0.01
DENSITY_EI = 0.03
DENSITY_IE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNN network-size sweep (proportional inhibition across sizes)"
    )

    # --- Sweep parameters ---
    parser.add_argument(
        "--n-exc",
        type=int,
        default=1024,
        help="Excitatory population size (must be an even perfect square for oriented_rf)",
    )
    parser.add_argument(
        "--n-inh",
        type=int,
        default=225,
        help="Inhibitory population size",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # --- Dataset / training control (RF_article sweep convention) ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "notmnist", "geomfig", "fcx1"],
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--train-all", type=int, default=59000)
    parser.add_argument("--train-batch", type=int, default=1000)
    parser.add_argument("--val-all", type=int, default=1000)
    parser.add_argument("--val-batch", type=int, default=1000)
    parser.add_argument("--test-all", type=int, default=10000)
    parser.add_argument("--test-batch", type=int, default=10000)

    # --- Output ---
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Append one summary line here for live tracking "
        "(default: <results-root>/size_sweep.jsonl)",
    )

    return parser.parse_args()


def _check_even_square(n_exc: int) -> int:
    """Fail fast with a clear message instead of a deep numpy assertion."""
    side = int(round(n_exc**0.5))
    if side * side != n_exc:
        sys.exit(f"--n-exc {n_exc} is not a perfect square (oriented_rf requires one)")
    if side % 2 != 0:
        sys.exit(
            f"--n-exc {n_exc} side {side} is odd; block orientation_mode "
            "(n_orientations=4 -> 2x2 grid) needs an even side"
        )
    return side


def build_output_dir(args) -> str:
    if args.output_dir is not None:
        return args.output_dir
    base = os.path.join(os.path.dirname(__file__), "results")
    tag = f"exc{args.n_exc}_inh{args.n_inh}_s{args.seed}"
    return os.path.join(base, tag)


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON line for live tracking. Best-effort; never fatal.

    The authoritative aggregation is rebuilt from the per-cell results.json by
    aggregate_size.py, so a partial / interleaved live jsonl is harmless.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as e:
        print(f"Warning: could not append to {path} ({e})")


def main():
    args = parse_args()
    _check_even_square(args.n_exc)

    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Proportional inhibition: scale both inhibitory peaks by N_INH_BASE / N_inh
    # so the per-neuron inhibitory influence stays constant across sizes (1/4 at 4x).
    # ------------------------------------------------------------------
    inh_scale = N_INH_BASE / args.n_inh
    peak_ei = PEAK_EI_BASE * inh_scale
    peak_ie = PEAK_IE_BASE * inh_scale

    weights = snn.weights.oriented_receptive_fields(
        density_se=DENSITY_SE,
        density_ee=DENSITY_EE,
        density_ei=DENSITY_EI,
        density_ie=DENSITY_IE,
        peak_se=PEAK_SE_BASE,
        peak_ee=PEAK_EE_BASE,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
        sigma_x=3.0,
        sigma_x_lognormal_std=0.0,
        sigma_x_lognormal_max=0.0,
        n_orientations=4,
        orientation_mode="block",
        sigma_ee_mean=0.0,
        sigma_ee_lognormal_std=0.5,
        ablate_ee=False,
        ablate_ie=False,
    )

    # ------------------------------------------------------------------
    # Layer — the swept dimension is (N_exc, N_inh)
    # ------------------------------------------------------------------
    layer = snn.Layer(
        N_exc=args.n_exc,
        N_inh=args.n_inh,
        membrane=snn.membrane.LIF(
            tau_m_exc=20.0,
            tau_m_inh=15.0,
            tau_syn_exc=10.0,
            tau_syn_inh=9.0,
            membrane_resistance_exc=15.0,
            membrane_resistance_inh=15.0,
            resting_potential=-70.0,
            reset_potential=-80.0,
            spike_threshold=-55.0,
            min_mp=-100.0,
            max_mp=40.0,
            mean_noise=0.0,
            var_noise=0.0,
            spike_adaptation=True,
            tau_adaptation=200.0,
            delta_adaptation=0.5,
        ),
        weights=weights,
    )

    # ------------------------------------------------------------------
    # Learner — run_test default (static x_tar, engaged-inhibition regime)
    # ------------------------------------------------------------------
    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,
        tau_trace=20,
        w_max=10.0,
        mu_weight=0.5,
        x_tar_mode="static",
        x_tar_pct_se=60.0,
        x_tar_pct_ee=30.0,
        x_tar_static_se=1.0,
        x_tar_static_ee=0.5,
        update_freq=100,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    # ------------------------------------------------------------------
    # Regulariser — fixed: neuron-wise normalisation
    # ------------------------------------------------------------------
    regularizer = snn.regularizer.Normalize(frequency=1050, mode="neuron")

    # ------------------------------------------------------------------
    # Model / data
    # ------------------------------------------------------------------
    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=args.seed,
        num_steps=350,
        all_images_train=args.train_all,
        batch_image_train=args.train_batch,
        all_images_val=args.val_all,
        batch_image_val=args.val_batch,
        all_images_test=args.test_all,
        batch_image_test=args.test_batch,
        image_dataset=args.dataset,
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    config = dict(
        dataset=args.dataset,
        weight_type="oriented_rf",
        seed=args.seed,
        epochs=args.epochs,
        val_every=args.val_every,
        n_exc=args.n_exc,
        n_inh=args.n_inh,
        inh_scale=inh_scale,
        peak_se=PEAK_SE_BASE,
        peak_ee=PEAK_EE_BASE,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
    )
    print(
        f"\nConfig — dataset: {args.dataset} | weight_type: oriented_rf"
        f" | N_exc: {args.n_exc} | N_inh: {args.n_inh}"
        f" | inh_scale: {inh_scale:.4f}"
        f" | peak_ei: {peak_ei:.4f} | peak_ie: {peak_ie:.4f}"
        f" | seed: {args.seed} | epochs: {args.epochs}\n"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []

    t_start = time.time()

    for result in model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=args.epochs,
        train_weights=True,
        save_model=True,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=False,
        pca_variance=0.95,
        stat_tracking_frequency=10500,
    ):
        if result.accuracy is None:
            continue

        if result.batch % args.val_every == 0:
            val = model.validate()
            val_acc = val.accuracy if val.accuracy is not None else float("nan")
            val_phi = val.phi if val.phi is not None else float("nan")
            print(
                f"epoch {result.epoch + 1}  batch {result.batch:>3}"
                f"  train_acc {result.accuracy:.4f}  train_phi {result.phi:.3f}"
                f"  val_acc {val_acc:.4f}  val_phi {val_phi:.3f}"
            )
            val_history.append(
                {"batch": result.batch, "val_acc": val_acc, "val_phi": val_phi}
            )
            if val.accuracy is not None and val.accuracy > best_val_acc:
                best_val_acc = val.accuracy
                best_val_phi = val_phi

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\n{'=' * 55}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test phi      : {test_phi:.4f}")
    print(f"Best val acc  : {best_val_acc:.4f}")
    print(f"Elapsed (s)   : {elapsed:.1f}")
    print(f"{'=' * 55}")

    # ------------------------------------------------------------------
    # Persist results
    # ------------------------------------------------------------------
    results = {
        "config": config,
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "elapsed_s": round(elapsed, 1),
        "val_history": val_history,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {results_path}")

    # Live-tracking jsonl (one summary line per run). aggregate_size.py rebuilds an
    # authoritative jsonl from the per-cell results.json, so this is best-effort only.
    jsonl_path = args.jsonl or os.path.join(
        os.path.dirname(__file__), "results", "size_sweep.jsonl"
    )
    append_jsonl(
        jsonl_path,
        {
            "n_exc": args.n_exc,
            "n_inh": args.n_inh,
            "seed": args.seed,
            "inh_scale": inh_scale,
            "peak_ei": peak_ei,
            "peak_ie": peak_ie,
            "test_acc": test_acc,
            "test_phi": test_phi,
            "best_val_acc": best_val_acc,
            "best_val_phi": best_val_phi,
            "elapsed_s": round(elapsed, 1),
        },
    )
    print(f"Tracking row appended -> {jsonl_path}")


if __name__ == "__main__":
    main()
