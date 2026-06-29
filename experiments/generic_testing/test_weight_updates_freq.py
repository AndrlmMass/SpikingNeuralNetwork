"""
Sweep over STDP weight-update frequencies to see whether updating weights
every 1, 10, 100, or 1000 timesteps affects classification performance.

Each config: 10 000 training images/epoch x 3 epochs (30 000 effective samples).
Results are written to experiments/generic_testing/results/update_freq/freq_<N>/results.json

Usage
-----
python experiments/generic_testing/test_weight_updates_freq.py
python experiments/generic_testing/test_weight_updates_freq.py --seed 7
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import neurosnn as snn

TRAIN_ALL = 10_000
EPOCHS = 3
VAL_ALL = 1_000
TEST_ALL = 5_000
SEED = 42


def run_one(update_freq: int, seed: int) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  update_freq = {update_freq}")
    print(f"  train {TRAIN_ALL}/epoch x {EPOCHS} epochs  |  seed {seed}")
    print(f"{'=' * 60}\n")

    weights = snn.weights.oriented_receptive_fields(
        density_se=0.01,
        density_ee=0.01,
        density_ei=0.03,
        density_ie=0.05,
        peak_se=4.0,
        peak_ee=1.0,
        peak_ei=2.0,
        peak_ie=-2.0,
        sigma_x=3.0,
        sigma_x_lognormal_std=1.0,
        sigma_x_lognormal_max=0.0,
        n_orientations=4,
        orientation_mode="block",
    )

    layer = snn.Layer(
        N_exc=1024,
        N_inh=225,
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

    learner = snn.learner.TraceSTDP(
        learning_rate=0.0004,
        tau_trace=20,
        w_max=10.0,
        mu_weight=0.5,
        x_tar_mode="static",
        x_tar_static_se=1.0,
        x_tar_static_ee=0.8,
        update_freq=update_freq,
        clip_weights=True,
        min_weight_exc=0.01,
        max_weight_exc=25.0,
        min_weight_inh=-25.0,
        max_weight_inh=-0.01,
    )

    regularizer = snn.regularizer.Normalize(frequency=1050, mode="layer")

    model = snn.Model(
        input_size=784,
        classes=list(range(10)),
        random_state=seed,
        num_steps=350,
        all_images_train=TRAIN_ALL,
        batch_image_train=1000,
        all_images_val=VAL_ALL,
        batch_image_val=1000,
        all_images_test=TEST_ALL,
        batch_image_test=1000,
        image_dataset="mnist",
        max_rate_hz=90.0,
        gain=1.0,
        gabor=False,
    )

    best_val_acc = 0.0
    best_val_phi = float("nan")
    val_history = []
    t_start = time.time()

    train_gen = model.train(
        layers=[layer],
        learner=learner,
        regularizer=regularizer,
        epochs=EPOCHS,
        train_weights=True,
        save_model=False,
        accuracy_method="pca_lr",
        use_LR=True,
        use_phi=True,
        use_pca=False,
        pca_variance=15,
        track_stats=False,
    )

    for result in train_gen:
        if result.accuracy is None:
            continue
        val = model.validate()
        val_acc = val.accuracy if val.accuracy is not None else float("nan")
        val_phi = val.phi if val.phi is not None else float("nan")
        elapsed = time.time() - t_start
        print(
            f"epoch {result.epoch + 1}/{EPOCHS}  batch {result.batch:>3}"
            f"  train_acc {result.accuracy:.4f}  train_phi {result.phi:.3f}"
            f"  val_acc {val_acc:.4f}  val_phi {val_phi:.3f}"
            f"  [{elapsed:.0f}s]"
        )
        val_history.append(
            {
                "epoch": result.epoch,
                "batch": result.batch,
                "val_acc": val_acc,
                "val_phi": val_phi,
                "train_acc": result.accuracy,
                "train_phi": result.phi,
                "elapsed_s": round(elapsed, 1),
            }
        )
        if val.accuracy is not None and val.accuracy > best_val_acc:
            best_val_acc = val.accuracy
            best_val_phi = val_phi

    elapsed = time.time() - t_start

    test = model.test()
    test_acc = test.accuracy if test.accuracy is not None else float("nan")
    test_phi = test.phi if test.phi is not None else float("nan")

    print(f"\nupdate_freq={update_freq}  test_acc={test_acc:.4f}  elapsed={elapsed:.0f}s\n")

    return {
        "update_freq": update_freq,
        "seed": seed,
        "train_all": TRAIN_ALL,
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
        "best_val_phi": best_val_phi,
        "test_acc": test_acc,
        "test_phi": test_phi,
        "elapsed_s": round(elapsed, 1),
        "val_history": val_history,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    update_freqs = [10, 100, 1000]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    date_str = datetime.now().strftime("%Y.%m.%d")

    summary = []
    summary_dir = None
    for freq in update_freqs:
        result = run_one(freq, args.seed)

        run_id = f"update_freq_{freq}_s{args.seed}"
        run_dir = os.path.join(repo_root, "results", "mnist", date_str, run_id)
        os.makedirs(os.path.join(run_dir, "statistics"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

        # Split val_history into statistics, keep top-level metrics in results.json
        stats = result.pop("val_history")
        with open(os.path.join(run_dir, "statistics", "val_history.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved -> {run_dir}")

        if summary_dir is None:
            summary_dir = os.path.join(repo_root, "results", "mnist", date_str)

        summary.append(
            {
                "update_freq": freq,
                "run_id": run_id,
                "test_acc": result["test_acc"],
                "test_phi": result["test_phi"],
                "best_val_acc": result["best_val_acc"],
                "elapsed_s": result["elapsed_s"],
            }
        )

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY  (seed={args.seed}, {TRAIN_ALL}/epoch x {EPOCHS} epochs)")
    print(f"{'=' * 60}")
    print(f"  {'update_freq':>12}  {'test_acc':>9}  {'test_phi':>9}  {'best_val':>9}")
    for row in summary:
        print(
            f"  {row['update_freq']:>12}  {row['test_acc']:>9.4f}"
            f"  {row['test_phi']:>9.4f}  {row['best_val_acc']:>9.4f}"
        )

    summary_path = os.path.join(summary_dir, "update_freq_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
