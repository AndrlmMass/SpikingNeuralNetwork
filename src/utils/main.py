import argparse
import os
import random
from datetime import datetime

import numpy as np

from src.network.model import SNNModel
from src.network.io import CheckpointManager
from src.network.runner import Runner
from src.utils.logger import HistoryTracker


def run_once(args, epoch):
    print(f"\n===== EPOCH {epoch + 1}/{args.epochs} =====")
    seed = np.random.seed(epoch)
    random.seed(seed)

    if args.dataset.lower() == "geomfig":
        classes = [0, 1, 2, 3]
        num_input = 225
        w_dense_se = 0.1
        tau_m = 30
        tau_syn = 30
        Rm = 30
        max_rate_hz = 67.0
    elif args.dataset.lower() == "fcx1":
        classes = [0, 1]
        num_input = 100
        w_dense_se = 0.5
        tau_m = 1
        tau_syn = 1
        Rm = 30
        max_rate_hz = 67.0
    else:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_input = 784
        w_dense_se = 0.05
        w_dense_ee = 0.05
        w_dense_ei = 0.05
        w_dense_ie = 0.05
        se_weights = 1.0
        ee_weights = 0.5
        ei_weights = 1.0
        ie_weights = -0.7
        tau_syn_exc = 10
        tau_syn_inh = 9
        learning_rate = 0.0004
        tau_m_exc = 20
        tau_m_inh = 15
        Rm_exc = 15
        Rm_inh = 15
        max_rate_hz = 90.0
        delta_adaption = 0.5
        tau_trace = 20
        tau_adaption = 200
        w_max = 10
        num_steps = 350
        mu_weight = 0.6
        pca_variance = 15
        reg_frequency = 1050
        sleep_duration = 100
        update_weights_freq = 100
        stat_tracking_frequency = 10500
        reg_mode = "static"
        noise_level = 4.0
        sleep = True
        norm = False

    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = SNNModel(
        classes=classes,
        random_state=seed,
        N_x=num_input,
        N_exc=1024,
        N_inh=225,
        ts_spec=ts_spec,
    )

    if args.test_mode:
        img_tr, img_va, img_te = 4, 4, 4
        b_tr, b_va, b_te = 4, 4, 4
    else:
        img_tr, img_va, img_te = 59000, 1000, 10000
        b_tr, b_va, b_te = 1000, 1000, 1000

    model.prepare_data(
        num_steps=num_steps,
        all_images_train=img_tr,
        batch_image_train=b_tr,
        all_images_test=img_te,
        batch_image_test=b_te,
        all_images_val=img_va,
        batch_image_val=b_va,
        image_dataset=args.dataset,
        max_rate_hz=max_rate_hz,
        gain=args.geom_gain,
    )

    model.prepare(
        plot_weights=False,
        w_dense_ee=w_dense_ee,
        w_dense_se=w_dense_se,
        w_dense_ei=w_dense_ei,
        w_dense_ie=w_dense_ie,
        se_weights=se_weights,
        ee_weights=ee_weights,
        ei_weights=ei_weights,
        ie_weights=ie_weights,
        random_weights=args.random_weights,
    )

    checkpoint = CheckpointManager()
    logger = HistoryTracker(ts_spec=ts_spec, image_dataset=args.dataset)
    runner = Runner(model=model, checkpoint=checkpoint, logger=logger)

    runner.run(
        train_weights=True,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        tau_m_exc=tau_m_exc,
        tau_m_inh=tau_m_inh,
        reg_mode=reg_mode,
        use_phi=True,
        clip_weights=True,
        membrane_resistance_exc=Rm_exc,
        membrane_resistance_inh=Rm_inh,
        delta_adaption=delta_adaption,
        tau_trace=tau_trace,
        mu_weight=mu_weight,
        PCA_plot=args.plot_PCA,
        heatmap_plot=args.heatmap_plot,
        save_model=args.save_model,
        var_noise=args.noise_level,
        track_weights=args.track_weights,
        stat_tracking_frequency=stat_tracking_frequency,
        sleep=sleep,
        track_stats=args.track_stats,
        A_minus=0.95,
        A_plus=1.0,
        tau_LTD=20,
        tau_LTP=20,
        tau_adaption=tau_adaption,
        w_max=w_max,
        learning_rate=learning_rate,
        accuracy_method="pca_lr",
        use_LR=True,
        reg_frequency=reg_frequency,
        sleep_duration=sleep_duration,
        update_weights_freq=update_weights_freq,
        normalize_weights=norm,
        profile=args.profile,
        pca_variance=pca_variance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-train", action="store_true", default=False)
    parser.add_argument("--on_timeout", type=str, default="give_up")
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--noise-level", type=float, nargs="+", default=0.1)
    parser.add_argument("--preview-dataset", action="store_true", default=False)
    parser.add_argument("--test-mode", action="store_true", default=False)
    parser.add_argument("--track-weights", action="store_true", default=False)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "kmnist", "fmnist", "fashionmnist", "fashion", "notmnist", "geomfig", "fcx1"],
        default="mnist",
    )
    parser.add_argument("--heatmap-plot", action="store_true", default=False)
    parser.add_argument("--plot-PCA", action="store_true", default=True)
    parser.add_argument("--get-giffed", action="store_true", default=False)
    parser.add_argument("--geom-noise-var", type=float, default=0.02)
    parser.add_argument("--geom-noise-mean", type=float, default=0.0)
    parser.add_argument("--geom-jitter", action="store_true", default=False)
    parser.add_argument("--geom-jitter-amount", type=float, default=0.05)
    parser.add_argument("--geom-gain", type=float, default=0.5)
    parser.add_argument("--geom-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--profile", action="store_true", default=True)
    parser.add_argument("--plot-acc-history", action="store_true", default=False)
    parser.add_argument("--plot-weights-per-epoch", action="store_true", default=False)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--plot-spikes-per-epoch", action="store_true", default=False)
    parser.add_argument("--sleep-mode", type=str, choices=["static", "group", "post"], default="static")
    parser.add_argument("--track-excel", action="store_true", default=False)
    parser.add_argument("--random-weights", action="store_true", default=True)
    parser.add_argument("--track-stats", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    args, _ = parser.parse_known_args()

    for e in range(args.epochs):
        run_once(args=args, epoch=e)


if __name__ == "__main__":
    main()
