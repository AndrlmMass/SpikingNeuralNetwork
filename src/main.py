from big_comb import snn_sleepy
import argparse
import numpy as np
import random
from datetime import datetime
import os


def run_once(
    args,
    epoch,
):
    print(f"\n===== EPOCH {epoch + 1}/{args.epochs+1} =====")
    # set seeds for run-to-run variance control
    seed = np.random.seed(epoch)
    random.seed(seed)

    if args.dataset.lower() == "geomfig":
        classes = [0, 1, 2, 3]
        num_input = 225
        use_validation_data = True
        w_dense_se = 0.1
        tau_m = 30
        tau_syn = 30
        Rm = 30
        max_rate_hz = 67.0
    elif args.dataset.lower() == "fcx1":
        classes = [0, 1]
        num_input = 100
        use_validation_data = False
        w_dense_se = 0.5
        tau_m = 1
        tau_syn = 1
        Rm = 30
        max_rate_hz = 67.0
    else:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_input = 784
        use_validation_data = True
        w_dense_se = 0.05  # original: 0.05
        w_dense_ee = 0.05  # original: 0.05
        w_dense_ei = 0.05  # original: 0.05
        w_dense_ie = 0.05  # original: 0.05
        se_weights = 1.0  # original: 1.0
        ee_weights = 0.5  # original: 0.7
        ei_weights = 1.0  # original: 0.7
        ie_weights = -0.7  # original: -0.5
        tau_syn_exc = 10  # original: 10
        tau_syn_inh = 9  # original: 9
        learning_rate = 0.0004  # original: 0.0004
        tau_m_exc = 20  # original: 20
        tau_m_inh = 15  # original: 15
        Rm_exc = 15  # original: 17
        Rm_inh = 15  # original: 19
        max_rate_hz = 90.0  # original: 250.0
        delta_adaption = 0.5  # original: 0.5
        tau_trace = 20  # original: 20
        tau_adaption = 200  # original: 200
        w_max = 10  # original: 10
        num_steps = 350  # original: 350
        mu_weight = 0.6  # original: 0.6
        pca_variance = 15  # original: 15
        reg_frequency = 1050
        sleep_duration = 100
        update_weights_freq = 100
        stat_tracking_frequency = 10500
        reg_mode = "static"
        noise_level = 4.0
        sleep = True
        norm = False

    ts_spec = datetime.now().strftime("%Y%m%d_%H%M%S")

    # init class
    snn_N = snn_sleepy(
        classes=classes,
        random_state=seed,
        N_x=num_input,
        N_exc=1024,
        N_inh=225,
        ts_spec=ts_spec,
    )

    # acquire data
    use_geomfig = args.dataset
    is_test_mode = args.test_mode
    # Use a lower gain for geomfig to reduce input drive
    gain_for_dataset = args.geom_gain
    if is_test_mode and use_geomfig:
        img_tr, img_va, img_te = 4, 4, 4
        b_tr, b_va, b_te = 4, 4, 4
        force_recreate_flag = True
    else:
        img_tr, img_va, img_te = 10000, 1000, 2000
        b_tr, b_va, b_te = 100, 1000, 1000
        force_recreate_flag = False
    snn_N.prepare_data(
        all_audio_train=22000,
        batch_audio_train=100,
        all_audio_test=7800,
        batch_audio_test=600,
        all_audio_val=200,
        batch_audio_val=200,
        num_steps=num_steps,
        all_images_train=img_tr,
        batch_image_train=b_tr,
        all_images_test=img_te,
        batch_image_test=b_te,
        all_images_val=img_va,
        batch_image_val=b_va,
        use_validation_data=use_validation_data,
        add_breaks=False,
        force_recreate=force_recreate_flag,
        noisy_data=False,
        gain=gain_for_dataset,
        noise_level=noise_level,
        max_rate_hz=max_rate_hz,
        audioMNIST=False,
        imageMNIST=True,
        create_data=False,
        plot_spectrograms=False,
        image_dataset=args.dataset,
    )

    # set up network for training
    snn_N.prepare_network(
        plot_weights=False,
        w_dense_ee=w_dense_ee,
        w_dense_se=w_dense_se,
        w_dense_ei=w_dense_ei,
        w_dense_ie=w_dense_ie,
        se_weights=se_weights,
        ee_weights=ee_weights,
        ei_weights=ei_weights,
        ie_weights=ie_weights,
        create_network=False,
        random_weights=args.random_weights,
    )

    snn_N.train_network(
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
        use_validation_data=False,
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
        use_QDA=False,
        use_LR=True,
        reg_frequency=reg_frequency,
        sleep_duration=sleep_duration,
        update_weights_freq=update_weights_freq,
        early_stopping=args.early_stopping,
        early_stopping_patience_pct=0.1,
        normalize_weights=norm,
        profile=args.profile,
        pca_variance=pca_variance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="number of repeated runs")
    parser.add_argument(
        "--no-train", action="store_true", default=False, help="disable training"
    )
    parser.add_argument(
        "--on_timeout", type=str, default="give_up", help="action to take on timeout"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="enable early stopping",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        nargs="+",
        default=0.1,
        help=(
            "noise level during training "
            "(can specify multiple, e.g., --sleep-rate 0.5 0.6 0.7)"
        ),
    )
    parser.add_argument(
        "--preview-dataset",
        action="store_true",
        default=False,
        help="plot a quick sample of the selected dataset before training",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="create minimal dataset (geomfig: 4 per split) and force recreate",
    )
    parser.add_argument(
        "--track-weights",
        action="store_true",
        default=False,
        help="track weights during training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "kmnist",
            "fmnist",
            "fashionmnist",
            "fashion",
            "notmnist",
            "geomfig",
            "fcx1",
        ],
        default="mnist",
        help="dataset to use (image-only or geomfig)",
    )
    parser.add_argument(
        "--heatmap-plot",
        action="store_true",
        default=False,
        help="plot the heatmap of the weights",
    )
    parser.add_argument(
        "--plot-PCA",
        action="store_true",
        default=True,
        help="plot the PCA of the network activity",
    )
    parser.add_argument(
        "--get-giffed",
        action="store_true",
        default=False,
        help="create gif from heatmap plots",
    )
    parser.add_argument(
        "--geom-noise-var",
        type=float,
        default=0.02,
        help="per-pixel Gaussian noise variance for geomfig generation",
    )
    parser.add_argument(
        "--geom-noise-mean",
        type=float,
        default=0.0,
        help="noise mean offset for geomfig generation",
    )
    parser.add_argument(
        "--geom-jitter",
        action="store_true",
        default=False,
        help="enable random jitter of size/thickness for geomfig (disabled by default)",
    )
    parser.add_argument(
        "--geom-jitter-amount",
        type=float,
        default=0.05,
        help="relative jitter amount for geomfig size/thickness (0.05 = ±5%)",
    )
    parser.add_argument(
        "--geom-gain",
        type=float,
        default=0.5,
        help="rate-coding gain for geomfig (multiplies pixel intensities before Poisson sampling)",
    )
    parser.add_argument(
        "--geom-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="number of parallel workers for geomfig generation (process-based)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=True,
        help="enable cProfile around training to find hotspots",
    )
    parser.add_argument(
        "--plot-acc-history",
        action="store_true",
        default=False,
        help="plot accuracy history at end (train+val together, test separate)",
    )
    parser.add_argument(
        "--plot-weights-per-epoch",
        action="store_true",
        default=False,
        help="plot and save weights after each epoch (for debugging, saves to plots/weights_epoch_*.png)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="store trained model weights for later re-training.",
    )
    parser.add_argument(
        "--plot-spikes-per-epoch",
        action="store_true",
        default=False,
        help="plot and save spikes after each epoch (for debugging, saves to plots/spikes_*_epoch_*.png)",
    )
    parser.add_argument(
        "--sleep-mode",
        type=str,
        choices=["static", "group", "post"],
        default="static",
        help="sleep target mode: static uses fixed targets; group uses group-mean magnitudes; post uses per-post mean magnitudes",
    )
    parser.add_argument(
        "--track-excel",
        action="store_true",
        default=False,
        help="track results in GLM/Results_.xlsx file after each run",
    )
    parser.add_argument(
        "--random-weights",
        action="store_true",
        default=True,
        help="use random weights (0.01 for ee, 0.05 for se, 0.05 for ei, 0.05 for ie)",
    )
    parser.add_argument(
        "--track-stats",
        action="store_true",
        default=False,
        help="track statistics during training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="define the number of epochs the model should train",
    )
    args, _ = parser.parse_known_args()

    for e in range(args.epochs):
        run_once(args=args, epoch=e)


if __name__ == "__main__":
    main()
