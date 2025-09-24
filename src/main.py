from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

# acquire data
snn_N.prepare_data(
    all_audio_train=22000,
    batch_audio_train=600,
    all_audio_test=7900,
    batch_audio_test=600,
    all_audio_val=100,
    batch_audio_val=100,
    all_images_train=22000,
    batch_image_train=600,
    all_images_test=7900,
    batch_image_test=600,
    all_images_val=100,
    batch_image_val=100,
    add_breaks=False,
    force_recreate=False,
    noisy_data=False,
    gain=1.0,
    noise_level=0.0,
    audioMNIST=True,
    imageMNIST=False,
    create_data=False,
    plot_spectrograms=False,
)

# set up network for training
snn_N.prepare_network(
    plot_weights=False,
    w_dense_ee=0.1,  # Increased from 0.05 for more diverse firing
    w_dense_se=0.03,  # Further reduced from 0.05
    w_dense_ei=0.15,  # Increased from 0.1 for stronger inhibition
    w_dense_ie=0.15,  # Increased from 0.1 for stronger inhibition
    se_weights=0.05,  # Further reduced from 0.1
    ee_weights=0.3,  # Increased from 0.2 for more excitation
    ei_weights=0.6,  # Increased from 0.4 for stronger inhibition
    ie_weights=-0.3,  # More negative from -0.2 for stronger inhibition
    create_network=False,
)

# train network
snn_N.train_network(
    train_weights=True,
    noisy_potential=True,  # Add small noise to break symmetry
    compare_decay_rates=False,
    check_sleep_interval=10000,
    weight_decay_rate_exc=[
        0.99997,
    ],
    weight_decay_rate_inh=[
        0.99997,
    ],
    samples=10,
    force_train=True,
    plot_spikes_train=False,
    plot_weights=False,
    plot_epoch_performance=True,
    sleep_synchronized=False,
    plot_top_response_test=False,
    plot_top_response_train=False,
    plot_tsne_during_training=False,
    tsne_plot_interval=1,
    plot_spectrograms=False,
    use_validation_data=False,
    var_noise=2,
    sleep=False,
    tau_syn=30,
    narrow_top=0.2,
    A_minus=0.3,
    A_plus=0.5,
    tau_LTD=7.5,
    tau_LTP=10,
    learning_rate_exc=0.0008,
    learning_rate_inh=0.005,
    accuracy_method="pca_lr",
    test_only=False,
)

# analyze results
snn_N.analyze_results(t_sne_test=True)
