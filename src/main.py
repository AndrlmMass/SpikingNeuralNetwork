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
    all_images_train=600,
    batch_image_train=600,
    all_images_test=2000,
    batch_image_test=200,
    all_images_val=200,
    batch_image_val=200,
    add_breaks=False,
    force_recreate=False,
    noisy_data=False,
    gain=1.0,
    noise_level=0.0,
    audioMNIST=False,
    imageMNIST=True,
    create_data=False,
    plot_spectrograms=False,
)

# set up network for training
snn_N.prepare_network(
    plot_weights=False,
    w_dense_ee=0.15,  # Moderate excitatory connectivity (typical: 0.1-0.2)
    w_dense_se=0.1,  # Sensory input density (typical: 0.05-0.15)
    w_dense_ei=0.2,  # Higher EI density for inhibition regulation (typical: 0.15-0.3)
    w_dense_ie=0.25,  # Higher IE density for lateral inhibition (typical: 0.2-0.4)
    se_weights=0.15,  # Moderate sensory strength (typical: 0.1-0.3)
    ee_weights=0.3,  # Balanced excitation (typical: 0.2-0.5)
    ei_weights=0.6,  # Strong EI for inhibitory control (typical: 0.4-0.8)
    ie_weights=-0.5,  # Strong lateral inhibition (typical: -0.5 to -1.0)
    create_network=False,
)

# train network
snn_N.train_network(
    train_weights=True,
    noisy_potential=True,  # Add small noise to break symmetry
    compare_decay_rates=False,
    check_sleep_interval=35000,
    samples=10,
    force_train=True,
    plot_spikes_train=False,
    plot_weights=False,
    plot_epoch_performance=False,
    sleep_synchronized=False,
    plot_top_response_test=False,
    plot_top_response_train=False,
    plot_tsne_during_training=False,
    tsne_plot_interval=1,
    plot_spectrograms=False,
    use_validation_data=False,
    var_noise=2,
    sleep=True,
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
    use_QDA=True,
    compare_sleep_rates=True,
    weight_decay_rate_exc=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999],
    weight_decay_rate_inh=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999],
)

# analyze results
snn_N.analyze_results(t_sne_test=True)
