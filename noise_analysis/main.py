from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy(classes=[0, 1, 2, 3])

# acquire data
snn_N.prepare_data(
    add_breaks=False, num_images=100, force_recreate=False, noisy_data=True
)

# set up network for training
snn_N.prepare_training(
    ei_weights=0.5,
    w_dense_ee=0.1,
    w_dense_ei=0.1,
    plot_weights=False,
)

# train network
snn_N.train(
    train_weights=True,
    noisy_potential=True,
    force_train=False,
    save_test_data=True,
    plot_weights=False,
    plot_spikes_train=True,
    plot_top_response_test=True,
    plot_top_response_train=True,
)

# analyze results
snn_N.analysis(clustering_estimation=True)
