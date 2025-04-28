from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# acquire data
snn_N.prepare_data(
    add_breaks=False,
    force_recreate=False,
    noisy_data=True,
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
    plot_accuracy_test=False,
    plot_accuracy_train=False,
    compare_decay_rates=True,
    weight_decay_rate_exc=[
        0.999995,
        0.999993,
        0.99999,
        0.99997,
        0.99995,
        0.99993,
        0.9999,
        0.9998,
        0.9996,
        0.9992,
        0.997,
        0.0,
    ],
    weight_decay_rate_inh=[
        0.999995,
        0.999993,
        0.99999,
        0.99997,
        0.99995,
        0.99993,
        0.9999,
        0.9998,
        0.9996,
        0.9992,
        0.997,
        0.0,
    ],
    samples=10,
    force_train=False,
    save_test_data=True,
    plot_weights=False,
    plot_spikes_train=False,
    plot_top_response_test=False,
    plot_top_response_train=False,
)

# analyze results
snn_N.analysis(t_sne_test=True)
