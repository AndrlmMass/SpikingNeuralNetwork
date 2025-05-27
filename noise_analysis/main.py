from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

# acquire data
snn_N.prepare_data(
    tot_images_train=5000,
    single_train=1000,
    single_test=200,
    tot_images_test=1000,
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
    compare_decay_rates=False,
    weight_decay_rate_exc=[
        0.99997,
    ],
    weight_decay_rate_inh=[
        0.99997,
    ],
    samples=10,
    force_train=False,
    plot_weights=False,
    plot_spikes_train=False,
    plot_top_response_test=False,
    plot_top_response_train=False,
)

# analyze results
snn_N.analysis(t_sne_test=True)
