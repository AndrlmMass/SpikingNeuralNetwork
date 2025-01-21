from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy()

# acquire data
snn_N.prepare_data(
    num_images=10,
    recreate=False,
    plot_spikes=False,
    noisy_data=True,
    noise_level=0.005,
    add_breaks=True,
    break_lengths=[1000, 1500, 2000],
    gain=0.5,
)

# set up network for training
snn_N.prepare_training(plot_weights=False, neg_weight=-1)

# train network
snn_N.train_network_(
    plot_spikes=True,
    plot_mp=True,
    plot_weights=True,
    update_weights=True,
    learning_rate_exc=0.00001,
    learning_rate_inh=0.00001,
    var_noise=3,
    min_weight_inh=-4,
    max_weight_exc=1,
)

# analyse results
## K
