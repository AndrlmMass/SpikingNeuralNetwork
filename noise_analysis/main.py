from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy()

# acquire data
snn_N.prepare_data(
    num_images=30,
    recreate=False,
    plot_spikes=False,
    noisy_data=True,
    noise_level=0.005,
    add_breaks=True,
    break_lengths=[1000, 1500, 2000],
    gain=0.5,
)

# set up network for training
snn_N.prepare_training(plot_weights=False, neg_weight=-0.5, pos_weight=0.5)

# train network
snn_N.train_network_(
    plot_spikes=True,
    plot_mp=False,
    plot_weights=True,
    update_weights=True,
    learning_rate_exc=0.001,
    learning_rate_inh=0.001,
    var_noise=5,
    min_weight_inh=-1,
    max_weight_exc=1,
    tau_decay_exc=9.5,
    tau_decay_inh=9.5,
    weight_decay_rate_exc=0.99,
    weight_decay_rate_inh=0.99,
)

# analyse results
