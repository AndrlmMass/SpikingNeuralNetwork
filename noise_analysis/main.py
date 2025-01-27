from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_exc=200, N_inh=50, N_x=100)

# acquire data
snn_N.prepare_data(
    num_images=10,
    recreate=True,
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
    plot_spikes=False,
    plot_mp=False,
    plot_weights=False,
    plot_threshold=False,
    train_weights=False,
    learning_rate_exc=0.001,
    learning_rate_inh=0.001,
    var_noise=5,
    min_weight_inh=-1,
    max_weight_exc=1,
    spike_threshold_default=-55,
    perform_t_SNE=True,
    min_mp=-80,
    weight_decay=False,
    weight_decay_rate_exc=0.99,
    weight_decay_rate_inh=0.99,
    noisy_potential=True,
    noisy_threshold=False,
    noisy_weights=False,
    spike_adaption=False,
    delta_adaption=0.1,
    tau_adaption=2,
)

# analyse results
