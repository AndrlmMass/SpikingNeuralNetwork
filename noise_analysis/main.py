from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_exc=200, N_inh=50, N_x=100)

# acquire data
snn_N.prepare_data(
    num_images=50,
    recreate=True,
    plot_spikes=False,
    noisy_data=True,
    noise_level=0.005,
    add_breaks=True,
    break_lengths=[500, 1500, 1000],
    gain=0.5,
)

# set up network for training
snn_N.prepare_training(
    plot_weights=False,
    neg_weight=-0.5,
    pos_weight=0.5,
)

# train network
snn_N.train_network_(
    plot_spikes=False,
    plot_mp=False,
    plot_weights=True,
    plot_threshold=False,
    train_weights=True,
    learning_rate_exc=0.1,
    learning_rate_inh=100.0,
    w_target_exc=0.1,
    w_target_inh=-0.1,
    var_noise=2,
    min_weight_inh=-1,
    max_weight_inh=-0.05,
    max_weight_exc=1,
    min_weight_exc=0.05,
    spike_threshold_default=-55,
    min_mp=-80,
    sleep=False,
    weight_decay=False,
    weight_decay_rate_exc=3,
    weight_decay_rate_inh=3,
    noisy_potential=True,
    noisy_threshold=False,
    noisy_weights=False,
    spike_adaption=True,
    delta_adaption=0.5,
    tau_adaption=10,
    save_weights=True,
    trace_update=True,
    timing_update=False,
    vectorized_trace=False,
    clip_exc_weights=True,
    clip_inh_weights=True,
    alpha=2.25,
    A_plus=0.1,
    A_minus=1,
)

# analyze results
snn_N.analysis(t_sne=True, pls=False)
