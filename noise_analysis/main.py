from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_exc=200, N_inh=50, N_x=100)

# acquire data
snn_N.prepare_data(
    num_images=5,
    recreate=False,
    plot_comparison=False,
    plot_spikes=False,
    noisy_data=False,
    noise_level=0.005,
    add_breaks=False,
    break_lengths=[500, 1500, 1000],
    classes=[0, 1, 2, 3],
    gain=1.0,
    test_data_ratio=0.5,
    max_time=2000,
)

# set up network for training
snn_N.prepare_training(
    plot_weights=False,
    neg_weight=-0.2,
    pos_weight=0.4,
    weight_affinity_hidden_exc=0.1,
    weight_affinity_hidden_inh=0.1,
    weight_affinity_input=0.05,
)

# train network
snn_N.train(
    plot_spikes_train=False,
    plot_spikes_test=False,
    plot_mp_train=False,
    plot_mp_test=False,
    plot_weights=True,
    plot_threshold=False,
    plot_traces_=False,
    train_weights=True,
    learning_rate_exc=0,
    learning_rate_inh=0,
    w_target_exc=0.01,
    w_target_inh=-0.01,
    var_noise=1,
    min_weight_inh=-25,
    max_weight_inh=-0.05,
    max_weight_exc=25,
    min_weight_exc=0.05,
    spike_threshold_default=-55,
    check_sleep_interval=1000,
    interval=100,
    min_mp=-80,
    sleep=True,
    weight_decay=False,
    weight_decay_rate_exc=0.99,
    weight_decay_rate_inh=0.99,
    noisy_potential=True,
    noisy_threshold=False,
    noisy_weights=False,
    spike_adaption=True,
    delta_adaption=0.5,
    tau_adaption=100,
    save_weights=True,
    trace_update=False,
    timing_update=True,
    vectorized_trace=False,
    clip_exc_weights=True,
    clip_inh_weights=True,
    alpha=1.5,
    beta=1.0,
    A_plus=0.25,
    A_minus=0.5,
    test=True,
    tau_LTD=10,
    tau_LTP=20,
)


# analyze results
snn_N.analysis(t_sne=True, pls=True, n_components=2)
