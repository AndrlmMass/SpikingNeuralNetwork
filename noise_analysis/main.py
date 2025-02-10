from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_exc=200, N_inh=50, N_x=100)

# acquire data
snn_N.prepare_data(
    num_images=100,
    recreate=True,
    plot_comparison=False,
    plot_spikes=False,
    noisy_data=True,
    noise_level=0.005,
    add_breaks=True,
    break_lengths=[500, 1500, 1000],
    classes=[0, 1],
    gain=1.5,
    test_data_ratio=0.5,
)

# set up network for training
snn_N.prepare_training(
    plot_weights=False,
    neg_weight=-0.5,
    pos_weight=0.5,
    weight_affinity_hidden_exc=0.2,
    weight_affinity_hidden_inh=0.1,
    weight_affinity_input=0.1,
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
    learning_rate_exc=0.0005,
    learning_rate_inh=0.0005,
    w_target_exc=0.5,
    w_target_inh=-0.5,
    var_noise=2,
    min_weight_inh=-100,
    max_weight_inh=-0.05,
    max_weight_exc=100,
    min_weight_exc=0.05,
    spike_threshold_default=-55,
    check_sleep_interval=100,
    interval=1000,
    min_mp=-80,
    sleep=True,
    weight_decay=False,
    weight_decay_rate_exc=0.5,
    weight_decay_rate_inh=0.5,
    noisy_potential=True,
    noisy_threshold=False,
    noisy_weights=False,
    spike_adaption=True,
    delta_adaption=0.8,
    tau_adaption=10,
    save_weights=True,
    trace_update=False,
    timing_update=True,
    vectorized_trace=False,
    clip_exc_weights=True,
    clip_inh_weights=True,
    alpha=1.25,
    beta=0.8,
    A_plus=0.25,
    A_minus=0.5,
    test=True,
    tau_LTD=10,
    tau_LTP=20,
)


# analyze results
snn_N.analysis(t_sne=True, pls=True)
