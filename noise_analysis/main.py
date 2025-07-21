from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

### REMOVE PREPARE

# acquire data
snn_N.prepare_data(
    tot_images_train=10000,
    single_train=1000,
    single_test=200,
    tot_images_test=1000,
    add_breaks=False,
    force_recreate=False,
    noisy_data=True,
    gain=1.0,
)

# set up network for training
snn_N.prepare_training(
    plot_weights=False,
    w_dense_ee=0.01,
    w_dense_se=0.1,
    w_dense_ei=0.8,
    w_dense_ie=0.5,
    se_weights=0.5,
    ee_weights=0.05,
    ei_weights=0.3,
    ie_weights=-0.8,
)


# train network
snn_N.train(
    train_weights=True,
    noisy_potential=True,
    compare_decay_rates=False,
    weight_decay_rate_exc=[
        0.99997,
    ],
    weight_decay_rate_inh=[
        0.99997,
    ],
    samples=10,
    force_train=False,
    plot_spikes_train=True,
    plot_weights=True,
    sleep_synchronized=True,
    plot_top_response_test=False,
    plot_top_response_train=False,
    use_validation_data=True,
    validation_split=0.2,
    tau_syn=30,
    narrow_top=0.1,
    A_minus=0.5,
    A_plus=0.5,
    tau_LTD=8,
    tau_LTP=10,
    learning_rate_exc=0.001,
)

# analyze results
snn_N.analysis(t_sne_test=True)
