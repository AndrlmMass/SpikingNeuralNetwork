from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

# acquire data
snn_N.prepare_data(add_breaks=False, num_images=500, force_recreate=False)

# set up network for training
snn_N.prepare_training(
    tp_weight=0,
    tn_weight=0,
    fp_weight=0,
    fn_weight=0,
    pn_weight=0,
    pp_weight=0,
    weight_affinity_output=0.1,
)

# train network
snn_N.train(
    train_weights=True,
    plot_spikes_train=False,
    noisy_potential=True,
    plot_weights=False,
    force_train=True,
    plot_accuracy_train=True,
    plot_accuracy_test=True,
    plot_spikes_test=False,
    start_time_spike_plot=450000,
)

# analyze results
snn_N.analysis(t_sne=True, pca=False, t_sne_train=True)
