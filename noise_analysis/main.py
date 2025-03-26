from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy(
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], supervised=True, unsupervised=False
)

# acquire data
snn_N.prepare_data(add_breaks=True, num_images=500, force_recreate=False)

# set up network for training
snn_N.prepare_training(
    tp_weight=10,
    tn_weight=10,
    fp_weight=-1,
    fn_weight=-1,
    pn_weight=-1,
    pp_weight=1,
    epn_weight=1,
    epp_weight=1,
    weight_affinity_output=0.1,
    weight_affinity_hidden_exc=0.05,
)

# train network
snn_N.train(
    train_weights=True,
    plot_spikes_train=True,
    noisy_potential=True,
    plot_weights=True,
    force_train=False,
    plot_accuracy_train=True,
    plot_accuracy_test=True,
    plot_spikes_test=True,
)

# analyze results
snn_N.analysis(t_sne=True, pca=False, t_sne_train=True)
