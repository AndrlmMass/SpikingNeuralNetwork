from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

# acquire data
snn_N.prepare_data(add_breaks=False, num_images=1000, force_recreate=True)

# set up network for training
snn_N.prepare_training(tp_weight=1, tn_weight=-1, fp_weight=-1, fn_weight=-1)

# train network
snn_N.train(
    train_weights=True,
    plot_spikes_train=False,
    noisy_potential=True,
    plot_weights=False,
    force_train=True,
    plot_accuracy_train=True,
    plot_accuracy_test=True,
)

# analyze results
snn_N.analysis(t_sne=True, pca=False, t_sne_train=True)
