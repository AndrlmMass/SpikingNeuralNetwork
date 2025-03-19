from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy()

# acquire data
snn_N.prepare_data(add_breaks=False, num_images=500)

# set up network for training
snn_N.prepare_training(tp_weight=200, tn_weight=-50, fp_weight=-50, fn_weight=-50)

# train network
snn_N.train(
    train_weights=True,
    plot_spikes_train=False,
    plot_accuracy_=True,
    noisy_potential=True,
    plot_weights=True,
    force_train=True,
)

# analyze results
snn_N.analysis(t_sne=True, pca=False)
