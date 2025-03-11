from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy()

# acquire data
snn_N.prepare_data()

# set up network for training
snn_N.prepare_training(tp_weight=10, tn_weight=-10, fp_weight=-10, fn_weight=-10)

# train network
snn_N.train(train_weights=True, plot_spikes_train=True)

# analyze results
snn_N.analysis(t_sne=True, pls=True)
