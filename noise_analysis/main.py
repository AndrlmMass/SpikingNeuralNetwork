from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_x=100, classes=[1, 0])

# acquire data
snn_N.prepare_data(num_images=100)

# set up network for training
snn_N.prepare_training(tp_weight=100, tn_weight=100, fp_weight=-10, fn_weight=-10)

# train network
snn_N.train(train_weights=True, plot_spikes_test=False, plot_spikes_train=False)

# analyze results
snn_N.analysis(t_sne=True, t_sne_test=True, t_sne_train=True, pls=False)
