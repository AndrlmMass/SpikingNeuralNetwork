from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_x=100)

# acquire data
snn_N.prepare_data()

# set up network for training
snn_N.prepare_training(tp_weight=10, tn_weight=-10, fp_weight=-10, fn_weight=-10)

# train network
snn_N.train(plot_accuracy_=True)

# analyze results
snn_N.analysis(t_sne=False, pls=False)
