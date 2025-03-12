from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_x=100, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# acquire data
snn_N.prepare_data(num_images=300)

# set up network for training
snn_N.prepare_training(tp_weight=100, tn_weight=100, fp_weight=-100, fn_weight=-100)

# train network
snn_N.train(
    train_weights=False,
    sleep=True,
    force_train=False,
    timing_update=True,
    plot_accuracy_=True,
)

# analyze results
snn_N.analysis(t_sne=True, t_sne_test=True, t_sne_train=True, pls=False)
