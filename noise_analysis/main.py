from big_comb import snn_sleepy

# init class
snn_N = snn_sleepy(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# acquire data
snn_N.prepare_data(add_breaks=False, num_images=100, force_recreate=True)

# set up network for training
snn_N.prepare_training(
    tp_weight=0.01,
    tn_weight=0.01,
    fp_weight=-0.01,
    fn_weight=-0.01,
    pn_weight=-1,
    pp_weight=1,
    epn_weight=1,
    epp_weight=1,
    ei_weights=0.5,
    w_dense_ee=0.1,
    w_dense_ei=0.1,
    plot_weights=False,
    weight_affinity_output=0.1,
)

# train network
snn_N.train(
    train_weights=True,
    noisy_potential=True,
    force_train=False,
    save_test_data=True,
    plot_weights=False,
)

# analyze results
snn_N.analysis(clustering_estimation=True)
