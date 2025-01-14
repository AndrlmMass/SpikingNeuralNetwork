from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy()

# acquire data
snn_N.prepare_data()

# set up network for training
snn_N.prepare_training()

# train network
snn_N.train_network_(
    plot_spikes=True, plot_mp=True, start_index_mp=-50, stop_index_mp=350
)

# analyse results
