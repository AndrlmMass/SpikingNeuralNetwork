from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy()

# acquire data
snn_N.prepare_data(num_images=5)

# set up network for training
snn_N.prepare_training(plot_weights=False)

# train network
snn_N.train_network_(
    plot_spikes=True, plot_mp=True, plot_weights=True, load_weights=True
)

# analyse results

## K
