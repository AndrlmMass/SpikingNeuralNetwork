from big_comb import SNN_noisy

# init class
snn_N = SNN_noisy(N_exc=200, N_inh=50)

# acquire data
snn_N.prepare_data(
    plot_spikes=False,
    plot_heat_map=False,
    retur=False,
    pixel_size=10,
    num_steps=1000,
    gain=1,
    offset=0,
    first_spike_time=0,
    time_var_input=False,
    download=False,
    num_images=20,
)


# set up network for training
snn_N.prepare_training(
    weight_affinity_hidden=0.1,
    weight_affinity_input=0.05,
    pos_weight=1,
    neg_weight=4,
    plot_weights=False,
    retur=False,
)
