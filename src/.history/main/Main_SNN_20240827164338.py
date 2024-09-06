# Main SNN execution file

# Set cw
import os
import sys

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "gen"))
sys.path.append(os.path.join(base_path, "main"))

from SNN_functions import SNN_STDP


# Store the parameters in a dictionary
snn_params = {
    "V_th": -50,
    "V_reset": -70,
    "P": 20,  # Potential strength
    "C": 1,  # Where does it say that this should be 1?
    "U": 0.2,  # Initial release probability parameter
    "tau_plus": 20,
    "tau_minus": 20,
    "tau_slow": 100,
    "tau_ht": 100,
    "tau_hom": 1.2 * 10**6,  # metaplasticity time constant (20 minutes)
    "tau_istdp": 20,
    "tau_H": 6 * 10**2,  # 10s
    "tau_thr": 2,  # 2ms
    "tau_ampa": 5,
    "tau_nmda": 100,
    "tau_gaba": 10,
    "tau_a": 100,  # 100ms - GOOD
    "tau_b": 20 * 10**3,  # 20s
    "tau_d": 200,
    "tau_f": 600,
    "delta_a": 0.1,  # decay unit
    "delta_b": 5 * 10**-4,  # seconds
    "U_exc": 0,
    "U_inh": -80,
    "alpha_exc": 0.2,
    "alpha_inh": 0.3,
    "learning_rate": 2 * 10**-5,
    "gamma": 4,  # Target population rate in Hz (this might be wrong)
    "num_items": 16,
    "dt": 1,  # time unit for modelling
    "T": 1000,  # total time each item will appear
    "wp": 0.5,
    "V_rest": -60,
    "min_weight": 0,  # minimum allowed weight
    "max_weight": 5,  # maximum allowed weight
    "num_epochs": 1,  # number of epochs -> not currently in use
    "A": 1 * 10**-3,  # LTP rate
    "B": 1 * 10**-3,  # LTD rate
    "beta": 0.05,
    "delta": 2 * 10**-5,  # Transmitter triggered plasticity
    "tau_cons": 1.8 * 10**6,  # 30 minutes until weight convergence
    "euler": 5,
    "U_cons": 0.2,
}

# Initiate SNN object
snn = SNN_STDP(**snn_params)

# Initialize & visualize pre-trained network
(
    MemPot,
    spikes,
    W_exc,
    W_inh,
    W_exc_ideal,
) = snn.initialize_network(
    N_input_neurons=484,
    N_excit_neurons=484,
    N_inhib_neurons=121,
    radius_=4,
    W_ee_prob=0.1,
    retur=True,
)
# Visualize network
snn.vis_network(heatmap=False, weight_layer=False)

# Generate data
snn.gen_data(
    N_classes=4,
    noise_rand=True,
    noise_variance=0.05,
    mean=0,
    blank_variance=0.01,
    save=True,
    retur=True,
    avg_high_freq=45,
    avg_low_freq=10,
    var_high_freq=0.05,
    var_low_freq=0.05,
)  # Need to add on/off period here

# Visualize data
snn.visualize_data(single_data=False, raster_plot_=False, alt_raster_plot=True)

# Train network
(
    spikes,
    MemPot,
    W_exc,
) = snn.train_data(
    retur=True,
    save_model=True,
    item_lim=20,
    force_retrain=True,
)

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=True,
    idx_start=484,
    idx_stop=605,
    mv=True,
    overlap=True,
    traces=True,
    tsne=True,
)
