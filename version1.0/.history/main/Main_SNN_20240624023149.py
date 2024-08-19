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
    "P": 20,
    "C": 1,
    "U": 0.2,
    "tau_plus": 20,
    "tau_minus": 20,
    "tau_slow": 100,
    "tau_m": 20,
    "tau_ht": 100,
    "tau_hom": 1.2 * 10**6,  # metaplasticity time constant
    "tau_istdp": 20,
    "tau_H": 6 * 10**2,  # 10s
    "tau_thr": 2,  # 2ms
    "tau_ampa": 5,
    "tau_nmda": 100,
    "tau_gaba": 10,
    "tau_a": 100,  # 100ms
    "tau_b": 2 * 10**4,  # 20s
    "tau_d": 200,
    "tau_f": 600,
    "delta_a": 0.112,  # decay unit
    "delta_b": 5 * 10**-4,  # seconds
    "U_exc": 0,
    "U_inh": -80,
    "alpha_exc": 0.2,
    "alpha_inh": 0.3,
    "learning_rate": 2 * 10**-5,
    "gamma": 4,
    "num_items": 36,
    "dt": 0.001,
    "T": 1,
    "wp": 0.5,
    "V_rest": -60,
    "min_weight": 0,
    "max_weight": 5,
    "num_epochs": 1,
    "A": 1 * 10**-3,  # LTP rate
    "B": 1 * 10**-3,  # LTD rate
    "beta": 0.05,
    "delta": 2 * 10**-5,
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
    noise_rand_ls=[0.05],
    mean=0,
    blank_variance=0.01,
    save=True,
    retur=False,
    avg_high_freq=45,
    avg_low_freq=10,
    var_high_freq=0.05,
    var_low_freq=0.05,
)  # Need to add off/on period here

# Load data
data, labels = snn.load_data(rand_lvl=0.05, retur=True)

# Visualize data
snn.visualize_data(run=False)

# Train network
(
    spikes,
    MemPot,
    W_exc,
) = snn.train_data(
    retur=True,
    save_model=True,
    item_lim=20,
)

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=True,
    idx_start=484,
    idx_stop=600,
    mv=True,
    overlap=False,
    traces=True,
    tsne=True,
)
