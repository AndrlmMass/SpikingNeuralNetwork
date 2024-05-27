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

import SNN_functions
from SNN_functions import SNN_STDP

# Store the parameters in a dictionary
snn_params = {
    "V_th": -50,
    "V_reset": -70,
    "P": 20,
    "C": 1,
    "R": 100,
    "tau_plus": 0.5,
    "tau_minus": 1,
    "tau_slow": 0.07,
    "tau_m": 0.525,
    "tau_mm": 0.125,
    "tau_ht": 0.15,
    "tau_hom": 0.157,
    "tau_stdp": 0.1,
    "tau_H": 10,
    "learning_rate": 0.001,
    "gamma": 0.1,
    "num_items": 16,
    "dt": 0.001,
    "T": 0.1,
    "wp": 0.5,
    "V_rest": -60,
    "min_weight": 0,
    "max_weight": 5,
    "num_epochs": 1,
    "init_cals": 1,
    "A": 0.01,
    "beta": 0.05,
    "delta": 0.02,
    "tau_const": 10,
    "euler": 5,
}

# Initiate SNN object
snn = SNN_STDP(**snn_params)

# Initialize & visualize pre-trained network
(
    MemPot,
    spikes,
    W_se,
    W_ee,
    W_ei,
    W_ie,
    W_se_ideal,
    W_ee_ideal,
    W_ei_ideal,
    W_ie_ideal,
) = snn.initialize_network(
    N_input_neurons=484,
    N_excit_neurons=484,
    N_inhib_neurons=121,
    radius_=4,
    W_ee_prob=0.1,
    retur=True,
)

# Generate data
snn.gen_data(
    N_classes=4,
    noise_rand=True,
    noise_rand_ls=[0, 0.01, 0.03, 0.05],
    mean=0,
    blank_variance=0.01,
    input_scaler=20,
    save=True,
    retur=False,
)

# Load data
data, labels = snn.load_data(rand_lvl=0.05, retur=True)

# Visualize network


# Train network
(
    spikes,
    MemPot,
    W_se,
    W_ee,
    W_ei,
    W_ie,
) = snn.train_data(
    retur=True,
    save_model=True,
    item_lim=24,
)

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=False,
    idx_start=484,
    idx_stop=600,
    mv=False,
    overlap=False,
    traces=False,
    tsne=True,
)

snn.plot_I_in()
