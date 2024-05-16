# Main SNN execution file

# Set cw
import os
import sys

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
else:
    os.chdir(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
import SNN_functions
from SNN_functions import SNN_STDP

# Store the parameters in a dictionary
snn_params = {
    "V_th": -50,
    "V_reset": -70,
    "P": 20,
    "C": 1,
    "R": 100,
    "tau_plus": 0.1,
    "tau_minus": 0.1,
    "tau_slow": 0.1,
    "tau_m": 0.125,
    "tau_ht": 0.15,
    "tau_hom": 0.157,
    "tau_stdp": 0.1,
    "tau_H": 10,
    "learning_rate": 0.00001,
    "gamma": 0.1,
    "num_items": 4,
    "dt": 0.001,
    "T": 0.1,
    "V_rest": -60,
    "min_weight": 0,
    "max_weight": 5,
    "num_epochs": 1,
    "init_cals": 1,
    "A": 0.01,
    "beta": 0.5,
    "delta": 0.002,
    "tau_const": 10,
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
    run=False,
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
    w_p=0.5,
    update_frequency=5,
    save_model=True,
)

# Reload model
W_se, W_ee, W_ie, W_ei, spikes, mempot = snn.reload_model()

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=True,
    idx_start=484,
    idx_stop=600,
    mv=False,
    overlap=True,
)

snn.plot_I_in()
