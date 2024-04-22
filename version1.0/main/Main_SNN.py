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
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(
    V_th=-55,
    V_reset=-60,
    P=1,  # Don't know starting value
    C=1,  # Don't know starting value, also needs updating
    R=1,  # Don't know starting value, also needs updating
    tau_m=100,  # Don't know starting value
    num_items=8,
    tau_stdp=0.1,  # Don't know starting value
    dt=0.001,
    T=0.1,
    V_rest=-65,
    alpha=1,  # Don't know starting value
    min_weight=0,
    max_weight=5,
    num_epochs=100,  # N/A
    init_cals=1,  # N/A
    A=1,  # Regulates hebbian learning -> larger == more hebbian learning
    B=0.1,  # Regulates hebbian learning -> larger == less hebbian learning
    beta=1,  # Regulates heterosynpatic learning
    delta=10,  # Regulates dopamin_reg
)

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
    input_scaler=10,
    save=False,
    retur=False,
)

# Load data
data, labels = snn.load_data(rand_lvl=0.01, retur=True)

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
    interactive_tool=False,
    update_frequency=5,
)

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
snn.plot_training(
    ws_nd_spikes=False,
    idx_start=0,
    idx_stop=605,
    mv=True,
    time_start=0,
    time_stop=50,
)
