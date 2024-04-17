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
    V_th=-65,
    V_reset=-75,
    P=1,  # Don't know starting value
    C=1,  # Don't know starting value, also needs updating
    R=1,  # Don't know starting value, also needs updating
    tau_m=0.01,  # Don't know starting value
    num_items=8,
    tau_stdp=0.01,  # Don't know starting value
    dt=0.001,
    T=0.1,
    V_rest=-70,
    alpha=1,  # Don't know starting value
    min_weight=0,
    max_weight=5,
    input_scaler=2,
    num_epochs=100,  # N/A
    init_cals=1,  # N/A
    A=1,  # Don't know starting value, also needs updating
    B=1,  # Don't know starting value, also needs updating
    beta=1,  # Don't know starting value, also needs updating
    delta=1,  # Don't know starting value, also needs updating
)

# Initialize & visualize pre-trained network
snn.initialize_network(
    N_input_neurons=484,
    N_excit_neurons=484,
    N_inhib_neurons=121,
    radius_=4,
    W_ee_prob=0.1,
    retur=False,
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
snn.load_data(rand_lvl=0.01, retur=False)

# Visualize network


# Train network
snn.train_data(
    retur=False,
    w_p=0.5,
    plot_spikes=True,
    plot_weights=True,
    interactive_tool=True,
    update_frequency=100,
)

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
