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
    C=1,
    R=1,
    tau_m=0.01,
    num_items=20,
    tau_stdp=0.01,
    dt=0.001,
    T=1,
    V_rest=-70,
    alpha=1,
    max_weight=5,
    min_weight=0,
    input_scaler=2,
    num_epochs=100,  # N/A
    init_cals=1,  # N/A
    target_weight=0,
    A=1,
    B=1,
    beta=1,
    delta=1,
)

# Initialize & visualize pre-trained network
snn.initialize_network(
    N_input_neurons=1600,
    N_excit_neurons=160,
    N_inhib_neurons=40,
    radius_=2,
    W_ee_prob=0.1,
    retur=False,
)
# Generate data
snn.gen_data()

# Load data
snn.load_data(rand_lvl=0.01, retur=False)

# Visualize network


# Train network
snn.train_data(retur=False, w_p=0.5)

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
