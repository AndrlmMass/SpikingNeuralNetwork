# Main SNN execution file

# Set cw
import os

os.chdir(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
)
# os.chdir(
#    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork"
# )

# Import libraries
from main.SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(
    V_th=-65,
    V_reset=-75,
    C=1,
    R=1,
    tau_m=0.01,
    num_neurons=3600,
    num_items=20,
    tau_stdp=0.01,
    dt=0.001,
    T=1,
    V_rest=-70,
    excit_inhib_ratio=0.8,
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
)

# Initialize & visualize pre-trained network
MemPot, t_since_spik, W_se, W_ee, W_ei, W_ie = snn.initialize_network(
    N_input_neurons=1600,
    N_excit_neurons=1600,
    N_inhib_neurons=400,
    radius=2,
    W_ie_prob=0.1,
    retur=True,
)

# Load data
snn.load_data()

snn.visualize_network(drw_edg=False)

# Train network
avg_spike_counts = snn.neuronal_activity()

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
snn.plot_training(plt_mV=False)
