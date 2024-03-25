# Main SNN execution file

# Set cw
import os
import numpy as np

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork\\version1.0"
)

# Import libraries
from main.SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(
    V_th=0.7,
    V_reset=-0.7,
    R=1,
    tau_m=0.7,
    num_items=100,
    tau_stdp=50,
    A_plus=0.004,
    A_minus=-0.004,
    dt=0.00005,
    T=0.01,
    V_rest=0.1,
    num_neurons=20,
    init_cals=700,
    excit_inhib_ratio=0.8,
    alpha=0.2,
    num_input_neurons=4,
    max_weight=1,
    min_weight=-1,
    input_scaler=1,
)

# Prepare data
prep_data = snn.prepping_data(base_mean=10, mean_increment=3, variance=1)

# Initialize & visualize pre-trained network
MemPot, t_since_spik, W_se, W_ee, W_ei, W_ie = snn.initialize_network()

# Load data
snn.load_data()

snn.visualize_network(drw_edg=False)

# Train network
avg_spike_counts = snn.neuronal_activity()

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
snn.plot_training(plt_mV=False)
