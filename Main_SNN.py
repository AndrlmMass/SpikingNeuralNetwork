# Main SNN execution file

# Set cw
import os

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork"
)

# Import libraries
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(
    V_th=0.7,
    V_reset=-0.2,
    R=1,
    A_minus_in=-1,
    tau_m=0.1,
    num_items=10,
    tau_stdp=1,
    A_plus_in=0.007,
    A_plus_ex=1,
    A_minus_ex=-0.0035,
    dt=0.001,
    T=0.1,
    V_rest=0.1,
    num_neurons=20,
    init_cals=700,
    excit_inhib_ratio=0.8,
    alpha=1,
    num_input_neurons=4,
    max_weight=1,
    min_weight=-1,
    input_scaler=1.5,
    num_classes=2,
)

# Prepare data
prep_data = snn.prepping_data(base_mean=10, mean_increment=3, variance=1)

# Initialize & visualize pre-trained network
MemPot, t_since_spik, weights = snn.initialize_network()
snn.visualize_network()

# Train network
avg_spike_counts = snn.neuronal_activity()

# Evaluate performance

# Test network on unseen data and estimate performance

# Visualize training and testing results
snn.plot_training()
