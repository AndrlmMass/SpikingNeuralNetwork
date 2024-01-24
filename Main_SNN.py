# Main SNN execution file

# Set cw
import os
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(V_th=-55, V_reset=-90, C=10, R=1, A_minus=-0.1, tau_m=0.02, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=0.1, V_rest=-70, leakage_rate=0.99, num_neurons=20,
                 excit_inhib_ratio = 0.8, FF_FB_ratio = 0.7, alpha=1, perc_input_neurons=0.1, interval=0.03,
                 max_weight=1, min_weight=-1)

# Prepare data
snn.prep_data()

# Initialize network
snn.initialize_network()

# Train network
snn.neuronal_activity()

# Visualize results & network
snn.visualize_learning(num_items_to_plot=3, num_neurons_to_plot=10)
snn.visualize_network()

