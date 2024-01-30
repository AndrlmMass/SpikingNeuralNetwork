# Main SNN execution file

# Set cw
import os
#os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(V_th=-55, V_reset=-90, C=10, R=1, A_minus=-0.1, tau_m=0.02, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=1, V_rest=-70, num_neurons=20,
                 excit_inhib_ratio = 0.8, alpha=1, num_input_neurons=4, 
                 interval=0.03, max_weight=1, min_weight=-1, input_scaler=10)

# Prepare data
prep_data = snn.prepping_data(base_mean=10, mean_increment=3, variance=1)

# Initialize network
MemPot, t_since_spik, weights = snn.initialize_network()

# Train network
snn.neuronal_activity()

# Visualize network
snn.visualize_network()

# Visualize training
snn.plot_training(num_neurons=10, num_items=20)

