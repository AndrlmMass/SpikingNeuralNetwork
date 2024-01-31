# Main SNN execution file

# Set cw
import os
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
#os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(V_th=0.7, V_reset=-0.2, C=10, R=1, A_minus=-0.001, tau_m=0.02, num_items=100, 
                 tau_stdp=0.01, A_plus=0.001, dt=0.001, T=0.1, V_rest=0, num_neurons=20,
                 excit_inhib_ratio = 0.8, alpha=1, num_input_neurons=4, 
                 max_weight=1, min_weight=-1, input_scaler=2.5)

# Prepare data
prep_data = snn.prepping_data(base_mean=10, mean_increment=3, variance=1)

# Initialize network
MemPot, t_since_spik, weights = snn.initialize_network()

# Train network
avg_spike_counts, count = snn.neuronal_activity()
print(f"This training had {count[0]} excitatory strengthenings and {count[1]} weakenings. While inhibitory connections had {count[2]} strenghtenings and {count[3]} weakenings.")

# Visualize network
#snn.visualize_network()

# Visualize training
snn.plot_training(num_neurons=20, num_items=100, num_weights=10)

