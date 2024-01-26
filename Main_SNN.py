# Main SNN execution file

# Set cw
import os
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
#os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
from SNN_functions import *

# Initialize SNN object
snn = SNN_STDP(V_th=-55, V_reset=-75, C=1, R=1, A_minus=-0.1, tau_m=0.01, num_items=10, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=0.1, V_rest=-70, num_neurons=5,
                 excit_inhib_ratio = 0.8, alpha=1, perc_input_neurons=0.1, 
                 interval=0.03, max_weight=1, min_weight=-1, input_scaler=400)

# Prepare data
data = snn.prep_data()
print(data)
# Initialize network
MemPot, t_since_spik, weights = snn.initialize_network()

snn.visualize_network()

# Train network
snn.neuronal_activity()

# Visualize results & network
snn.visualize_learning()


