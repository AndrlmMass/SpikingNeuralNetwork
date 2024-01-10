# Main SNN execution file

# Set cw
import os
#os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
import importlib

from SNN_functions import *

# Initialize neurons
snn = SNN_STDP()
MemPot, Spikes, Ws = snn.gen_neuron_layer()

# Prepare data
train_simpl = snn.prep_data_()
train_enc = snn.encode_input_poisson(train_simpl)

# Train network
spikes, Vt, W_ff = snn.neuronal_activity(Ws=Ws, spikes=Spikes, X=train_enc, V=MemPot)
#importlib.reload(SNN_functions.SNN_STDP)

# Visualize results
snn.visualize_learning(spikes,Vt)

