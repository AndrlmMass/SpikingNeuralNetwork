# Main SNN execution file

# Set cw
import os
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')

# Import libraries
from SNN_functions import *
import tensorflow as tf

# Initialize neurons
snn = SNN_STDP()
MemPot, Spikes, Ws = snn.gen_neuron_layer()

# Prepare data
train_simpl = snn.prep_data_()
train_enc = snn.encode_input(train_simpl)

# Train network
spikes, Vt, W_ff = snn.neuronal_activity(Ws=Ws, spikes=Spikes, X=train_enc, V=MemPot)

# Visualize results
snn.visualize_learning(spikes,Vt)

