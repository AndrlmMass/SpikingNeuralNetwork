# Main SNN execution file

# Set cw
import os
import sys

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\src"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "main"))

from SNN_functions import SNN_STDP

# Initiate SNN object
snn = SNN_STDP()

# Initialize network
snn.initialize_network()

# Generate data
snn.gen_data()

# Train network
snn.train_()

# Plot training
snn.plot_training()
