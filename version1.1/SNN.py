# Set cd
import os

#os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.1")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.1"
)

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        N_input_neurons,
        N_excit_neurons,
        N_inhib_neurons,
        delay, #determine the delay from STDP to weight change(?)
        EE_plastic,
        EI_static,
        IE_plastic,
        II_static,
        StimE_plastic,
        
    ):
    