# Set cd
import os

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.1")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.1"
)

# Import directory sensitive functions
from init_network import *


# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        N_input_neurons,
        N_excit_neurons,
        N_inhib_neurons,
    ):
        self.N_input_neurons = N_input_neurons
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons

    def init_network(self):
        gw = gen_weights()
        """
          Generate Stimulation to excitatory neuron weights (StimE weights)
        """
        StimE_W = gw.gen_StimE(self.N_input_neurons, self.N_excit_neurons)

        """ 
        Generate the rest of the weights (i.e., excitatory to excitatory; EE, 
        # excitatory to inhibitory; EI, and inhibitory to excitatory; IE) 
        """
        EI_W, IE_W, EE_W = gw.gen_EI_IE_EE_weights(
            self.N_excit_neurons, self.N_inhib_neurons
        )

        """ 
        Create array for spike and membrane potential storage
        """

    def gen_data(self):
        data = 11
