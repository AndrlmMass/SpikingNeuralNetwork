# Set cd
import os

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.1")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.1"
)

# Import directory sensitive libraries
from init_network import *
from training import *


# Initialize class variable
class SNN_STDP:
    # Initialize neuronal parameters
    def __init__(
        self,
        g_ampa_max,
        g_nmda_max,
        g_gaba_max,
        tau_ampa,
        tau_nmda,
        tau_gaba,
        V_reset,
        E_leak,
        V_rest,
        tau_m,
        items,
        Vm_th,
        E_inh,
        E_ex,
        dt,
        T,
    ):

        self.time = np.arange(0, self.items, dt)
        self.timesteps = int(self.T / self.dt)
        self.g_ampa_max = g_ampa_max
        self.g_nmda_max = g_nmda_max
        self.g_gaba_max = g_gaba_max
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.tau_gaba = tau_gaba
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.E_leak = E_leak
        self.tau_m = tau_m
        self.items = items
        self.E_inh = E_inh
        self.Vm_th = Vm_th
        self.E_ex = E_ex
        self.dt = dt
        self.T = T

    def init_network(self, radius, N_input_neurons, N_excit_neurons, N_inhib_neurons):
        self.N_neurons = N_input_neurons + N_inhib_neurons + N_excit_neurons
        # Generate weights from init_network script
        gw = gen_weights()
        self.StimE_Ws = gw.gen_StimE(radius, N_input_neurons, N_excit_neurons)
        self.EI, self.IE, self.II, self.EE = gw.gen_EI_IE_EE_weights(
            N_excit_neurons, N_inhib_neurons
        )

        # Set g_neurotransmitter arrays
        g_ampa = np.zeros((self.N_neurons, self.time.shape))  # Conductance for AMPA
        g_nmda = np.zeros((self.N_neurons, self.time.shape))  # Conductance for NMDA
        g_gaba = np.zeros((self.N_neurons, self.time.shape))  # Conductance for GABA

        # Generate membrane potential (mV) and spikes array
        self.mV = np.full((self.N_neurons, self.time.shape), self.V_rest)
        self.spikes = np.zeros((self.N_neurons, self.time.shape))

    def training(self):
        training_network(
            self.items,
            self.timesteps,
            self.dt,
            self.mV,
            self.StimE_Ws,
            self.EI,
            self.IE,
            self.II,
            self.EE,
        )

    def testing(self):
        dfdfs
