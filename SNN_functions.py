#SNN simple three input neurons and one output neuron

# Import libraries
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import small_world_network as swn
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
#os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.002, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=1, V_rest=-70, num_neurons=20, excit_inhib_ratio = 0.8, 
                 alpha=1, perc_input_neurons=0.1, interval=0.03, max_weight=1, min_weight=-1, input_scaler=1000):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = tau_m
        self.tau_stdp = tau_stdp
        self.dt = dt
        self.T = T
        self.num_timesteps = int(T/dt)
        self.V_rest = V_rest
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.leakage_rate = 1/self.R
        self.num_items = num_items
        self.num_neurons = num_neurons
        self.perc_input_neurons = perc_input_neurons
        self.excit_inhib_ratio = excit_inhib_ratio
        self.alpha = alpha
        self.interval = interval
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.input_scaler = input_scaler

    def initialize_network(self):
        # Generate weights 
        self.weights = swn.generate_small_world_network_power_law(num_neurons=self.num_neurons, 
                                                                  excit_inhib_ratio=self.excit_inhib_ratio, 
                                                                  alpha=self.alpha,
                                                                  perc_input_neurons=0.1)
        # Generate membrane potential and spikes array
        self.MemPot = np.zeros(shape=(self.num_timesteps, self.num_neurons, self.num_items))
        self.MemPot[0,:,:] = self.V_rest
        self.t_since_spike = np.ones(shape=(self.num_timesteps, self.num_neurons, self.num_items))
        return self.MemPot, self.t_since_spike, self.weights

    def prep_data(self):
        # Simulate data
        self.data = self.encode_input_poisson(np.random.rand(self.num_items, self.num_neurons))
        return self.data
    
    def neuronal_activity(self):
        for l in tqdm(range(self.num_items), desc="Processing items"):
            for t in range(1, self.num_timesteps):
                for n in range(0,self.num_neurons):
                    # Check if neuron is an input neuron
                    if (self.weights[n,:] != 0).any():
                        spikes = (self.t_since_spike[t-1,:,l] == 0).astype(int) 
                        I_in = np.dot(self.weights[n,:],spikes.T)
                    
                        # Update equation
                        self.MemPot[t,n,l] = self.MemPot[t-1,n,l] + (-self.MemPot[t-1,n,l] + self.V_rest + self.R * I_in) / self.tau_m * self.dt

                        # Update spikes
                        if self.MemPot[t,n,l] > self.V_th:
                            self.t_since_spike[t,n,l] = 0
                            self.MemPot[t,n,l] = self.V_reset
                        else:
                            self.t_since_spike[t,n,l] += 1
                    else:
                        # Update input neuron spike
                        if self.data[t,n,l] == 1:
                            self.t_since_spike[t,n,l] = 0
                        else:
                            self.t_since_spike[t,n,l] += 1

                    # Perform STDP for hidden neurons
                    for s in range(0,self.num_neurons):
                        if s != n and (self.t_since_spike[t,n,l] == 0 or self.t_since_spike[t,s,l] == 0):
                            # Calculate the spike diff for input and output neuron
                            spike_diff = self.t_since_spike[t,n,l] - self.t_since_spike[t,s,l]
                            # Check if excitatory or inhibitory 
                            if self.weights[n,s] > 0:
                                if spike_diff > 0:
                                    self.weights[n,s] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                                else:
                                    self.weights[n,s] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                            elif self.weights[n,s] < 0:
                                if spike_diff < 0:
                                    self.weights[n,s] -= self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                                else:
                                    self.weights[n,s] -= self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)

            # Update self.MemPot to include the membrane potential from the previous step
            # as the beginning of the next step. 
            if l < self.num_items-1:
                self.MemPot[0,:,l+1] = self.MemPot[t,:,l]

    def encode_input_poisson(self, input):
        # 2D-array: items x neurons
        poisson_input = np.zeros((self.num_timesteps, self.num_neurons, self.num_items))

        for i in range(self.num_items):
            for j in range(self.num_neurons):
                # Calculate the mean spike count for the Poisson distribution
                # Assuming 'input' is the rate (spikes/sec), we multiply by 'dt' to get the average number of spikes per time step
                lambda_poisson = input[i, j]*self.dt*self.input_scaler

                # Generate spikes using Poisson distribution
                for t in range(self.num_timesteps):
                    spike_count = np.random.poisson(lambda_poisson)
                    poisson_input[t, j, i] = 1 if spike_count > 0 else 0

        return poisson_input

    def visualize_learning(self):
        num_items = self.t_since_spike.shape[2]
        num_neurons = self.t_since_spike.shape[1]

        # Define colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_neurons))  # Using the jet colormap

        # Create a figure with subplots for each item
        fig, axs = plt.subplots(num_items, 2, figsize=(12, 8 * num_items))  # 2 columns for each item

        spike_data = []
        for item in range(num_items):
            for neuron in range(num_neurons):
                # Find the time steps where the neuron fired
                neuron_spike_times = np.where(self.t_since_spike[:, neuron, item] == 0)[0]
                spike_data.append(neuron_spike_times)

        # Set lineoffsets and linelengths for spacing
        lineoffsets = np.arange(num_neurons)
        linelengths = 0.8  # Adjust this value to control the length of spikes

        # Spike Raster Plot for each item
        axs[item, 0].eventplot(spike_data, lineoffsets=lineoffsets, linelengths=linelengths, colors=colors)
        axs[item, 0].set_yticks(lineoffsets)
        axs[item, 0].set_yticklabels([f'Neuron {i}' for i in range(num_neurons)])
        axs[item, 0].set_xlabel('Time')
        axs[item, 0].set_ylabel('Neuron')
        axs[item, 0].set_title(f'Item {item+1} - Spike Raster Plot')

        plt.tight_layout()
        plt.show()











    def visualize_network(self, drw_edg = True, drw_netw = True):
        if drw_netw:
            swn.draw_network(self.weights)
        if drw_edg:
            swn.draw_edge_distribution(self.weights)


    



