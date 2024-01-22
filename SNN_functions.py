#SNN simple three input neurons and one output neuron

# Import libraries
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import small_world_network as swn
#os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.02, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=0.1, V_rest=-70, leakage_rate=0.99, num_neurons=20,
                 excit_inhib_ratio = 0.8, FF_FB_ratio = 0.7, alpha=1, perc_input_neurons=0.1, interval=0.03,
                 max_weight=1, min_weight=-1):
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
        self.leakage_rate = leakage_rate 
        self.num_items = num_items
        self.num_neurons = num_neurons
        self.perc_input_neurons = perc_input_neurons
        self.excit_inhib_ratio = excit_inhib_ratio
        self.FF_FB_ratio = FF_FB_ratio
        self.alpha = alpha
        self.interval = interval
        self.max_weight = max_weight
        self.min_weight = min_weight

    def initialize_network(self):
        # Generate weights 
        self.weights = swn.generate_small_world_network_power_law(num_neurons=self.num_neurons, 
                                                             excit_inhib_ratio=self.excit_inhib_ratio, 
                                                             FF_FB_ratio=self.FF_FB_ratio, alpha=self.alpha,
                                                             perc_input_neurons=0.1)
        # Generate membrane potential and spikes array
        self.MemPot = np.zeros(shape=(self.num_timesteps, self.num_neurons, self.num_items))
        self.MemPot[0,:,0] = self.V_rest
        self.t_since_spike = np.zeros(shape=(self.num_timesteps, self.num_neurons, self.num_items))

    def prep_data(self):
        # Simulate data
        self.data = self.encode_input_poisson(np.random.rand(self.num_items, self.num_neurons))
    
    def neuronal_activity(self):
        for l in tqdm(range(self.num_items), desc="Processing items"):
            for t in range(1, self.num_timesteps):
                for n in range(0,self.num_neurons):
                    # Check if neuron is an input neuron
                    if (self.weights[:,:,1] < 0).any():
                        I_in = self.data[t,n,l]
                    else:
                        # Calculate the sum of incoming input from the previous step
                        spikes = int(self.t_since_spike[t-1,:,l] == 0) # This might be completely incorrect
                        I_in = np.dot(self.Weights[n,:,0],spikes.T)
                    
                    # Update membrane potential
                    self.MemPot[t,n,l] = self.MemPot[t-1, n, l] + (-self.MemPot[t-1, n, l] + self.V_rest \
                                                               + I_in * self.dt / self.tau_m) / self.tau_m
                    # Update spikes
                    if self.MemPot[t,n,l] > self.V_th:
                        self.t_since_spike[t,n,l] = 0
                        self.MemPot[t,n,l] = self.V_reset
                    else:
                        self.t_since_spike[t,n,l] += 1 

                    # Perform STDP for hidden neurons
                    for s in range(0,self.num_neurons):
                        if s != n:
                            # Calculate the spike diff for input and output neuron
                            spike_diff = self.t_since_spike[t,n,l] - self.t_since_spike[t,s,l]
                            # Check if excitatory or inhibitory 
                            if self.weights[n,s,0] > 0:
                                if spike_diff > 0:
                                    self.weights[n,s,0] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                                else:
                                    self.weights[n,s,0] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                            elif self.weights[n,s,0] < 0:
                                if spike_diff < 0:
                                    self.weights[n,s,0] -= self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                                else:
                                    self.weights[n,s,0] -= self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                    
            # Perform clipping of weights
            self.weights = np.clip(self.weights, self.min_weight, self.max_weight)

    def encode_input_poisson(self, input):
        # 2D-array: items x neurons
        poisson_input = np.zeros((self.num_timesteps, self.num_neurons, self.num_items))

        for i in range(self.num_items):
            for j in range(self.num_neurons):
                # Calculate the mean spike count for the Poisson distribution
                # Assuming 'input' is the rate (spikes/sec), we multiply by 'dt' to get the average number of spikes per time step
                lambda_poisson = input[i, j]*self.dt

                # Generate spikes using Poisson distribution
                for t in range(self.num_timesteps):
                    spike_count = np.random.poisson(lambda_poisson)
                    poisson_input[t, j, i] = 1 if spike_count > 0 else 0

        return poisson_input

    def visualize_learning(self, spikes, V):
        num_neurons = spikes.shape[1]
        
        # Define colors for each neuron
        colors = plt.cm.jet(np.linspace(0, 1, num_neurons))  # Using the jet colormap

        # Plotting Spike Raster Plot
        plt.figure(figsize=(12, 8))
        
        # Subplot for Spike Raster Plot
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        spike_data = []
        for neuron in range(num_neurons):
            neuron_spike_times = np.where(spikes[:, neuron] == 1)[0]
            spike_data.append(neuron_spike_times)

        # Set lineoffsets and linelengths for spacing
        lineoffsets = np.arange(num_neurons)
        linelengths = 0.8  # Adjust this value to control the length of spikes

        plt.eventplot(spike_data, lineoffsets=lineoffsets, linelengths=linelengths, colors=colors)
        plt.yticks(lineoffsets, labels=[f'Neuron {i}' for i in range(num_neurons)])
        plt.xlabel('Time')
        plt.ylabel('Neuron')
        plt.title('Spike Raster Plot')

        # Subplot for Membrane Potential Plot
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        for neuron in range(num_neurons):
            plt.plot(V[:, neuron], label=f'Neuron {neuron}', color=colors[neuron])
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Membrane Potential Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
    



