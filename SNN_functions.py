#SNN simple three input neurons and one output neuron

# Import libraries
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

import plot_training as pt
import plot_network as pn
import Gen_weights_nd_data as gwd

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.002, num_items=100, num_input_neurons=4,
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=1, V_rest=-70, num_neurons=20, excit_inhib_ratio = 0.8, 
                 alpha=1, interval=0.03, max_weight=1, min_weight=-1, input_scaler=100):
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
        self.num_classes = num_input_neurons
        self.excit_inhib_ratio = excit_inhib_ratio
        self.alpha = alpha
        self.interval = interval
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.input_scaler = input_scaler
        self.num_input_neurons = num_input_neurons

    def initialize_network(self):
        # Generate weights 
        self.weights = gwd.generate_small_world_network_power_law(num_neurons=self.num_neurons, 
                                                                  excit_inhib_ratio=self.excit_inhib_ratio, 
                                                                  alpha=self.alpha,
                                                                  num_input_neurons=self.num_input_neurons,
                                                                  num_items=self.num_items, num_timesteps=self.num_timesteps)
        # Generate membrane potential and spikes array
        self.MemPot = np.zeros(shape=(self.num_timesteps, self.num_neurons, self.num_items))
        self.MemPot[0,:,:] = self.V_rest
        self.t_since_spike = np.ones(shape=(self.num_timesteps, self.num_neurons, self.num_items))
        return self.MemPot, self.t_since_spike, self.weights

    def prepping_data(self, base_mean, mean_increment, variance, ):
        # Simulate data
        self.data, self.classes = gwd.generate_multidimensional_data(self.num_classes, base_mean, 
                                        mean_increment, variance, self.num_items, self.num_input_neurons,
                                        self.num_timesteps, self.num_items, self.dt, self.input_scaler)
        return self.data, self.classes
    
    def neuronal_activity(self):
        spike_counts = np.zeros((self.num_neurons, self.num_items))

        # Add input data before training
        input_indices = np.where(np.all(self.weights[:,:,0] == 0, axis=1))[0]
        print(self.data.shape)
        for j in range(len(input_indices)):
            for t in range(self.num_timesteps):
                for i in range(self.num_items):
                    if self.data[t,j,i] == 1:
                        self.t_since_spike[t,j,i] = 0
                    else:
                        self.t_since_spike[t,j,i] = self.t_since_spike[t-1,j,i] + 1

        for l in tqdm(range(self.num_items), desc="Training network"):
            for t in range(1, self.num_timesteps):
                for n in range(self.num_neurons):
                    # Check if neuron is an input neuron
                    if (self.weights[n,:,l-1] != 0).any():
                        spikes = (self.t_since_spike[t-1,:,l] == 0).astype(int) 
                        I_in = np.dot(self.weights[n,:,l-1],spikes.T)
                    
                        # Update equation
                        self.MemPot[t,n,l] = self.MemPot[t-1,n,l] + (-self.MemPot[t-1,n,l] + self.V_rest + self.R * I_in) / self.tau_m * self.dt

                        # Update spikes
                        if self.MemPot[t,n,l] > self.V_th:
                            self.t_since_spike[t,n,l] = 0
                            self.MemPot[t,n,l] = self.V_reset
                            spike_counts[n,l] += 1
                        else:
                            self.t_since_spike[t,n,l] = self.t_since_spike[t-1,n,l] + 1

                    # Perform STDP for hidden neurons
                    for s in range(0,self.num_neurons):
                        if s != n and (self.t_since_spike[t,n,l] == 0 or self.t_since_spike[t,s,l] == 0):
                            # Calculate the spike diff for input and output neuron
                            spike_diff = self.t_since_spike[t,n,l] - self.t_since_spike[t,s,l]
                            # Check if excitatory or inhibitory 
                            if self.weights[n,s,l-1] > 0:
                                if spike_diff > 0:
                                    self.weights[n,s, l] += round(self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp),4)
                                else:
                                    self.weights[n,s,l] += round(self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp),4)
                            elif self.weights[n,s,l-1] < 0:
                                if spike_diff < 0:
                                    self.weights[n,s,l] -= round(self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp),4)
                                else:
                                    self.weights[n,s,l] -= round(self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp),4)

            # Clip weights to avoid exploding or diminishing gradients
            self.weights[:,:,l] = np.clip(self.weights[:,:,l], a_max=self.max_weight, a_min=self.min_weight)

            # Update self.MemPot to include the membrane potential from the previous step
            # as the beginning of the next step. 
            if l < self.num_items-1:
                self.MemPot[0,:,l+1] = self.MemPot[t,:,l]

        # Calculate average spike count for each neuron per item
        avg_spike_counts = spike_counts / self.num_timesteps

        return avg_spike_counts

    def visualize_network(self, drw_edg = True, drw_netw = True):
        if drw_netw:
            pn.draw_network(self.weights)
        if drw_edg:
            pn.draw_edge_distribution(self.weights)

    def plot_training(self, num_neurons, num_items):
        pt.plot_spikes(num_neurons_to_plot=num_neurons, num_items_to_plot=num_items, t_since_spike=self.t_since_spike)
        pt.plot_weights(self.weights, num_weights=10)

    



