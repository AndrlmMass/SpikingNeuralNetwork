#SNN simple three input neurons and one output neuron

# Import libraries
import os
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
#os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

class LIFNeuron:
    """ Leaky Integrate-and-Fire Neuron model """
    def __init__(self, threshold=-55.0, dt=0.001, rest_potential=-70, cell_effect="excitatory", cell_type="input", V_reset=-75):
        self.threshold = threshold
        self.potential = [rest_potential]
        self.V_reset = V_reset
        self.V_rest = rest_potential
        self.W = random.random()
        self.spikes = [0]
        self.cell_effect = cell_effect

    def receive_spike(self, t=1, inp=0):
        I_ff = self.W*self.spikes[t-1]
        I_in = inp+I_ff
        self.potential = self.potential[t-1] * np.exp(-self.dt / self.tau_m) + (self.V_rest * (1 - np.exp(-self.dt / self.tau_m)) + I_in * self.R)

    def update(self, t):
        if self.potential >= self.threshold:
            self.spikes[t] = 1
            self.potential = self.V_reset
        else:
            self.spikes[t] = 0

    
    def update_weight(self, t=0, spike_diff=0, A_plus=0.1, A_minus=-0.1):
        if self.cell_effect == "excitatory":
            if self.spikes[t] == 1:
                # Here spike diff = current_spike - output neuron spike
                if spike_diff < 0:
                    self.W += A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                else:
                    self.W += A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
        else:
            if self.spikes[t] == 1:
                if spike_diff < 0:
                    self.W += A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                else:
                    self.W += A_minus * np.exp(abs(spike_diff) / self.tau_stdp)

class LIF_Neuron:
     def __init__(self, total_time, dt, num_items, weights):
          self.membrane_potential = np.zeros((total_time/dt)*num_items)
          self.spikes = np.zeros((total_time/dt)*num_items)
          self.weights = weights 
    
     def update_membrane_potential(self, new_Vm, t):
          self.membrane_potential[t] = new_Vm
    
     def update_spikes(self, new_spike, t):
          self.spikes[t] = new_spike
    
     def update_weights(self, new_weight, id):
          self.weights[id] = new_weight

     def get_membrane_potential(self,t):
         return self.membrane_potential
     
     def get_spikes(self,t):
         return self.spikes
     
     def get_weights(self):
         return self.weights

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.02, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=3.0, V_rest=-70, leakage_rate=0.99, 
                 num_input_neurons=3, num_hidden_neurons=3, num_output_neurons=1):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = tau_m
        self.tau_stdp = tau_stdp
        self.dt = dt
        self.T = T
        self.V_rest = V_rest
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.leakage_rate = leakage_rate
        self.num_items = num_items
        self.num_input_neurons = num_input_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_output_neurons = num_output_neurons

    def initiate_network(self, num_input_neurons, num_hidden_neurons, hid_inhib):
        #Initiate input neurons
        inp_weights = np.ones(num_hidden_neurons)
        inhibitory_indices_inp = random.sample(range(num_hidden_neurons), hid_inhib)
        self.inp_labels = [-1 if i in inhibitory_indices_inp else 1 for i in range(num_input_neurons)]
        self.input_neurons = [LIF_Neuron(total_time=self.T,dt=self.dt, num_items=self.num_items,  
                                        weights=inp_weights) for _ in range(num_input_neurons)]

        #Initiate hidden neurons
        hid_weights = np.random.rand(num_input_neurons)
        inhibitory_indices_hid = random.sample(range(num_hidden_neurons), hid_inhib)
        self.hid_labels = [-1 if i in inhibitory_indices_hid else 1 for i in range(num_hidden_neurons)]
        self.hidden_neurons = [LIF_Neuron(total_time=self.T, dt=self.dt, num_items=self.num_items,
                                        weights=hid_weights) for _ in range(num_hidden_neurons)]
        
        #Initiate output neurons
        out_weights = np.random.rand(num_hidden_neurons)
        inhibitory_indices_out = random.sample(range(num_hidden_neurons), hid_inhib)
        self.out_labels = [-1 if i in inhibitory_indices_out else 1 for i in range(num_hidden_neurons)]
        self.hidden_neurons = LIF_Neuron(total_time=self.T, dt=self.dt, num_items=self.num_items,
                                        PSP_effect=None, weights=out_weights)
        
    def neuronal_activity(self, Ws, spikes, X, V):
        [num_items, _, num_timesteps] = X.shape
        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                for n in range(0,self.num_input_neurons):
                    # initiate current neuron
                    input_neuron = self.input_neurons[n]
                    
                    # update membrane potential
                    I_in = input_neuron.get_spikes()
                    Vm = input_neuron.get_membrane_potential[t-1]*np.exp(-self.dt/self.tau_m) + (self.V_rest * (1 - np.exp(-self.dt /self.tau_m)) + I_in*self.R)
                    input_neuron.update_membrane_potential(new_Vm = Vm, t=t)

                    # update spikes based on Vm
                    if Vm > self.V_th:
                        input_neuron.update_spikes(new_spike=1,t=t)
                        input_neuron.update_membrane_potential(new_Vm = self.v_reset, t=t)
                    else:
                        input_neuron.update_spikes(new_spike=0,t=t)
                    
                    # update weights with STDP


                # Constrain weights to prevent them from growing too large
                Ws = np.clip(Ws, a_min=0, a_max=1)
        
        return spikes, V, Ws

    def previous_spike(self, X, current_idx, current_layer_size, next_layer_size):
        spike_checks = current_layer_size
        prev_spike = np.zeros(shape=())

    
    def closest_spike_idx(self, output_idx, current_idx, spikes_arr):
        for i in range(current_idx,0,-1):
            if spikes_arr[i,output_idx]:
                return i
            else:
                return None
            
    def prep_data_(self):
        training_simplified = np.random.rand(self.num_items, 3)
        return training_simplified

    def encode_input_poisson(self, input):
        num_inputs, num_neurons = input.shape
        self.num_steps = int(self.T / self.dt)

        # 3D array: inputs x neurons x time steps
        poisson_input = np.zeros((num_inputs, num_neurons, self.num_steps))

        for i in range(num_inputs):
            for j in range(num_neurons):
                # Calculate the mean spike count for the Poisson distribution
                # Assuming 'input' is the rate (spikes/sec), we multiply by 'dt' to get the average number of spikes per time step
                lambda_poisson = input[i, j]*self.dt*400

                # Generate spikes using Poisson distribution
                for t in range(self.num_steps):
                    spike_count = np.random.poisson(lambda_poisson)
                    poisson_input[i, j, t] = 1 if spike_count > 0 else 0

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
    



