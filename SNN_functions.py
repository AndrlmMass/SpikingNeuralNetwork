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

class LIF_Neuron:
     def __init__(self, total_time, dt, num_items, weights, weight_labels):
          self.membrane_potential = np.zeros((total_time/dt)*num_items)
          self.spikes = np.zeros((total_time/dt)*num_items)
          self.weights = weights
          self.weights_labels = weight_labels 
    
     def update_membrane_potential(self, new_Vm, t):
          self.membrane_potential[t] = new_Vm
    
     def update_spikes(self, new_spike, t):
          self.spikes[t] = new_spike
    
     def update_weights(self, new_weight, id):
          self.weights[id] = new_weight

     def get_membrane_potential(self):
         return self.membrane_potential
     
     def get_spikes(self):
         return self.spikes
     
     def get_weights(self):
         return self.weights, self.weight_labels

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.02, num_items=100, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=3.0, V_rest=-70, leakage_rate=0.99):
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
    
    def neuronal_activity(self, Ws, spikes, X, V):
        [num_items, _, num_timesteps] = X.shape
        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                for n in range(0,self.num_input_neurons):
                    # update membrane potential
                    I_in = spikes(X[l,n,t]) #Is this right?
                    Vm = membrane_potential[t-1]*np.exp(-self.dt/self.tau_m) + (self.V_rest * (1 - np.exp(-self.dt /self.tau_m)) + I_in*self.R)
                    input_neuron.update_membrane_potential(new_Vm = Vm, t=t)

                    # update spikes based on Vm
                    if Vm > self.V_th:
                        input_neuron.update_spikes(new_spike=1,t=t)
                        input_neuron.update_membrane_potential(new_Vm = self.v_reset)
                    else:
                        input_neuron.update_spikes(new_spike=0,t=t)
                        input_neuron.update_membrane_potential(new_Vm = membrane_potential[t-1], t=t)
                    
                    # update weights with STDP
                    for w_idx in range(0,weights):
                        #Calculate spike diff
                        spike_diff = spike_diff(input_spikes=)
                        if weights[w_idx] >= 0:
                                
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
    
    def initiate_network(self, num_neurons, inhib_excit_ratio):
        # Initiate data
        Z = np.zeros(self.T/self.dt, 3, self.num_items)
        Z[0,2,0] = [self.V_rest]*num_neurons
        
        # Initiate weights
        self.weights = self.generate_small_world_network(num_neurons=num_neurons, inhib_excit_ratio=inhib_excit_ratio)

    def spike_diff(output_spikes, input_spikes, t):
        output_spike_idx, input_spike_idx = None, None
        for j in range(t,0,-1):
            if output_spikes[j] == 1:
                output_spike_idx = j
                break
        for t in range(t,0,-1):
            if input_spikes[t] == 1:
                input_spike_idx = t
        return output_spike_idx-input_spike_idx

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
    
def generate_small_world_network(num_neurons, inhib_excit_ratio):
    # Array dimensions
    #self.num_neurons = num_neurons
    n_rows, n_cols = num_neurons, num_neurons

    # Probabilities for multinomial distribution: positive float, zero, negative float
    probabilities = [0.4, 0.2, 0.4]  # Sum should be 1.0

    # Generate multinomial distribution for the entire matrix
    multinomial_distribution = np.random.multinomial(1, probabilities, size=n_rows*n_cols)

    # Initialize the array
    array = np.zeros((n_rows, n_cols))

    # Iterate through the matrix and assign values based on the distribution
    for i in range(n_rows):
        for j in range(n_cols):
            distribution_result = multinomial_distribution[i*n_cols + j]
            if distribution_result[0] == 1:  # Positive float
                array[i, j] = [np.random.uniform(0, 1), np.random.choice(-1,1)]
            elif distribution_result[2] == 1:  # Negative float
                array[i, j] = [np.random.uniform(-1, 0), np.random.choice(-1,1)]
            else:
                array[i, j] = [0, np.random.choice(-1,1)]


    # Adjusting the ratio of positive to negative floats to 80:20
    # Count the number of positive and negative values
    num_positives = np.sum(array > 0)
    num_negatives = np.sum(array < 0)

    # Desired ratio
    desired_ratio = inhib_excit_ratio

    # Number of positives and negatives to achieve the desired ratio
    total_non_zeros = num_positives + num_negatives
    desired_num_positives = int(total_non_zeros * desired_ratio)
    desired_num_negatives = total_non_zeros - desired_num_positives

    # Adjust the matrix to achieve the desired ratio
    # If there are too many positives, randomly convert some to negatives
    while num_positives > desired_num_positives:
        i, j = np.random.randint(0, n_rows), np.random.randint(0, n_cols)
        if array[i, j] > 0:
            array[i, j] = np.random.uniform(-1, 0)
            num_positives -= 1

    # If there are too many negatives, randomly convert some to positives
    while num_negatives > desired_num_negatives:
        i, j = np.random.randint(0, n_rows), np.random.randint(0, n_cols)
        if array[i, j] < 0:
            array[i, j] = np.random.uniform(0, 1)
            num_negatives -= 1

    return array


