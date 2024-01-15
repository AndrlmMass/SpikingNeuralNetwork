#SNN simple three input neurons and one output neuron

# Import libraries
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
#os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

class LIFNeuron:
    """ Leaky Integrate-and-Fire Neuron model """
    def __init__(self, threshold=-55.0, dt=0.001, V_rest =-70, V_reset=-75, cell_type = ["input"], neuron_ID = 0, num_connections=3, exhib_labels=np.array([1,1,-1])):
        self.threshold = threshold
        self.num_connections = num_connections
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.potential = [self.V_rest]
        if cell_type != "input":
            self.weights = np.random.rand(self.num_connections) # These are the weights for the incoming
        self.spikes = []
        self.output_or_hidden = cell_type
        self.neuron_ID = neuron_ID
        self.weight_labels = exhib_labels

    def update_spike(self, current_potential):
        if current_potential >= self.threshold:
            self.potential.append(self.V_reset)
            self.spikes.append(1)
        else:
            self.spikes.append(0)
            self.potential.append(self.potential[-1])

    def update_membrane_potential(self, t, input_data, l):
        '''
        t = time unit
        X = previous spiking activity
        l = num_unit (only relevant for input neurons)
        '''
        # Update this part to make sure the weights are applied correctly
        if self.cell_type != "input":
            # Update weights based on weight labels (i.e., whether they are excitatory or inhibitory)
            weights_ = np.dot(self.weights,self.weight_labels)
            I_in = np.sum(np.dot(weights_,self.spikes[t-1]))
            self.potential[t] = self.potential[t-1] * np.exp(-self.dt / self.tau_m) + (self.V_rest * (1 - np.exp(-self.dt / self.tau_m)) + I_in * self.R)
        else:
            I_in = input_data[l,self.neuron_ID]+self.spikes[t-1] # Not sure if I should keep the previous spike in the input neuron to calculate the Vm or not
            self.potential[t] = self.potential[t-1] * np.exp(-self.dt / self.tau_m) + (self.V_rest * (1 - np.exp(-self.dt / self.tau_m)) + I_in * self.R)

    def previous_spike(self,X, current_idx):
        prev_spike = np.zeros(shape=(4))
        if self.cell_type == "hidden" or self.cell_type == "output":
            # update time since spike for main neuron
            for i in range(current_idx,0,-1):
                if self.spikes[i]:
                    prev_spike[0] = i
                else:
                    prev_spike[0] = None
        
        # update time since spike for projecting neurons
        for l in range(current_idx,0,-1):
            for t in range(0,3):
                if X[i,t] == 1:
                    prev_spike[t+1] = l
                else:
                    prev_spike[1] = None
        
        return prev_spike
    
    
    def STDP_W_update(self):
        
        # Chec
        for weight in self.weight:
            # update weights for excitatory connections
            if self.weight_label:
                if self.cell_type == "hidden":
                    if self.spike_diff < 0:
                        self.weight[self.neuron_ID] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                    else:
                        self.weight[self.neuron_ID] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
            # Update weights for inhibitory connections
            else:
                if self.cell_type == "hidden":
                    if self.spike_diff < 0:
                        self.weight[self.neuron_ID] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                    else:
                        self.weight[self.neuron_ID] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)



# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.02, 
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

    def prep_data_(self):
        training_simplified = np.random.rand(1, 3)
        return training_simplified

    # Initialize neuronal & weight layers
    def gen_neuron_layer(self, input_neurons=3, output_neurons=1):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.total_neurons = input_neurons+output_neurons

        # Initialize membrane potential array 10000 x 4
        MembranePotentials = np.full((int(self.T/self.dt), self.total_neurons), self.V_rest)
        
        # Initialize spike array (binary) as 10000 x 4
        Spikes = np.zeros((int(self.T/self.dt), self.total_neurons))

        # Random initial weights (3 x 1)
        Weights = np.random.rand(input_neurons)
        self.weights = Weights

        return MembranePotentials, Spikes, Weights
    
    # Encode inputs into spike trains
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


    # Update the neuronal_activity method in the SNN_STDP class
    def neuronal_activity(self, Ws, spikes, X, V):
        [num_items, _, num_timesteps] = X.shape
        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                # Calculate feedforward current 
                I_ff = np.dot(Ws, spikes[t-1, 0:3])
                
                # Calculate total input current for all input neurons
                I_in = X[l, :, t] + I_ff

                # Update membrane potential with exponential decay
                V[t, 0:3] = V[t-1, 0:3] * np.exp(-self.dt / self.tau_m) + \
                            (self.V_rest * (1 - np.exp(-self.dt / self.tau_m)) + I_in * self.R)
                V[t, 3] = V[t-1, 3] * np.exp(-self.dt / self.tau_m) + \
                        (self.V_rest * (1 - np.exp(-self.dt / self.tau_m)) + I_ff * self.R)

                # Generate spikes
                spike_indices = np.where(V[t, :] > self.V_th)[0]
                spikes[t, spike_indices] = 1
                V[t, spike_indices] = self.V_reset

                # STDP learning rule
                prev_out_spike_idx = self.closest_spike_idx(output_idx=3, current_idx=t, spikes_arr=spikes)
                if prev_out_spike_idx is not None:
                    spike_diff = t - prev_out_spike_idx
                    for neuron_id in range(self.input_neurons):
                        if spikes[t, neuron_id] == 1:
                            # Perform STDP
                            if spike_diff < 0:
                                Ws[neuron_id] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                            else:
                                Ws[neuron_id] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
                # Constrain weights to prevent them from growing too large
                Ws = np.clip(Ws, a_min=0, a_max=1)
        
        return spikes, V, Ws

    
    def closest_spike_idx(self, output_idx, current_idx, spikes_arr):
        for i in range(current_idx,0,-1):
            if spikes_arr[i,output_idx]:
                return i
            else:
                return None

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
    



