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
    def __init__(self, threshold=-55.0, V_rest =-70, V_reset=-75, cell_type = "input", 
                 num_connections=3, exhib_labels=np.array([1,1,-1])):
        self.threshold = threshold
        self.num_connections = num_connections
        self.V_reset = V_reset
        self.V_rest = V_rest
        self.potential = np.zeros()
        self.spikes = np.zeros()
        if cell_type != "input":
            self.weights = np.random.rand(self.num_connections) # These are the weights for the incoming
        else:
            if exhib_labels != [1,1,1]:
                raise "Warning, you are initiating weights for input with non-identity values. This is not recommended"
        self.spikes = []
        self.output_or_hidden = cell_type
        self.weight_labels = exhib_labels

    def update_spike(self):
        if self.potential >= self.threshold:
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
        
        '''
        This return a 1x4 array, where the first element is the index of the previous time the 
        main neuron had a spike, and the three next elements are the indexes of those respective times
        '''

        prev_spike = np.zeros(shape=(4))
        # update time since spike for main neuron
        for i in range(current_idx,0,-1):
            if self.spikes[i]:
                prev_spike[0] = i
            else:
                prev_spike[0] = None
        
        # update time since spike for projecting neurons
        for l in range(current_idx,0,-1):
            for t in range(0,3):
                if X[l,t] == 1:
                    prev_spike[t+1] = l
                else:
                    prev_spike[t+1] = None
        return prev_spike
    
    def STDP_W_update(self,X,t):
        
        '''
        This function updates the weights based on the timing 
        between the main neuron and its connective neurons. 
        '''

        self.prev_spike = self.spike_diff(X=X,current_idx=t)

        for w_idx in range(0,len(self.prev_spike)):
            #Calculate spike difference
            spike_diff = self.prev_spike[0]-self.prev_spike[w_idx]
            # update weights for excitatory connections
            if self.weight_label[w_idx-1] > 0: #Not sure if w_idx-1 or w_idx
                if self.spike_diff < 0:
                    self.weight[w_idx] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                else:
                    self.weight[w_idx] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
            # update weights for inhibitory connections
            else:
                if self.spike_diff > 0:
                    self.weight[w_idx] += self.A_minus * np.exp(abs(spike_diff) / self.tau_stdp)
                else:
                    self.weight[w_idx] += self.A_plus * np.exp(abs(spike_diff) / self.tau_stdp)
        
        # Clip weights to make them smaller
        self.weights = np.clip(self.weights, a_min=0, a_max=1)


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

    def initiate_network(self, network_size=[3,3,1], hidden_exhib=2):
        #Initiate input neurons
        self.input_neurons = LIFNeuron(num_connections=0, exhib_labels=[1,1,1])
        #Initiate hidden neurons
        self.input_neurons = []
        for i in range(network_size[1]):
            hidden_exhib = [1]*hidden_exhib+[-1]
            self.input_neurons.append(LIFNeuron(exhib_labels=))
        [LIFNeuron() for _ in range(network_size[0])]
        self.hidden_neurons = [LIFNeuron() for _ in range(network_size[1])] # Would be nice to initate a more than 1-dim sized array of hidden neurons
        self.output_neurons = LIFNeuron()

        self.  




    def prep_data_(self, num_data_points):
        training_simplified = np.random.rand(num_data_points, 3)
        return training_simplified
    
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
    def neuronal_activity(self, X):
        [num_items, _, num_timesteps] = X.shape
        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                # update membrane potential & spike
                self.input_neurons.update_membrane_potential(t=self.dt, input_data=X, l=l).update_spike()
                self.hidden_neurons.update_membrane_potential(t=self.dt, input_data=X, l=l).update_spike()
                self.output_neurons.update_membrane_potential(t=self.dt, input_data=X, l=l).update_spike()
                
                # apply STDP learning rule
                self.hidden_neurons.previous_spike().STDP_W_update()
                self.output_neurons.previous_spike().STDP_W_update()

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
    



