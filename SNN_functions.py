#SNN simple three input neurons and one output neuron

# Import libraries
import os
from tqdm import tqdm
import time 
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-50, V_reset=-65, C=10, R=1, 
                 dt=0.001, T=0.1, V_rest=-70, duration=0.1):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = C*R/100
        self.dt = dt
        self.T = T
        self.V_rest = V_rest
        self.duration = duration 

    def prep_data_(self):
        training_simplified = np.random.rand(100, 3)
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

        return MembranePotentials, Spikes, Weights
    
    # Encode inputs into spike trains
    def encode_input(self, input):
        num_inputs, num_neurons = input.shape
        self.num_steps = int(self.duration / self.dt)

        # 3D array: inputs x neurons x time steps
        fixed_input = np.zeros((num_inputs, num_neurons, self.num_steps))

        input *= 10000 #Adjust based on plotting afterwards

        for i in range(num_inputs):
            for j in range(num_neurons):
                spike_prob = input[i, j] * self.dt
                spikes = np.random.rand(self.num_steps) < spike_prob
                fixed_input[i, j, :] = spikes
        return fixed_input


    # Neuronal activity
    def neuronal_activity(self, Ws, spikes, X, V):
        [num_items, _, num_timesteps] = X.shape

        '''
        Ws     = weights (output_neurons x input_neurons)
        Spikes = spikes array (num_timesteps x total_neurons)
        X      = spike-encoded array (samples x timesteps x total_neurons)
        V      = membrane potential (num_timesteps x total_neurons)
        '''

        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                # Calculate feedforward current 
                I_ff = np.dot(Ws, spikes[t-1, 0:3])
                
                # Calculate total input current for all neurons
                I_in = X[l, :, t] + I_ff
                I_out = np.sum(I_in)

                # Update membrane potential
                V[t, 0:3] = V[t-1, 0:3] + (-V[t-1, 0:3] + self.V_rest + I_in * self.dt / self.tau_m) / self.tau_m
                V[t, 3] = V[t-1, 3] + (-V[t-1, 3] + self.V_rest + I_out * self.dt / self.tau_m) / self.tau_m

                # Generate spikes
                spike_indices = np.where(V[t, :] > self.V_th)[0]
                spikes[t, spike_indices] = 1
                V[t, spike_indices] = self.V_reset

                # Refractory period
                refractory_indices = np.where(spikes[t-1, :] == 1)[0]
                V[t, refractory_indices] = self.V_rest

        return spikes, V, Ws

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
    
    # Generate network structure based on the Barabasi-Albert model
    def create_scale_free_network(num_neurons, m):
        G = nx.barabasi_albert_graph(num_neurons, m)
        Weights = np.zeros((num_neurons, num_neurons))

        for i, j in G.edges():
            Weights[i, j] = np.random.rand()

        return G, Weights

    def plot_network(G, title='Scale-Free Network'):
        # Draw the network
        pos = nx.spring_layout(G)  # Positions for all nodes
        degrees = dict(G.degree())

        # Scale node size by the degree (number of connections)
        node_size = [v * 10 for v in degrees.values()]
        nx.draw(G, pos, with_labels=False, node_size=node_size, node_color='lightblue', edge_color='gray')
        plt.title(title)
        plt.show()

    # Example usage
    #num_neurons = 250  # Number of neurons
    #m = 2  # Number of edges to attach from a new node to existing nodes
    #G = create_scale_free_network(num_neurons, m)

    # Plot the network
    #plot_network(G)




