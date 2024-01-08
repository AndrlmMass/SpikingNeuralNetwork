#SNN simple three input neurons and one output neuron

# Import libraries
import os
from tqdm import tqdm
import time 
import numpy as np
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN_scripts_folder')


# Initialize class variable
class SNN_STDP():
    
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
        # Import data
        #mnist = tf.keras.datasets.mnist
        # Prepare data
        #(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        #del test_images, test_labels
        #train_images = train_images/train_images.max()
        #training, labels = train_images[:10].reshape(10,28*28), train_labels[:10]
        training_simplified = np.random.rand(100, 3)
        
        return training_simplified

    # Initialize neuronal & weight layers
    def gen_neuron_layer(self, num_neurons=3):
        self.num_neurons = num_neurons
        # Initialize membrane potential array
        MembranePotentials = np.full((int(self.T/self.dt), num_neurons), self.V_rest)
        
        # Initialize spike array (binary)
        Spikes = np.zeros((int(self.T/self.dt), self.num_neurons))

        # Random initial weights
        Weights = np.random.rand(num_neurons, num_neurons)

        return MembranePotentials, Spikes, Weights
    
    # Encode inputs into spike trains
    def encode_input(self, input):
        num_inputs, num_neurons = input.shape
        self.num_steps = int(self.duration / self.dt)

        # 3D array: inputs x neurons x time steps
        fixed_input = np.zeros((num_inputs, num_neurons, self.num_steps))

        input *= 100000

        for i in range(num_inputs):
            for j in range(num_neurons):
                spike_prob = input[i, j] * self.dt
                spikes = np.random.rand(self.num_steps) < spike_prob
                fixed_input[i, j, :] = spikes
        return fixed_input


    # Neuronal activity
    def neuronal_activity(self, W_ff, spikes, X, V, tau_stdp=0.02, A_plus=0.01):
        [num_items, num_neurons, num_timesteps] = X.shape

        '''
        W_ff   = feedforward weights (num_neurons x num_neurons)
        spikes = spikes array (num_neurons x num_timesteps)
        X      = spike-encoded array (samples x features x timesteps)
        V      = membrane potential (num_neurons x num_timesteps)
        '''

        for l in tqdm(range(num_items), desc="Processing items"):
            for t in range(1, num_timesteps):
                # Calculate feedforward current 
                I_ff = np.dot(W_ff, spikes[t-1, :])
                
                # Calculate total input current for all neurons
                I_in = X[l, :, t] + I_ff

                # Update membrane potential
                V[t, :] = V[t-1, :] + (-V[t-1, :] + self.V_rest + I_in * self.R * self.dt) / self.tau_m

                # Generate spikes
                spike_indices = np.where(V[t, :] > self.V_th)[0]
                spikes[t, spike_indices] = 1
                V[t, spike_indices] = self.V_reset

                # Refractory period
                refractory_indices = np.where(spikes[t-1, :] == 1)[0]
                V[t, refractory_indices] = self.V_rest

                # STDP learning rule
                for ip in range(num_neurons):
                    for jp in range(num_neurons):
                        if spikes[t-1, ip] == 1 and spikes[t, jp] == 1:
                            delta_w = A_plus * np.exp(-(t - 1 - tau_stdp) / tau_stdp)
                            W_ff[ip, jp] += delta_w

        return spikes, V, W_ff

    def visualize_learning(self, spikes, V):
        num_neurons = spikes.shape[1]
        num_timesteps = spikes.shape[0]
        
        # Plotting Spike Raster Plot
        plt.figure(figsize=(12, 8))
        
        # Subplot for Spike Raster Plot
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        spike_times = []
        neuron_ids = []
        for neuron in range(num_neurons):
            for time in range(num_timesteps):
                if spikes[time, neuron] == 1:
                    spike_times.append(time)
                    neuron_ids.append(neuron)
        plt.eventplot([np.array(spike_times)[np.array(neuron_ids) == i] for i in range(num_neurons)])
        plt.xlabel('Time')
        plt.ylabel('Neuron')
        plt.title('Spike Raster Plot')

        # Subplot for Membrane Potential Plot
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        for neuron in range(num_neurons):
            plt.plot(V[:, neuron], label=f'Neuron {neuron}')
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




