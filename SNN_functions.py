#SNN simple three input neurons and one output neuron

# Import libraries
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork')
#os.chdir('C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork')

# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(self, V_th=-55, V_reset=-75, C=10, R=1, A_minus=-0.1, tau_m=0.02, 
                 tau_stdp=0.02, A_plus=0.1, dt=0.001, T=0.5, V_rest=-70, leakage_rate=0.99):
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




