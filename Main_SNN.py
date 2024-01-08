# Main SNN execution file

# Set cw
import os
os.chdir('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN_scripts_folder')

# Import libraries
from SNN_functions import *
import tensorflow as tf

# Initialize neurons
snn = SNN_STDP()
MemPot, Spikes, Ws = snn.gen_neuron_layer()

# Prepare data
train_simpl = snn.prep_data_()
train_enc = snn.encode_input(train_simpl)

# Train network
spikes, Vt, W_ff = snn.neuronal_activity(W_ff=Ws, spikes=Spikes, X=train_enc, V=MemPot)

# Visualize results
visualize_learning(spikes,Vt)

def visualize_learning(spikes, V):
        num_neurons = spikes.shape[1]
        num_timesteps = spikes.shape[0]
        
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

