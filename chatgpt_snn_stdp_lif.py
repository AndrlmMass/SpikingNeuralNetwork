import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    """ Leaky Integrate-and-Fire Neuron model """
    def __init__(self, threshold=1.0, leak_factor=0.99):
        self.threshold = threshold
        self.leak_factor = leak_factor
        self.potential = 0.0

    def receive_spike(self, weight):
        self.potential += weight

    def update(self):
        if self.potential >= self.threshold:
            self.potential = 0
            return 1
        self.potential *= self.leak_factor
        return 0

class SpikingNeuralNetwork:
    """ Spiking Neural Network with STDP and Feedback Loop """
    def __init__(self, num_input_neurons, num_hidden_neurons, learning_rate=0.01, initial_weight=2.0):
        self.input_neurons = [LIFNeuron() for _ in range(num_input_neurons)]
        self.hidden_neurons = [LIFNeuron() for _ in range(num_hidden_neurons)]
        self.output_neuron = LIFNeuron()
        self.input_to_hidden_weights = np.full((num_input_neurons, num_hidden_neurons), initial_weight)
        self.hidden_to_output_weights = np.full(num_hidden_neurons, initial_weight)
        self.output_to_input_weights = np.full(num_input_neurons, initial_weight)
        self.learning_rate = learning_rate

    def feed_forward(self, input_spikes):
        for i, input_spike in enumerate(input_spikes):
            if input_spike > 0:
                for j in range(len(self.hidden_neurons)):
                    self.hidden_neurons[j].receive_spike(self.input_to_hidden_weights[i][j])
        for j in range(len(self.hidden_neurons)):
            hidden_spike = self.hidden_neurons[j].update()
            hidden_spikes[j] = hidden_spike
            if hidden_spike > 0:
                self.output_neuron.receive_spike(self.hidden_to_output_weights[j])
        output_spike = self.output_neuron.update()
        if output_spike > 0:
            for i in range(len(self.input_neurons)):
                self.input_neurons[i].receive_spike(self.output_to_input_weights[i])
        return input_spikes, hidden_spikes, output_spike

    def apply_stdp(self, input_spikes, hidden_spikes, output_spike):
        # STDP for Input to Hidden Layer
        for i in range(len(self.input_neurons)):
            for j in range(len(self.hidden_neurons)):
                if input_spikes[i] > 0 and hidden_spikes[j] > 0:
                    self.input_to_hidden_weights[i][j] += self.learning_rate  # Potentiation
                elif input_spikes[i] > 0:
                    self.input_to_hidden_weights[i][j] -= self.learning_rate  # Depression
        # STDP for Hidden to Output Layer
        for j in range(len(self.hidden_neurons)):
            if hidden_spikes[j] > 0 and output_spike > 0:
                self.hidden_to_output_weights[j] += self.learning_rate  # Potentiation
            elif hidden_spikes[j] > 0:
                self.hidden_to_output_weights[j] -= self.learning_rate  # Depression

# Example Usage
snn = SpikingNeuralNetwork(num_input_neurons=3, num_hidden_neurons=2)

# Simulate the network
for _ in range(100):  # Simulation steps
    input_spikes = np.random.randint(0, 2, 3)  # Random spikes for input neurons
    input_spikes, hidden_spikes, output_spike = snn.feed_forward(input_spikes)
    snn.apply_stdp(input_spikes, hidden_spikes, output_spike)

# Visualization (code similar to previous example)
# Redefining the neuron and network classes to store additional data for visualization
class LIFNeuronViz(LIFNeuron):
    """ Extended LIFNeuron to record potentials """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_history = []

    def update(self):
        spike = super().update()
        self.potential_history.append(self.potential)
        return spike

class SpikingNeuralNetworkViz(SpikingNeuralNetwork):
    """ Extended SpikingNeuralNetwork to record spikes """
    def __init__(self, num_input_neurons, num_hidden_neurons, learning_rate=0.01, initial_weight=2.0):
        super().__init__(num_input_neurons, num_hidden_neurons, learning_rate, initial_weight)
        self.input_neurons = [LIFNeuronViz() for _ in range(len(self.input_neurons))]
        self.hidden_neurons = [LIFNeuronViz() for _ in range(len(self.hidden_neurons))]
        self.output_neuron = LIFNeuronViz()
        self.spikes_history = []


    def feed_forward(self, input_spikes):
        self.spikes_history.append(input_spikes)
        return super().feed_forward(input_spikes)
    
# Creating the network with visualization capabilities
snn_viz = SpikingNeuralNetworkViz(num_input_neurons=3, num_hidden_neurons=2)

# Simulate the network again
for _ in range(100):
    input_spikes = np.random.randint(0, 2, 3)
    input_spikes, hidden_spikes, output_spike = snn_viz.feed_forward(input_spikes)
    snn_viz.apply_stdp(input_spikes, hidden_spikes, output_spike)


# Preparing data for raster plot
input_spikes_data = np.array(snn_viz.spikes_history)
    
# Correcting the code to properly determine the output spikes
output_spikes_data = [1 if snn_viz.output_neuron.potential_history[i] == 0 and snn_viz.output_neuron.potential_history[i-1] != 0 else 0 for i in range(1, len(snn_viz.output_neuron.potential_history))]

# Plotting the raster plot for spikes
plt.figure(figsize=(12, 6))

# Raster plot
plt.subplot(2, 1, 1)
plt.eventplot([np.where(input_spikes_data[:,i])[0] for i in range(input_spikes_data.shape[1])], lineoffsets=1, linelengths=1)
plt.eventplot(np.where(output_spikes_data)[0], lineoffsets=4, linelengths=1, color="red")
plt.xlabel("Time Steps")
plt.ylabel("Neuron")
plt.title("Raster Plot of Neuron Spiking")
plt.yticks([1, 2, 3, 4], ["Input 1", "Input 2", "Input 3", "Output"])
plt.grid(True)

# Voltage plot
plt.subplot(2, 1, 2)
for neuron in snn_viz.input_neurons:
    plt.plot(neuron.potential_history, label="Input Neuron")
plt.plot(snn_viz.output_neuron.potential_history, label="Output Neuron", color="red")
plt.xlabel("Time Steps")
plt.ylabel("Membrane Potential")
plt.title("Membrane Potential Over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
