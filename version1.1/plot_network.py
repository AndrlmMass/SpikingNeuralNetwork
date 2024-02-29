# Plot network

# Import relevant libraries 
import numpy as np     
import matplotlib.pyplot as plt

# Plot heatmap
def draw_heatmap(self, EE_weights, N_input_neurons):
        # Aggregate weights for each input neuron
        input_weights_sum = np.sum(EE_weights, axis=1)
        
        # Reshape to 2D input space
        input_shape = int(np.sqrt(N_input_neurons))
        weights_matrix = input_weights_sum.reshape(input_shape, input_shape)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(weights_matrix, cmap='Reds', interpolation='nearest')
        plt.colorbar(label='Input Intensity')
        plt.title('Heatmap of Input Space')
        plt.xlabel('Input Neuron X Coordinate')
        plt.ylabel('Input Neuron Y Coordinate')
        plt.show()