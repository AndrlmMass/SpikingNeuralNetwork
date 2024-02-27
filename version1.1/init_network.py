# Initialize snn

# Import relevant libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class gen_weights:
    def gen_StimE(self, radius, N_input_neurons, N_excit_neurons):
        input_shape = int(np.sqrt(N_input_neurons))
        circle_pos = np.arange(N_input_neurons).reshape(input_shape, input_shape)
        circle_pos_valid = circle_pos

        if circle_pos_valid.size == 0:
            raise ValueError("circle_pos_valid has invalid shape")

        circle_pos_flat = circle_pos_valid.flatten()
        circle_draws = np.random.choice(a=circle_pos_flat, size=N_excit_neurons)
        StimE_weights = np.zeros((N_input_neurons, N_excit_neurons))

        for j in range(N_excit_neurons):
            center_idx = np.argwhere(circle_pos == circle_draws[j])[
                0
            ]  # Find the 2D index of the center

            # Calculate the bounds for slicing around the center with the given radius
            # Ensure bounds are within the array limits
            row_start = max(0, center_idx[0] - radius)
            row_end = min(input_shape, center_idx[0] + radius+1)
            col_start = max(0, center_idx[1] - radius)
            col_end = min(input_shape, center_idx[1] + radius+1)

            # Example operation: for each selected position, set a weight in EE_weights
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    StimE_weights[circle_pos[row, col], j] = np.random.uniform(
                        low=0, high=1
                    )  

        return StimE_weights

    def gen_EI_IE_EE_weights(self, N_excit_neurons, N_inhib_neurons):    
        # I assume N_excit_neurons is equal to N_inhib_neurons
        EI_weights = np.random.rand(N_excit_neurons)

        # Inhibitory to excitatory array to increase signal to noise ratio
        IE_weights = np.random.rand(N_inhib_neurons, N_excit_neurons)

        # Excitatory to excitatory connections
        EE_weights = np.random.rand(N_excit_neurons, N_excit_neurons)

        return EI_weights, IE_weights, EE_weights

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




