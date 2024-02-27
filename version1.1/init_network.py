# Initialize snn

# Import relevant libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def init_network(N_input_neurons, N_excit_neurons, N_inhib_neurons):

    # N_input_neurons must be equal to N_excit_neurons
    if N_input_neurons != N_excit_neurons:
        raise ValueError("you messed up fool")




class gen_weights:
    def gen_EE(self, radius, N_input_neurons, N_excit_neurons):
        input_shape = int(np.sqrt(N_input_neurons))
        circle_pos = np.arange(N_input_neurons).reshape(input_shape, input_shape)
        circle_pos_valid = circle_pos[
            radius : -radius, radius: -radius
        ]

        if circle_pos_valid.size == 0:
            raise ValueError("circle_pos_valid has invalid shape")

        circle_pos_flat = circle_pos_valid.flatten()
        circle_draws = np.random.choice(a=circle_pos_flat, size=N_excit_neurons)
        EE_weights = np.zeros((N_input_neurons, N_excit_neurons))

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
            print(f"row start: {row_start}, row end: {row_end}, col start: {col_start}, and col end: {col_end}")

            # Example operation: for each selected position, set a weight in EE_weights
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    print(circle_pos[row,col])
                    EE_weights[circle_pos[row, col], j] = np.random.uniform(
                        low=0, high=1
                    )  

        return EE_weights

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

# Example usage:
N_input_neurons = 100  # 10x10 input space
N_excit_neurons = 50
radius = 2

gen = gen_weights()
EE_weights = gen.gen_EE(radius, N_input_neurons, N_excit_neurons)
gen.draw_heatmap(EE_weights, N_input_neurons)

# Placeholder: Generate a simple weight matrix for demonstration
N_input_neurons = 49
N_excit_neurons = 49
gw = gen_weights()
EE_weights = gw.gen_EE(radius=1, N_input_neurons=49, N_excit_neurons=49)

# Create a graph
G = nx.DiGraph()

# Add nodes with different attributes for input and excitatory neurons for visual distinction
input_neurons = range(N_input_neurons)
excit_neurons = range(N_input_neurons, N_input_neurons + N_excit_neurons)
G.add_nodes_from(input_neurons, bipartite=0, layer="input")
G.add_nodes_from(excit_neurons, bipartite=1, layer="excitatory")

# Add edges based on the weight matrix
for i in range(N_input_neurons):
    for j in range(N_excit_neurons):
        weight = EE_weights[i, j]
        if weight > 0:  # Assuming you only want to add edges for non-zero weights
            G.add_edge(i, N_input_neurons + j, weight=weight)

# Drawing the network
pos = nx.spring_layout(G)  # Positions for all nodes

# Separately draw different layers
nx.draw_networkx_nodes(
    G, pos, nodelist=input_neurons, node_color="r", label="Input Neurons"
)
nx.draw_networkx_nodes(
    G, pos, nodelist=excit_neurons, node_color="b", label="Excitatory Neurons"
)
nx.draw_networkx_edges(G, pos)

# Labels for nodes
nx.draw_networkx_labels(G, pos)
plt.legend(scatterpoints=1)
plt.show()
