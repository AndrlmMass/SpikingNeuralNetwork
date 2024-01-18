import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_small_world_network_power_law(num_neurons, inhib_excit_ratio, alpha):
    # Array dimensions
    n_rows, n_cols = num_neurons, num_neurons

    # Generate power-law distribution for weights
    weights = np.random.power(alpha, size=n_rows*n_cols)

    # Initialize the arrays for weights and signs
    weight_array = np.zeros((n_rows, n_cols))
    sign_array = np.zeros((n_rows, n_cols), dtype=int)

    # Assign weights and signs
    for i in range(n_rows):
        for j in range(n_cols):
            if i != j:  # No self-connections
                weight_array[i, j] = weights[i*n_cols + j]
                sign_array[i, j] = 1 if np.random.rand() < inhib_excit_ratio else -1

    # Adjust weights for sign
    weight_array[sign_array == -1] *= -1

    # Concatenate the weight and sign arrays
    combined_array = np.stack((weight_array, sign_array), axis=-1)

    return combined_array

# Define the parameters for power-law distribution
num_neurons = 20
inhib_excit_ratio = 0.8
alpha = 1  # Exponent for power-law distribution

# Generate a small world network with power-law distribution
combined_array_power_law = generate_small_world_network_power_law(num_neurons, inhib_excit_ratio, alpha)

# Draw the network and plot the distribution


def draw_network(combined_array):
    n_rows, n_cols = combined_array.shape[0], combined_array.shape[1]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_rows):
        G.add_node(i)

    # Add edges with weights
    for i in range(n_rows):
        for j in range(n_cols):
            if i != j and combined_array[i, j, 0] != 0:
                G.add_edge(i, j, weight=combined_array[i, j, 0])

    # Draw the network
    pos = nx.spring_layout(G)  # positions for all nodes

    # Define edges based on weight
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, width=1, edge_color='g', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, width=1, edge_color='b', style='solid')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

    plt.axis('off')
    plt.show()

draw_network(combined_array_power_law)

