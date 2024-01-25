import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit

def generate_small_world_network_power_law(num_neurons, excit_inhib_ratio, FF_FB_ratio, alpha, perc_input_neurons):
    n_rows, n_cols = num_neurons, num_neurons

    # Generate power-law distribution for the probability of connections
    connection_probabilities = np.random.power(alpha, size=n_rows * n_cols)

    # Initialize the arrays for weights and signs
    weight_array = np.ones((n_rows, n_cols))
    np.fill_diagonal(weight_array, 0)

    # Assign weights and signs based on connection probability
    for i in range(n_rows):
        for j in range(n_cols):  
            if weight_array[i, j] != 0:
                if connection_probabilities[i * n_cols + j] > np.random.rand():
                    # Assign weights
                    const = 1 if np.random.rand() < excit_inhib_ratio else -1
                    weight_array[i, j] = np.random.rand() * const
                    weight_array[j, i] = 0

                else:
                    weight_array[i, j] = 0

    # Calculate ratio of input neurons to hidden neurons
    perc_inp = np.mean(np.all(weight_array == 0, axis=1))
    print(perc_inp)

    iteration = 0
    if perc_inp < perc_input_neurons:
        while perc_inp < perc_input_neurons and iteration < num_neurons:
            # Choose a random neuron that is not an input neuron to become an input neuron
            non_input_neurons = np.where(np.any(weight_array > 0, axis=1))[0]
            idx = np.random.choice(non_input_neurons)
            weight_array[idx, :] = np.zeros(num_neurons)

            # Check if the new input neuron has any PSPs
            if np.sum(weight_array[:,idx]) == 0:
                rand_synapse = np.random.choice(non_input_neurons)
                # Change rand_synapse if it's the same as the pre-synaptic potential idx
                while rand_synapse == idx:
                    if rand_synapse != idx and np.sum(weight_array[rand_synapse,:]) != 0:
                        break
                    else:
                        rand_synapse = np.random.choice(non_input_neurons)
                weight_array[rand_synapse,idx] = np.random.rand()

            # Recalculate the percentage of input neurons
            perc_inp = np.mean(np.all(weight_array == 0, axis=1))
            iteration += 1

    print(perc_inp)

    return weight_array


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
            if i != j and combined_array[i, j] != 0:
                G.add_edge(j, i, weight=combined_array[i, j])

    # Draw the network
    pos = nx.spring_layout(G)  # positions for all nodes

    # Define edges based on weight
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, width=1, edge_color='g', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, width=1, edge_color='r', style='dotted')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

    pos_2_neg = f"Percentage of positive edges: {round(len(positive_edges)/(len(negative_edges)+len(positive_edges)),2)}"
    custom_line = mlines.Line2D([], [], color='black', marker='*', linestyle='None', label=pos_2_neg)

    plt.legend(handles=[custom_line])
    plt.axis('off')
    plt.show()

def power_law(x, a, b):
    return a * np.power(x, b)

def draw_edge_distribution(array):
    # Extract the presence of edges and count them
    edges = np.count_nonzero(array, axis=0)
    sorted_edges = np.sort(edges)[::-1] # Sort in descending order

    # Generate a rank for each edge (their index)
    x_data = np.arange(1, len(sorted_edges) + 1)
    
    # Plotting the edge distribution
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, sorted_edges, label='Edge Weight Distribution', marker='o', linestyle='-', markersize=4)

    # Fit the distribution to a power law
    # We need to fit it to the number of edges
    # Ensure that y_data for fitting does not contain zero values
    params, _ = curve_fit(power_law, x_data, sorted_edges, maxfev=5000)

    # We plot the fitted line using the parameters obtained from the curve fitting
    fitted_line = power_law(x_data, *params)
    plt.plot(x_data, fitted_line, label='Fitted Power Law', linestyle='--', color='red')

    # Adding labels and title
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of edges')
    plt.xticks(np.arange(min(x_data),max(x_data)+1,2.0))
    plt.title('Edge/Node Distribution and Power Law Fit')
    plt.legend()

    plt.show()

