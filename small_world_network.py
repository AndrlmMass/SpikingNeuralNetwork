import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import curve_fit

def generate_small_world_network_power_law(num_neurons, excit_inhib_ratio, FF_FB_ratio, alpha, perc_input_neurons):
    n_rows, n_cols = num_neurons, num_neurons

    # Generate power-law distribution for the probability of connections
    connection_probabilities = np.random.power(alpha, size=n_rows*n_cols)

    # Initialize the arrays for weights and signs
    weight_array = np.zeros((n_rows, n_cols))
    sign_array = np.zeros((n_rows, n_cols), dtype=int)

    # Assign weights and signs based on connection probability
    for i in range(n_rows):
        for j in range(n_cols):
            if i != j and connection_probabilities[i*n_cols + j] > np.random.rand():  # Determine if there is a connection
                # Make weights either inhibitory (const = -1) or excitatory (const = 1) 
                const = 1 if np.random.rand() < excit_inhib_ratio else -1
                # Assign a fixed weight or a randomly chosen weight
                weight_array[i, j] = np.random.rand()*const  # You can also multiply by a random or fixed weight factor here
                # Sign array containing directionality of connections (1 = forward, -1 = backward, & 0 = no connection)
                sign_array[i, j] = 1 if np.random.rand() < FF_FB_ratio else -1

    # Calculate ratio of input neurons to hidden neurons
    perc_inp = np.mean(np.all(sign_array[:,:] >= 0, axis=1))

    while perc_inp < perc_input_neurons:
        # Choose a random neuron to become an input neuron
        idx = np.random.randint(num_neurons)
        sign_array[idx, :] = np.abs(sign_array[idx, :])

        # Recalculate the percentage of input neurons
        perc_inp = np.mean(np.all(sign_array >= 0, axis=1))

        if perc_inp >= perc_input_neurons:
            break

    # Similarly, we can correct for too many input neurons. We assume 
    # this to be more than 125% of the original desired ratio
    while perc_inp > perc_input_neurons*1.25: #This is an arbitrary border
        # Choose a random neuron to become an input neuron
        idx = np.random.randint(num_neurons)
        sign_array[idx, :, 1] = np.abs(sign_array[idx, :])

        # Recalculate the percentage of input neurons
        perc_inp = np.mean(np.all(sign_array >= 0, axis=1))

        if perc_inp < perc_input_neurons*1.25:
            break

    # Concatenate the weight and sign arrays
    combined_array = np.stack((weight_array, sign_array), axis=-1)

    return combined_array

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
    edges = np.abs(array[:,:,1])  # Using absolute value to consider the strength of an edge regardless of sign
    edges_count = np.sum(edges, axis=0)
    print(edges_count)
    sorted_edges = np.sort(edges_count)[::-1] # Sort in descending order
    print(sorted_edges)

    # Generate a rank for each edge (their index)
    x_data = np.arange(1, len(sorted_edges) + 1)
    print(x_data)
    
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


