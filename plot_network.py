from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Draw the network and plot the distribution
def draw_network(combined_array):
    n_rows, n_cols = combined_array.shape[0], combined_array.shape[1]
    print(combined_array)

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_rows):
        G.add_node(i)

    # Add edges with weights
    for i in range(n_rows):
        for j in range(n_cols):
            if combined_array[i, j,0] != 0:
                G.add_edge(j, i, weight=combined_array[i, j,0])

    # Draw the network
    pos = nx.spring_layout(G)  # positions for all nodes

    # Define edges based on weight
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]

    print(len(positive_edges),len(negative_edges))

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
    array = array[:,:,0]
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

