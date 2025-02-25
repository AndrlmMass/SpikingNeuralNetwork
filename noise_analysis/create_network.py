import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np


# Create weight array
def create_weights(
    N_exc,
    N_inh,
    N_x,
    weight_affinity_hidden_exc,
    weight_affinity_hidden_inh,
    weight_affinity_input,
    pos_weight,
    neg_weight,
    plot_weights,
    plot_network,
):
    N = N_exc + N_inh + N_x

    # Create weights based on affinity rate
    mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
    mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
    mask_input = np.random.random((N, N)) < weight_affinity_input
    weights = np.zeros(shape=(N, N))

    # input_weights
    weights[:N_x, N_x:-N_inh][mask_input[:N_x, N_x:-N_inh]] = pos_weight
    # excitatory weights
    weights[N_x:-N_inh, N_x:][mask_hidden_exc[N_x:-N_inh, N_x:]] = pos_weight
    # inhibitory weights
    weights[-N_inh:, N_x:-N_inh][mask_hidden_inh[-N_inh:, N_x:-N_inh]] = neg_weight
    # remove self-connections (diagonal) to 0 for excitatory weights
    np.fill_diagonal(weights[N_x:-N_inh, N_x:-N_inh], 0)
    # remove recurrent connections from exc to inh
    inh_mask = weights[N_x:-N_inh, -N_inh:].T == 1
    weights[-N_inh:, N_x:-N_inh][inh_mask] = 0

    if plot_weights:
        plt.imshow(weights)
        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.show()

    if plot_network:

        total_nodes = N_x + N_exc + N_inh

        # --- Create a sample weighted adjacency matrix ---
        # For demonstration, we generate a random matrix.
        np.random.seed(42)
        A = weights

        # Optionally, if your matrix is supposed to be symmetric, you could do:
        # A = (A + A.T) / 2

        # --- Create the NetworkX graph from the numpy array ---
        G = nx.from_numpy_array(A)

        # --- Partition the nodes ---
        input_nodes = list(range(N_x))
        exc_nodes = list(range(N_x, N_x + N_exc))
        inh_nodes = list(range(N_x + N_exc, total_nodes))

        # --- Assign custom positions ---
        # We'll place each cluster in its own vertical column.
        pos = {}

        # For input nodes (left column, x=0)
        for i, node in enumerate(input_nodes):
            # Distribute vertically between 0 and 1
            y = 1 - (i / (len(input_nodes) - 1)) if len(input_nodes) > 1 else 0.5
            pos[node] = (0, y)

        # For excitatory nodes (middle column, x=1)
        for i, node in enumerate(exc_nodes):
            y = 1 - (i / (len(exc_nodes) - 1)) if len(exc_nodes) > 1 else 0.5
            pos[node] = (1, y)

        # For inhibitory nodes (right column, x=2)
        for i, node in enumerate(inh_nodes):
            y = 1 - (i / (len(inh_nodes) - 1)) if len(inh_nodes) > 1 else 0.5
            pos[node] = (2, y)

        # --- Define colors for each cluster ---
        node_colors = {}
        for node in input_nodes:
            node_colors[node] = "skyblue"  # Input nodes color
        for node in exc_nodes:
            node_colors[node] = "lightgreen"  # Excitatory nodes color
        for node in inh_nodes:
            node_colors[node] = "salmon"  # Inhibitory nodes color

        colors = [node_colors[node] for node in G.nodes()]

        # --- Draw the graph ---
        plt.figure(figsize=(8, 4))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=100)

        # Draw edges with widths proportional to the weight (scaling factor for visibility)
        edges = G.edges(data=True)
        edge_weights = [data["weight"] for (u, v, data) in edges]
        nx.draw_networkx_edges(G, pos, width=[5 * w for w in edge_weights], alpha=0.1)

        # Draw labels for clarity
        nx.draw_networkx_labels(G, pos, font_size=5, font_color="black")

        plt.title("Partitioned Graph: Input, Excitatory, Inhibitory")
        plt.axis("off")
        plt.show()

    return weights


def create_arrays(
    N,
    resting_membrane,
    total_time_train,
    total_time_test,
    max_time,
    data_train,
    data_test,
    N_x,
):
    membrane_potential_train = np.zeros((total_time_train, N - N_x))
    membrane_potential_train[0] = resting_membrane

    membrane_potential_test = np.zeros((total_time_test, N - N_x))
    membrane_potential_test[0] = resting_membrane

    pre_trace = np.zeros((N))
    post_trace = np.zeros((N - N_x))

    spikes_train = np.zeros((total_time_train, N), dtype="int64")
    spikes_train[:, :N_x] = data_train

    spikes_test = np.zeros((total_time_test, N), dtype="int64")
    spikes_test[:, :N_x] = data_test

    spike_times = np.random.randint(low=max_time, high=max_time**2, size=N)

    return (
        membrane_potential_train,
        membrane_potential_test,
        pre_trace,
        post_trace,
        spikes_train,
        spikes_test,
        spike_times,
    )
