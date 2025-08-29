import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np


# Create weight array
def create_weights(
    N_exc,
    N_inh,
    N_x,
    N,
    w_dense_ee,
    w_dense_ei,
    w_dense_ie,
    w_dense_se,
    se_weights,
    ee_weights,
    ei_weights,
    ie_weights,
    plot_weights,
    plot_network,
):
    weights = np.zeros(shape=(N, N))

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory

    # Create weights based on affinity rates
    mask_ee = np.random.random((N_exc, N_exc)) < w_dense_ee
    mask_ei = np.random.random((N_exc, N_inh)) < w_dense_ei
    mask_ie = np.random.random((N_inh, N_exc)) < w_dense_ie
    mask_se = np.random.random((N_x, N_exc)) < w_dense_se

    # input poisson weights
    weights[:st, st:ex][mask_se] = se_weights

    # hidden excitatory weights
    weights[st:ex, st:ex][mask_ee] = ee_weights
    weights[st:ex, ex:ih][mask_ei] = ei_weights

    # hidden inhibitory weights
    weights[ex:ih, st:ex][mask_ie] = ie_weights

    # remove excitatory self-connecting (diagonal) weights
    np.fill_diagonal(weights[st:ex, st:ex], 0)

    # remove recurrent connections from exc to inh
    inh_mask = weights[st:ex, ex:ih].T != 0
    weights[ex:ih, st:ex][inh_mask] = 0

    if plot_weights:
        boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]

        # Create a ListedColormap with the exact colors you want:
        cmap = ListedColormap(["red", "white", "green"])

        # Use BoundaryNorm to map data values to the colormap bins
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)

        plt.imshow(weights, cmap=cmap, norm=norm)
        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.show()

    if plot_network:
        total_nodes = N_x + N_exc + N_inh

        # --- Create a sample weighted adjacency matrix ---
        # For demonstration, we generate a random matrix.
        np.random.seed(42)
        A = weights

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
    data_train,
    data_test,
    N_x,
    N_exc,
    N_inh,
):
    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory

    membrane_potential_train = np.zeros((total_time_train, ih - st))
    if total_time_train > 0:
        membrane_potential_train[0] = resting_membrane

    membrane_potential_test = np.zeros((total_time_test, ih - st))
    if total_time_test > 0:
        membrane_potential_test[0] = resting_membrane

    spikes_train = np.zeros((total_time_train, N), dtype=np.int8)
    if data_train is not None and total_time_train > 0:
        spikes_train[:, :st] = data_train

    spikes_test = np.zeros((total_time_test, N), dtype=np.int8)
    if data_test is not None and total_time_test > 0:
        spikes_test[:, :st] = data_test

    return (
        membrane_potential_train,
        membrane_potential_test,
        spikes_train,
        spikes_test,
    )
