import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np


# Create weight array
def create_weights(
    N_exc,
    N_inh,
    N_classes,
    supervised,
    N_x,
    true2pred_weight,
    weight_affinity_hidden_exc,
    weight_affinity_hidden_inh,
    weight_affinity_output_exc,
    weight_affinity_input,
    pos_weight,
    neg_weight,
    plot_weights,
    plot_network,
):
    if supervised:
        N = N_exc + N_inh + N_x + N_classes * 2

    else:
        N = N_exc + N_inh + N_x

    # Create weights based on affinity rate
    mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
    mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
    mask_output_exc = np.random.random((N, N)) < weight_affinity_output_exc
    mask_input = np.random.random((N, N)) < weight_affinity_input
    weights = np.zeros(shape=(N, N))

    # input_weights
    weights[:N_x, N_x : N_x + N_exc][mask_input[:N_x, N_x : N_x + N_exc]] = pos_weight
    # hidden excitatory weights
    weights[N_x : N_x + N_exc, N_x : N_x + N_exc + N_inh][
        mask_hidden_exc[N_x : N_x + N_exc, N_x : N_x + N_exc + N_inh]
    ] = pos_weight
    # hidden inhibitory weights
    weights[N_x + N_exc : N_x + N_exc + N_inh, N_x : N_x + N_exc][
        mask_hidden_inh[N_x + N_exc : N_x + N_exc + N_inh, N_x : N_x + N_exc]
    ] = neg_weight
    # remove self-connections (diagonal) to 0 for excitatory weights
    np.fill_diagonal(weights[N_x : N_x + N_exc, N_x : N_x + N_exc], 0)
    # remove recurrent connections from exc to inh
    inh_mask = weights[N_x : N_x + N_exc, N_x + N_exc : N_x + N_exc + N_inh].T == 1
    weights[N_x + N_exc : N_x + N_exc + N_inh, N_x : N_x + N_exc][inh_mask] = 0

    # readout weights
    if supervised:
        weights[N_x : N_x + N_exc, -N_classes * 2 : -N_classes][
            mask_output_exc[N_x : N_x + N_exc, -N_classes * 2 : -N_classes]
        ] = pos_weight
        true_weights = np.zeros((N_classes, N_classes))
        np.fill_diagonal(true_weights, true2pred_weight)
        weights[-N_classes:, -N_classes * 2 : -N_classes] = true_weights

    if plot_weights:
        boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]

        # Create a ListedColormap with the exact colors you want:
        cmap = ListedColormap(["red", "white", "green"])

        # Use BoundaryNorm to map data values to the colormap bins
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)

        plt.imshow(weights, cmap=cmap, norm=norm)

        # Plot the data
        boundaries = [N_x, N_x + N_exc, N - 2 * N_classes]
        class_names = ["N_exc", "N_inh", "N_out"]

        bbox_props = dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2")

        # Draw boundary lines using axhline and axvline. We subtract 0.5 to align with pixel edges.
        for n in range(len(boundaries)):
            val = boundaries[n]
            key = class_names[n]
            start_val = 20.5
            if key == "N_inh":
                col = "blue"
            else:
                col = "green"
            plt.axhline(val, color=col, linestyle="--", linewidth=2)
            plt.axvline(val, color=col, linestyle="--", linewidth=2)
            plt.text(
                val,
                start_val,
                key,
                ha="center",
                va="bottom",
                color=col,
                size=13,
                bbox=bbox_props,
            )
            plt.text(
                start_val,
                val,
                key,
                ha="center",
                va="bottom",
                color=col,
                size=13,
                bbox=bbox_props,
            )

        plt.gca().invert_yaxis()
        plt.title("Weights")
        plt.show()

    if plot_network:
        if supervised:
            total_nodes = N_x + N_exc + N_inh + N_classes * 2
        else:
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
    supervised,
    max_time,
    N_classes,
    data_train,
    data_test,
    N_x,
):
    if supervised:
        add = N_classes
    else:
        add = 0

    membrane_potential_train = np.zeros((total_time_train, N - N_x - add))
    membrane_potential_train[0] = resting_membrane

    membrane_potential_test = np.zeros((total_time_test, N - N_x - add))
    membrane_potential_test[0] = resting_membrane

    pre_trace = np.zeros((N - add))
    post_trace = np.zeros((N - N_x - add))

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
