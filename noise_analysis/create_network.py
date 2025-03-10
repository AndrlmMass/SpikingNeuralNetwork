import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np


# Create weight array
def create_weights(
    N_exc,
    N_inh,
    N_classes,
    supervised,
    unsupervised,
    N_x,
    N,
    weight_affinity_hidden_exc,
    weight_affinity_hidden_inh,
    weight_affinity_output,
    weight_affinity_input,
    pos_weight,
    neg_weight,
    plot_weights,
    plot_network,
    pp_weight,
    pn_weight,
    tp_weight,
    tn_weight,
    fp_weight,
    fn_weight,
    tl_weight,
    fl_weight,
):
    weights = np.zeros(shape=(N, N))

    if supervised:
        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih + N_classes  # predicted positive
        pn = pp + N_classes  # predicted negative
        tp = pn + N_classes  # true positive
        tn = tp + N_classes  # true negative
        fp = tn + N_classes  # false positive
        fn = fp + N_classes  # false negative
        tl = fn + N_classes  # true label
        fl = tl + N_classes  # false label

        # Create weights based on affinity rates
        mask_label = np.zeros((N_classes, N_classes))
        np.fill_diagonal(mask_label, val=1)
        mask_label = mask_label.astype(bool)
        mask_output = np.random.random((N_exc, N_classes)) < weight_affinity_output
        mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
        mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
        mask_input = np.random.random((N, N)) < weight_affinity_input

        # input_weights
        weights[:st, st:ex][mask_input[:st, st:ex]] = pos_weight

        # hidden excitatory weights
        weights[st:ex, st:ih][mask_hidden_exc[st:ex, st:ih]] = pos_weight

        # hidden inhibitory weights
        weights[ex:ih, st:ex][mask_hidden_inh[ex:ih, st:ex]] = neg_weight

        # remove excitatory self-connecting (diagonal) weights
        np.fill_diagonal(weights[st:ex, st:ex], 0)

        # remove recurrent connections from exc to inh
        inh_mask = weights[st:ex, ex:ih].T != 0
        weights[ex:ih, st:ex][inh_mask] = 0

        # update weights from excitatory-to-predictors (pos and negative)
        weights[st:ex, ih:pp][mask_output] = pos_weight
        weights[st:ex, pp:pn][mask_output] = pos_weight

        # update weights from predictors-to-excitatory
        weights[ih:pp, st:ex][mask_output.T] = pp_weight
        weights[pp:pn, st:ex][mask_output.T] = pn_weight  # negative weight

        # update weights from true-to-predictors
        weights[pn:tp, ih:pp][mask_label] = tp_weight
        weights[tp:tn, pp:pn][mask_label] = tn_weight

        # update weights from false-to-predictors
        weights[tn:fp, pp:pn][mask_label] = fp_weight  # negative weight
        weights[fp:fn, ih:pp][mask_label] = fn_weight  # negative weight

        # update weights from label-to-true
        weights[fn:tl, pn:tp][mask_label] = tl_weight
        weights[fn:tl, tn:fp][mask_label] = tl_weight

        # update weights from label-to-false
        weights[tl:fl, tp:tn][mask_label] = fl_weight
        weights[tl:fl, fp:fn][mask_label] = fl_weight

    elif unsupervised:
        # st = N_x
        # ex = st + N_exc
        # pp = ex + N_classes
        # pn = pp
        # ih = pn + N_inh
        # tp = ih
        # tn = ih
        # tr = tn + N_classes

        # # Create weights based on affinity rate
        # mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
        # mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
        # mask_input = np.random.random((N, N)) < weight_affinity_input

        # # input_weights
        # weights[:o0, o0:o1][mask_input[:o0, o0:o1]] = pos_weight
        # # hidden excitatory weights
        # weights[o0:o1, o0:o3][mask_hidden_exc[o0:o1, o0:o3]] = pos_weight
        # # hidden inhibitory weights
        # weights[o2:o3, o0:o1][mask_hidden_inh[o2:o3, o0:o1]] = neg_weight
        # # remove self-connections (diagonal) to 0 for excitatory weights
        # np.fill_diagonal(weights[o0:o1, o0:o1], 0)
        # # remove recurrent connections from exc to inh
        # inh_mask = weights[o0:o1, o2:o3].T == 1
        # weights[o2:o3, o0:o1][inh_mask] = 0
        ...
    else:
        # st = N_x  # stimulation
        # ex = st + N_exc  # excitatory
        # ih = ex + N_inh  # inhibitory
        # pp = ih  # predicted positive
        # pn = ih  # predicted negative
        # tp = ih  # true positive
        # tn = ih  # true negative
        # fp = ih  # false positive
        # fn = ih  # false negative
        # tl = ih  # true label
        # fl = ih  # false label

        # # Create weights based on affinity rate
        # mask_hidden_exc = np.random.random((N, N)) < weight_affinity_hidden_exc
        # mask_hidden_inh = np.random.random((N, N)) < weight_affinity_hidden_inh
        # mask_input = np.random.random((N, N)) < weight_affinity_input

        # # input_weights
        # weights[:o0, o0:o1][mask_input[:o0, o0:o1]] = pos_weight
        # # hidden excitatory weights
        # weights[o0:o1, o0:o3][mask_hidden_exc[o0:o1, o0:o3]] = pos_weight
        # # hidden inhibitory weights
        # weights[o2:o3, o0:o1][mask_hidden_inh[o2:o3, o0:o1]] = neg_weight
        # # remove self-connections (diagonal) to 0 for excitatory weights
        # np.fill_diagonal(weights[o0:o1, o0:o1], 0)
        # # remove recurrent connections from exc to inh
        # inh_mask = weights[o0:o1, o2:o3].T == 1
        # weights[o2:o3, o0:o1][inh_mask] = 0
        ...

    if plot_weights:
        boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]

        # Create a ListedColormap with the exact colors you want:
        cmap = ListedColormap(["red", "white", "green"])

        # Use BoundaryNorm to map data values to the colormap bins
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)

        plt.imshow(weights, cmap=cmap, norm=norm)

        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih + N_classes  # predicted positive
        pn = pp + N_classes  # predicted negative
        tp = pn + N_classes  # true positive
        tn = tp + N_classes  # true negative
        fp = tn + N_classes  # false positive
        fn = fp + N_classes  # false negative
        tl = fn + N_classes  # true label
        fl = tl + N_classes  # false label

        # Plot the data
        # boundaries = [st, ex, ih, ]
        # class_names = ["N_exc", "N_inh", "N_out"]

        # bbox_props = dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.2")

        # # Draw boundary lines using axhline and axvline. We subtract 0.5 to align with pixel edges.
        # for n in range(len(boundaries)):
        #     val = boundaries[n]
        #     key = class_names[n]
        #     start_val = 20.5
        #     if key == "N_inh":
        #         col = "blue"
        #     else:
        #         col = "green"
        #     plt.axhline(val, color=col, linestyle="--", linewidth=2)
        #     plt.axvline(val, color=col, linestyle="--", linewidth=2)
        #     plt.text(
        #         val,
        #         start_val,
        #         key,
        #         ha="center",
        #         va="bottom",
        #         color=col,
        #         size=13,
        #         bbox=bbox_props,
        #     )
        #     plt.text(
        #         start_val,
        #         val,
        #         key,
        #         ha="center",
        #         va="bottom",
        #         color=col,
        #         size=13,
        #         bbox=bbox_props,
        #     )

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
    labels_true,
    N_x,
    N_exc,
    N_inh,
):
    if supervised:
        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih + N_classes  # predicted positive
        pn = pp + N_classes  # predicted negative
        tp = pn + N_classes  # true positive
        tn = tp + N_classes  # true negative
        fp = tn + N_classes  # false positive
        fn = fp + N_classes  # false negative
        tl = fn + N_classes  # true label
        fl = tl + N_classes  # false label
    else:
        st = N_x  # stimulation
        ex = st + N_exc  # excitatory
        ih = ex + N_inh  # inhibitory
        pp = ih  # predicted positive
        pn = ih  # predicted negative
        tp = ih  # true positive
        tn = ih  # true negative
        fp = ih  # false positive
        fn = ih  # false negative
        tl = ih  # true label
        fl = ih  # false label

    membrane_potential_train = np.zeros((total_time_train, fn - st))
    membrane_potential_train[0] = resting_membrane

    membrane_potential_test = np.zeros((total_time_test, fn - st))
    membrane_potential_test[0] = resting_membrane

    pre_trace = np.zeros((pn))
    post_trace = np.zeros((pn - st))

    spikes_train = np.zeros((total_time_train, N), dtype="int64")
    spikes_train[:, :st] = data_train
    spikes_train[:, fn:] = labels_true

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
