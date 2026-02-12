import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np


def enforce_per_exc_sparsity(W_se, frac=0.10):
    """
    W_se: (N_x, N_exc)
    frac: fraction of nonzero weights per excitatory neuron
    """
    W = W_se.copy()
    N_x = W.shape[0]
    k = int(np.ceil(frac * N_x))

    for j in range(W.shape[1]):
        col = W[:, j]
        if k < len(col):
            thresh = np.partition(col, -k)[-k]
            col[col < thresh] = 0.0
        W[:, j] = col

    return W


def exc_coords(H_e, W_e):
    return np.array([(r, c) for r in range(H_e) for c in range(W_e)], dtype=float)


def topk_mask_per_col(W, frac):
    """Keep top frac per column (incoming sparsity per postsyn)."""
    W = W.copy()
    n_rows, n_cols = W.shape
    k = int(np.ceil(frac * n_rows))
    for j in range(n_cols):
        col = W[:, j]
        if k < n_rows:
            thr = np.partition(col, -k)[-k]
            col[col < thr] = 0.0
        W[:, j] = col
    return W


def gaussian_ei_local(N_exc, N_inh, H_e, W_e, sigma=1.5, peak=1.0, frac=None):
    coords = exc_coords(H_e, W_e)  # (N_exc,2)

    # Choose inhibitory "home" positions on the exc grid
    # Option A: random homes among exc neurons
    home_idx = np.random.randint(0, N_exc, size=N_inh)
    centers = coords[home_idx]  # (N_inh,2)

    # Distances from each exc neuron to each inhibitory center -> (N_exc, N_inh)
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    W_ei = np.exp(-d2 / (2 * sigma**2))

    # Normalize each inhibitory column and scale
    W_ei /= W_ei.max(axis=0, keepdims=True) + 1e-12
    W_ei *= peak

    if frac is not None:
        W_ei = topk_mask_per_col(W_ei, frac=frac)

    return W_ei, home_idx


def mexican_hat_ie_far(
N_inh, H_e, W_e, home_idx, peak=1.0, frac=None, r0=2.0, sigma_r=2.0
):
    coords = exc_coords(H_e, W_e)  # (N_exc,2)
    centers = coords[home_idx]  # (N_inh,2)

    # Distances from inhibitory center to each excit neuron -> (N_inh, N_exc)
    d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    d = np.sqrt(d2)
    ring = np.exp(-((d - r0)**2) / (2*sigma_r**2))
    ring /= ring.max(axis=1, keepdims=True) + 1e-12
    W_ie = peak * ring

    # Optional: sparsify per inhibitory neuron (per row)
    if frac is not None:
        W_ie = enforce_topk_per_row_abs(W_ie, frac=frac)

    # Guarantee "don't inhibit your own local pool" strongly:
    # (hard zero at the exact center exc neuron)
    W_ie[np.arange(N_inh), home_idx] = 0.0

    return W_ie


def gaussian_se_weights(
    N_x, N_exc, H, W, sigma=2.0, peak=1.0, truncate=0.01, fraction=0.1
):
    """
    Returns W_se with shape (N_x, N_exc): input->exc weights.
    Each excitatory neuron gets a Gaussian receptive field over the (H,W) input grid.
    """
    assert H * W == N_x

    # Input coordinates (N_x, 2)
    inp_coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)

    # Place excitatory centers on a grid that spans the input
    # Choose a grid for excitatory neurons: roughly sqrt(N_exc) by sqrt(N_exc)
    gh = int(np.floor(np.sqrt(N_exc)))
    gw = int(np.ceil(N_exc / gh))

    # Evenly spaced center coordinates in continuous input space
    centers_r = np.linspace(0, H - 1, gh)
    centers_c = np.linspace(0, W - 1, gw)
    centers = np.array([(r, c) for r in centers_r for c in centers_c], dtype=float)[
        :N_exc
    ]

    # Compute Gaussian weights: for each exc center, distance^2 to all inputs
    # Result: (N_exc, N_x)
    d2 = ((centers[:, None, :] - inp_coords[None, :, :]) ** 2).sum(axis=2)
    G = np.exp(-d2 / (2 * sigma**2))

    # Normalize each receptive field and scale by peak
    G /= G.max(axis=1, keepdims=True) + 1e-12
    G *= peak

    # Truncate tiny weights to exactly 0 for sparsity
    if truncate is not None:
        G[G < truncate] = 0.0

    # enforce sparsity
    G = enforce_per_exc_sparsity(G.T, frac=fraction)

    # Return in (N_x, N_exc) orientation for weights[:st, st:ex]
    return G


def grid_shape(n):
    H = int(np.floor(np.sqrt(n)))
    while n % H != 0:
        H -= 1
    W = n // H
    return H, W


def enforce_topk_per_row_abs(W, frac):
    W = W.copy()
    n = W.shape[1]
    k = int(np.ceil(frac * n))
    for i in range(W.shape[0]):
        row = W[i]
        if k < n:
            thr = np.partition(np.abs(row), -k)[-k]
            row[np.abs(row) < thr] = 0.0
        W[i] = row
    return W


def enforce_topk_per_row(W, frac):
    """Keep top frac of entries in each row (excluding ties may keep slightly >k)."""
    W = W.copy()
    n = W.shape[1]
    k = int(np.ceil(frac * n))
    for i in range(W.shape[0]):
        row = W[i]
        if k < n:
            thresh = np.partition(row, -k)[-k]
            row[row < thresh] = 0.0
        W[i] = row
    return W


def gaussian_ee_weights(
    N_exc, H_e, W_e, sigma=2.0, peak=1.0, frac=None, self_zero=True
):
    """
    Returns W_ee (N_exc, N_exc): excitatory->excitatory.
    Exc neurons live on a (H_e, W_e) grid.
    """
    assert H_e * W_e == N_exc

    coords = np.array(
        [(r, c) for r in range(H_e) for c in range(W_e)], dtype=float
    )  # (N_exc,2)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)  # (N_exc,N_exc)

    W = np.exp(-d2 / (2 * sigma**2))

    # normalize rows so max=1 then scale
    W /= W.max(axis=1, keepdims=True) + 1e-12
    W *= peak

    if self_zero:
        np.fill_diagonal(W, 0.0)

    if frac is not None:
        W = enforce_topk_per_row(W, frac=frac)
        if self_zero:
            np.fill_diagonal(W, 0.0)

    return W


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
    plot_weights = False

    # input poisson weights
    H = int(np.sqrt(N_x))
    W = H
    weights_st_ex = gaussian_se_weights(
        N_x,
        N_exc,
        H,
        W,
        sigma=1.0,
        peak=se_weights,
        truncate=0.01,
        fraction=w_dense_se,
    )
    weights[:st, st:ex] = weights_st_ex

    if plot_weights:
        for j in range(N_exc//2,N_exc): 
            rf = weights_st_ex[:, j].reshape(H, W)
            plt.imshow(rf); plt.title(f"S -> E {j}"); plt.colorbar(); plt.show()
            if input("Press Enter to continue...") == "q":
                break


    H_e, W_e = 25, 40

    weights_ex_ex = gaussian_ee_weights(
        N_exc,
        H_e,
        W_e,
        sigma=1.5,
        peak=ee_weights,
        frac=w_dense_ee,
    )
    weights[st:ex, st:ex] = weights_ex_ex

    if plot_weights:
        for j in range(N_exc//2,N_exc): 
            rf = weights_ex_ex[j, :].reshape(H_e, W_e)
            plt.imshow(rf); plt.title(f"E -> E {j}"); plt.colorbar(); plt.show()
            if input("Press Enter to continue...") == "q":
                break

    # --- E->I local pooling ---
    W_ei, home_idx = gaussian_ei_local(
        N_exc=N_exc,
        N_inh=N_inh,
        H_e=H_e,
        W_e=W_e,
        sigma=2,  # local pooling radius
        peak=ei_weights,  # excitatory onto inhibitory (positive)
        frac=w_dense_ei,  # density meaning: fraction of exc inputs to each I
    )
    weights[st:ex, ex:ih] = W_ei

    if plot_weights:
        for j in range(N_inh//4,N_inh):
            ei_field = W_ei[:, j].reshape(H_e, W_e)
            plt.imshow(ei_field); plt.title(f"E -> I {j}"); plt.colorbar(); plt.show()
            if input("Press Enter to continue...") == "q":
                break


    # --- I->E far inhibition (center-surround / mexican hat) ---
    W_ie = mexican_hat_ie_far(
        N_inh=N_inh,
        H_e=H_e,
        W_e=W_e,
        home_idx=home_idx,  
        peak=ie_weights,  # NOTE: if inhibitory weights are negative in your sim, set peak=-abs(ie_weights)
        frac=w_dense_ie,
        r0=4.0,
        sigma_r=1.0,
    )

    # If inhibitory synapses should be negative in your weight matrix:
    weights[ex:ih, st:ex] = W_ie

    if plot_weights:
        for j in range(N_inh//4,N_inh):
            ie_field = -W_ie[j, :].reshape(H_e, W_e)
            plt.imshow(ie_field); plt.title(f"I->E {j}"); plt.colorbar(); plt.show()
            if input("Press Enter to continue...") == "q":
                break

    # if plot_weights:
    if plot_weights:
        boundaries = [np.min(weights), -0.001, 0.001, np.max(weights)]
        print(boundaries)

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
