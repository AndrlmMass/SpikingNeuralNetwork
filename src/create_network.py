import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


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


def inh_grid_shape(N_inh, H_e, W_e):
    aspect = W_e / H_e
    H_i = int(np.floor(np.sqrt(N_inh / aspect)))
    H_i = max(H_i, 1)
    W_i = int(np.ceil(N_inh / H_i))
    return H_i, W_i


def grid_home_indices(N_inh, H_e, W_e):
    H_i, W_i = inh_grid_shape(N_inh, H_e, W_e)

    rs = np.linspace(0, H_e - 1, H_i)
    cs = np.linspace(0, W_e - 1, W_i)

    homes_rc = np.array([(r, c) for r in rs for c in cs], dtype=float)[:N_inh]

    # round to nearest E cell and convert (r,c) -> linear index
    r_idx = np.clip(np.rint(homes_rc[:, 0]).astype(int), 0, H_e - 1)
    c_idx = np.clip(np.rint(homes_rc[:, 1]).astype(int), 0, W_e - 1)
    home_idx = r_idx * W_e + c_idx
    return home_idx


def gaussian_ei_local(N_exc, N_inh, H_e, W_e, sigma=1.5, peak=1.0, frac=None):
    assert H_e * W_e == N_exc
    coords = exc_coords(H_e, W_e)  # (N_exc, 2) integer grid

    # Place I centers on a perfectly uniform continuous grid
    H_i, W_i = inh_grid_shape(N_inh, H_e, W_e)
    rs = np.linspace(0, H_e - 1, H_i)
    cs = np.linspace(0, W_e - 1, W_i)
    centers = np.array([(r, c) for r in rs for c in cs], dtype=float)[:N_inh]

    # For mexican_hat later: map each I center to nearest E neuron index
    d2_home = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    home_idx = d2_home.argmin(axis=1)

    # Gaussian E→I weights using continuous centers (no rounding artifacts)
    d2 = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    W_ei = np.exp(-d2 / (2 * sigma**2))

    W_ei[W_ei < 0.01] = 0.0

    W_ei /= W_ei.max(axis=0, keepdims=True) + 1e-12
    W_ei *= peak

    if frac is not None:
        W_ei = topk_mask_per_col(W_ei, frac)   # keeps top frac of excitatory inputs per inhibitory neuron

    return W_ei, home_idx, centers

def mexican_hat_ie_far(
    H_e, W_e, inh_centers, peak=1.0, frac=None, r0=2.0, sigma_r=2.0, local_r=2.0
):
    coords = exc_coords(H_e, W_e)  # (N_exc,2)
    centers = inh_centers           # (N_inh,2) — continuous positions

    # Distances from inhibitory center to each excit neuron -> (N_inh, N_exc)
    d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    d = np.sqrt(d2)
    ring = np.exp(-((d - r0) ** 2) / (2 * sigma_r**2))
    ring /= ring.max(axis=1, keepdims=True) + 1e-12
    W_ie = peak * ring

    # Optional: sparsify per inhibitory neuron (per row)
    if frac is not None:
        W_ie = enforce_topk_per_row_abs(W_ie, frac=frac)

    # Guarantee "don't inhibit your own local pool" strongly:
    # (hard zero at the exact center exc neuron)
    # inside mexican_hat_ie_far after computing d
    W_ie[d <= local_r] = 0.0

    return W_ie


def gaussian_se_weights(
    N_x, N_exc, H, W, H_e, W_e, sigma=2.0, peak=1.0, truncate=0.01, fraction=0.1
):
    """
    Returns W_se with shape (N_x, N_exc): input->exc weights.
    Each excitatory neuron gets a Gaussian receptive field over the (H,W) input grid.
    """
    assert H * W == N_x
    assert H_e * W_e >= N_exc

    inp_coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)

    # Use the SAME grid as E→E, E→I, I→E
    centers_r = np.linspace(0, H - 1, H_e)
    centers_c = np.linspace(0, W - 1, W_e)
    centers = np.array([(r, c) for r in centers_r for c in centers_c], dtype=float)[:N_exc]

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

def create_3D_weights_plot(weights, title, x_label, y_label, z_label, axis_flip, H_, W_):
    total_input = weights.sum(axis=axis_flip)
    Z = total_input.reshape(H_, W_)
    x = np.arange(W_)
    y = np.arange(H_)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.show()
    return fig, ax

def plot_weights_individual(weights, H_, W_, N, title):
    for j in range(N // 2, N):
        rf = weights[j, :].reshape(H_, W_)
        plt.imshow(rf)
        plt.title(f"{title} {j}")
        plt.colorbar()
        plt.show()
        if input("Press Enter to continue...") == "q":
            break

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
    random_weights,
    se_weights,
    ee_weights,
    ei_weights,
    ie_weights,
    plot_weights,
    rf_scale=1.0,
):
    weights = np.zeros(shape=(N, N))

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory
    plot_weights_st = False
    plot_weights_ex = False
    plot_weights_ei = False
    plot_weights_ie = False

    # input poisson weights
    H = int(np.sqrt(N_x))
    W = H

    H_e, W_e = 25, 40

    ref_x = np.sqrt(H * W)
    ref_e = np.sqrt(H_e * W_e)

    _fse = 1.0 / ref_x
    _fee = 1.0 / ref_e
    _fei = 1.0 / ref_e
    _fr0 = 3.0 / ref_e
    _fsr = 0.5 / ref_e
    _flr = 0.5 / ref_e

    if random_weights:
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
        # remove excitatory self-connecting (diagonal) weights
        np.fill_diagonal(weights[st:ex, st:ex], 0)

        weights[st:ex, ex:ih][mask_ei] = ei_weights

        # hidden inhibitory weights
        weights[ex:ih, st:ex][mask_ie] = ie_weights

        # remove recurrent connections from exc to inh
        inh_mask = weights[st:ex, ex:ih].T != 0
        weights[ex:ih, st:ex][inh_mask] = 0

    else:

        weights_st_ex = gaussian_se_weights(
            N_x,
            N_exc,
            H,
            W,
            H_e,
            W_e,
            sigma=rf_scale * _fse * ref_x,
            peak=se_weights,
            truncate=0.01,
            fraction=w_dense_se,
        )
        weights[:st, st:ex] = weights_st_ex

        weights_ex_ex = gaussian_ee_weights(
            N_exc,
            H_e,
            W_e,
            sigma=rf_scale * _fee * ref_e,
            peak=ee_weights,
            frac=w_dense_ee,
        )
        weights[st:ex, st:ex] = weights_ex_ex

        # --- E->I local pooling ---
        W_ei, _, inh_centers = gaussian_ei_local(
            N_exc=N_exc,
            N_inh=N_inh,
            H_e=H_e,
            W_e=W_e,
            sigma=rf_scale * _fei * ref_e,  # local pooling radius
            peak=ei_weights,  # excitatory onto inhibitory (positive)
            frac=w_dense_ei,  # density meaning: fraction of exc inputs to each I
        )
        weights[st:ex, ex:ih] = W_ei

        H_i, W_i = 10, 25

        # --- I->E far inhibition (center-surround / mexican hat) ---
        W_ie = mexican_hat_ie_far(
            H_e=H_e,
            W_e=W_e,
            inh_centers=inh_centers,
            peak=ie_weights,  
            frac=w_dense_ie,
            r0=rf_scale * _fr0 * ref_e,
            sigma_r=rf_scale * _fsr * ref_e,
            local_r=rf_scale * _flr * ref_e,
        )

        weights[ex:ih, st:ex] = W_ie

    if plot_weights_st:
        create_3D_weights_plot(weights[:st, st:ex], "ST->EX Outgoing Weights", "input neuron", "input neuron", "connectivity strength", 1, H, W)
        create_3D_weights_plot(weights[:st, st:ex], "ST->EX Incoming Weights", "excitatory neuron", "excitatory neuron", "connectivity strength", 0, H_e, W_e)
        plot_weights_individual(weights[:st, st:ex], H_e, W_e, N_x, "ST->EX")

    if plot_weights_ex:
        create_3D_weights_plot(weights[st:ex, st:ex], "EX->EX Outgoing Weights", "excitatory neuron", "excitatory neuron", "connectivity strength", 1, H_e, W_e)
        create_3D_weights_plot(weights[st:ex, st:ex], "EX->EX Incoming Weights", "excitatory neuron", "inhibitory neuron", "connectivity strength", 0, H_e, W_e)
        plot_weights_individual(weights[st:ex, st:ex], H_e, W_e, N_exc, "EX->EX")
    if plot_weights_ei:
        create_3D_weights_plot(weights[st:ex, ex:ih], "E->I Outgoing Weights", "excitatory neuron", "excitatory neuron", "connectivity strength", 1, H_e, W_e)
        create_3D_weights_plot(weights[st:ex, ex:ih], "E->I Incoming Weights", "inhibitory neuron", "inhibitory neuron", "connectivity strength", 0, H_i, W_i)
        plot_weights_individual(weights[st:ex, ex:ih], H_i, W_i, N_exc, "E->I")
    if plot_weights_ie:
        create_3D_weights_plot(np.abs(weights[ex:ih, st:ex]), "I->E Outgoing Weights", "inhibitory neuron", "inhibitory neuron", "connectivity strength", 1, H_i, W_i)
        create_3D_weights_plot(np.abs(weights[ex:ih, st:ex]), "I->E Incoming Weights", "excitatory neuron", "excitatory neuron", "connectivity strength", 0, H_e, W_e)
        plot_weights_individual(np.abs(weights[ex:ih, st:ex]), H_e, W_e, N_inh, "I->E")
    
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

        inh_in_per_E = (-W_ie).sum(axis=0)
        exc_in_per_E = weights_ex_ex.sum(axis=0)

        print(
            "exc incoming per E: min/mean/max =",
            exc_in_per_E.min(),
            exc_in_per_E.mean(),
            exc_in_per_E.max(),
        )
        print(
            "inh incoming per E: min/mean/max =",
            inh_in_per_E.min(),
            inh_in_per_E.mean(),
            inh_in_per_E.max(),
        )

        print(
            "mean inh/exc ratio =", inh_in_per_E.mean() / (exc_in_per_E.mean() + 1e-12)
        )
        print("max  inh/exc ratio =", inh_in_per_E.max() / (exc_in_per_E.max() + 1e-12))

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

    # create spike traces for each neuron
    spike_trace = np.zeros(N, dtype=np.float32)

    return (
        membrane_potential_train,
        membrane_potential_test,
        spikes_train,
        spikes_test,
        spike_trace,
    )
