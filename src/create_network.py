import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import networkx as nx
import numpy as np
import matplotlib

matplotlib.use("TkAgg")


def exc_coords(H_e, W_e):
    return np.array([(r, c) for r in range(H_e) for c in range(W_e)], dtype=float)


def inh_grid_shape(N_inh, H_e, W_e):
    aspect = W_e / H_e
    H_i = int(np.floor(np.sqrt(N_inh / aspect)))
    H_i = max(H_i, 1)
    W_i = int(np.ceil(N_inh / H_i))
    return H_i, W_i


def weight_compliance(frac, N, weights, peak, type):
    current_sum = np.sum(weights, axis=1)
    current_sum = np.where(current_sum == 0, 1e-12, current_sum)  # guard dead rows
    optimal_sum = frac * N * peak  # drop int() — truncation shifts your budget
    ratio = optimal_sum / current_sum
    weights = weights * ratio[:, None]
    # print changes
    weights_abs = abs(weights)
    structural = (weights_abs > 0).mean(axis=1).mean()  # should ≈ frac
    effective = weights_abs.sum(axis=1).mean() / (N * peak)  # should == frac exactly
    print(f"{type}: structural = {structural} and effective = {effective}")
    return weights


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


def gaussian_ei_local(
    N_exc, N_inh, H_e, W_e, sigma=1.5, peak=1.0, frac=None, torus=True
):
    assert H_e * W_e == N_exc
    coords = exc_coords(H_e, W_e)  # (N_exc, 2) integer grid

    # Place I centers on a perfectly uniform continuous grid
    H_i, W_i = inh_grid_shape(N_inh, H_e, W_e)
    rs = np.linspace(0, H_e - 1, H_i)
    cs = np.linspace(0, W_e - 1, W_i)
    centers = np.array([(r, c) for r in rs for c in cs], dtype=float)[:N_inh]

    # Gaussian E→I weights using continuous centers (no rounding artifacts)
    if torus:
        d2 = torus_d2(centers, coords, H_e, W_e)  # → (N_inh, N_exc)
    else:
        d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    W_ei = np.exp(-d2 / (2 * sigma**2))

    W_ei[W_ei < 0.01] = 0.0

    W_ei /= W_ei.max(axis=0, keepdims=True) + 1e-12
    W_ei *= peak

    # ensure weight strength complies with the sum of compliant weights
    W_ei = weight_compliance(frac=frac, N=N_exc, weights=W_ei, peak=peak, type="W_ei")

    W_ei = W_ei.T

    return (
        W_ei,
        d2.argmin(axis=1),
        centers,
    )


def mexican_hat_ie_far(
    H_e,
    W_e,
    N_exc,
    inh_centers,
    peak=1.0,
    frac=None,
    r0=2.0,
    sigma_r=2.0,
    local_r=2.0,
    jitter=0.1,
    torus=True,
):
    coords = exc_coords(H_e, W_e)  # (N_exc,2)
    centers = inh_centers  # (N_inh,2) — continuous positions

    if jitter is not None and jitter > 0:
        centers += np.random.uniform(-jitter, jitter, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], 0, H_e - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, W_e - 1)

    # Distances: (N_exc, N_x)
    if torus:
        d2 = torus_d2(centers, coords, H_e, W_e)
    else:
        d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    d = np.sqrt(d2)

    # ring profile
    ring = np.exp(-((d - r0) ** 2) / (2 * sigma_r**2))

    # soft center suppression: 1 at the ring, ~0 at the center
    # sigma_local controls how wide the no-inhibition zone is
    center_suppress = 1.0 - np.exp(-(d**2) / (2 * local_r**2))

    ring = ring * center_suppress

    ring /= ring.max(axis=1, keepdims=True) + 1e-12
    W_ie = peak * ring

    # ensure weight strength complies with the sum of compliant weights
    W_ie = weight_compliance(frac=frac, N=N_exc, weights=W_ie, peak=peak, type="W_ie")

    return W_ie


def torus_d2(centers, coords, H, W):
    """
    centers: (N_post, 2)  [row, col] float or int
    coords:  (N_pre,  2)  [row, col] float or int
    returns: (N_post, N_pre) squared torus distance
    """
    dr = np.abs(centers[:, None, 0] - coords[None, :, 0])
    dc = np.abs(centers[:, None, 1] - coords[None, :, 1])
    dr = np.minimum(dr, H - dr)
    dc = np.minimum(dc, W - dc)
    return dr * dr + dc * dc


def gaussian_se_weights(
    N_x,
    N_exc,
    H,
    W,
    H_e,
    W_e,
    sigma=2.0,
    peak=1.0,
    fraction=0.1,
    torus=True,
    jitter=0.25,
):
    assert H * W == N_x
    assert H_e * W_e == N_exc

    # Exc neurons live on (H_e, W_e) grid
    coords_e = exc_coords(H_e, W_e)  # (N_exc, 2) in exc-grid units

    # Map exc-grid coords -> input-grid coords (continuous)
    centers = np.empty_like(coords_e)
    centers[:, 0] = coords_e[:, 0] * (H - 1) / max(H_e - 1, 1)
    centers[:, 1] = coords_e[:, 1] * (W - 1) / max(W_e - 1, 1)

    # Break perfect alignment with discrete pixels (optional but often helps)
    if jitter is not None and jitter > 0:
        centers += np.random.uniform(-jitter, jitter, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], 0, H - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, W - 1)

    # Input coords
    inp_coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)

    # Distances: (N_exc, N_x)
    if torus:
        d2 = torus_d2(centers, inp_coords, H, W)
    else:
        d2 = ((centers[:, None, :] - inp_coords[None, :, :]) ** 2).sum(axis=2)

    sigmas = np.random.normal(loc=sigma, scale=0.3, size=(N_exc, 1))
    sigmas = np.clip(sigmas, 1.0, None)  # enforce positive + not-too-small
    G = np.exp(-d2 / (2 * sigmas**2))

    # pre x post
    W_se = G.T

    # ensure weight strength complies with the sum of compliant weights
    W_se = weight_compliance(
        frac=fraction, N=N_exc, weights=W_se, peak=peak, type="W_se"
    )

    return W_se


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
    N_exc,
    H_e,
    W_e,
    sigma=2.0,
    peak=1.0,
    frac=None,
    self_zero=True,
    torus=True,
):
    """
    Returns W_ee (N_exc, N_exc): excitatory->excitatory.
    Exc neurons live on a (H_e, W_e) grid.
    """
    assert H_e * W_e == N_exc

    coords = np.array(
        [(r, c) for r in range(H_e) for c in range(W_e)], dtype=float
    )  # (N_exc,2)

    if torus:
        d2 = torus_d2(coords, coords, H_e, W_e)
    else:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    W = np.exp(-d2 / (2 * sigma**2))

    # normalize rows so max=1 then scale
    W /= W.max(axis=1, keepdims=True) + 1e-12
    W *= peak

    if self_zero:
        np.fill_diagonal(W, 0.0)

    # ensure weight strength complies with the sum of compliant weights
    W = weight_compliance(frac=frac, N=N_exc, weights=W, peak=peak, type="W_ee")

    return W


def create_3D_weights_plot(
    weights, title, x_label, y_label, z_label, axis_flip, H_, W_
):
    total_input = weights.sum(axis=axis_flip)
    Z = total_input.reshape(H_, W_)
    x = np.arange(W_)
    y = np.arange(H_)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()
    return fig, ax


def plot_weights_individual(weights, H_, W_, N, title, dir):
    if dir == "outgoing":
        for j in range(N // 2, N):
            rf = weights[j, :].reshape(H_, W_)
            plt.imshow(rf)
            plt.title(f"{title} {j}")
            plt.colorbar()
            plt.show()
            if input("Press Enter to continue...") == "q":
                break
    elif dir == "incoming":
        for j in range(N // 2, N):
            rf = weights[:, j].reshape(H_, W_)
            plt.imshow(rf)
            plt.title(f"{title} {j}")
            plt.colorbar()
            plt.show()
            if input("Press Enter to continue...") == "q":
                break


def plot_single_neuron_weights(
    weights,
    st,
    ex,
    H,
    W,
    H_e,
    W_e,
    id_=None,
    loop=True,
):
    if loop:
        N = ex - st
    else:
        N = 1
    for i in range(st, ex):
        if id_ is not None:
            id = id_ + i - st
        else:
            id = i
        # create plot structure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        # fetch all weights related to id
        w_se = weights[:st, id].reshape(H, W)
        w_ee = weights[id, st:ex].reshape(H_e, W_e)
        ih_id = np.argmax(weights[id, ex:]) + ex
        w_ie = weights[st:ex, ih_id].reshape(H_e, W_e)
        w_ei = np.abs(weights[ih_id, st:ex]).reshape(H_e, W_e)

        # plot incoming and outgoing weights
        im1 = ax1.imshow(w_se)
        im2 = ax2.imshow(w_ee)
        im3 = ax4.imshow(w_ei)
        im4 = ax3.imshow(w_ie)

        # add colorbars
        for ax, im in zip([ax1, ax2, ax3, ax4], [im1, im2, im3, im4]):
            fig.colorbar(im, ax=ax)

        # set labels and title
        fig.suptitle(f"Incoming and outgoing weights for exc: {id}")
        ax1.set_title("Stimulation to Excitatory")
        ax2.set_title("Excitatory to Excitatory")
        ax3.set_title("Excitatory to Inhibitory")
        ax4.set_title("Inhibitory to Excitatory")

        plt.show()

        id = None
        cont = input("Continue?")

        if cont != "":
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
    plot_weights_se = False
    plot_weights_ee = False
    plot_weights_ei = False
    plot_weights_ie = False
    plot_single_ee = False

    # input poisson weights
    H = int(np.sqrt(N_x))
    W = H

    H_e, W_e = 32, 32

    ref_x = np.sqrt(H * W)
    ref_e = np.sqrt(H_e * W_e)

    _fse = 1.0 / ref_x
    _fee = 2.0 / ref_e
    _fei = 2.0 / ref_e
    _fr0 = 3.0 / ref_e
    _fsr = 2.0 / ref_e
    _flr = 2.0 / ref_e

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
        W_se = gaussian_se_weights(
            N_x,
            N_exc,
            H,
            W,
            H_e,
            W_e,
            sigma=rf_scale * _fse * ref_x,
            peak=se_weights,
            fraction=w_dense_se,
            torus=True,
        )
        weights[:st, st:ex] = W_se

        W_ee = gaussian_ee_weights(
            N_exc,
            H_e,
            W_e,
            sigma=rf_scale * _fee * ref_e,
            peak=ee_weights,
            frac=w_dense_ee,
        )
        weights[st:ex, st:ex] = W_ee

        # --- E->I local pooling ---
        (
            W_ei,
            _,
            inh_centers,
        ) = gaussian_ei_local(
            N_exc=N_exc,
            N_inh=N_inh,
            H_e=H_e,
            W_e=W_e,
            sigma=rf_scale * _fei * ref_e,  # local pooling radius
            peak=ei_weights,  # excitatory onto inhibitory (positive)
            frac=w_dense_ei,  # density meaning: fraction of exc inputs to each I
        )
        weights[st:ex, ex:ih] = W_ei

        H_i, W_i = 15, 15

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
            N_exc=N_exc,
        )

        weights[ex:ih, st:ex] = W_ie
    # if plot_by_exc_neuron:
    #     # add plotting function that compares all inputs and outputs to a single excitatory neuron
    #     ...

    if plot_weights_se:
        create_3D_weights_plot(
            weights[:st, st:ex],
            "ST->EX Outgoing Weights",
            "input neuron",
            "input neuron",
            "connectivity strength",
            1,
            H,
            W,
        )
        create_3D_weights_plot(
            weights[:st, st:ex],
            "ST->EX Incoming Weights",
            "excitatory neuron",
            "excitatory neuron",
            "connectivity strength",
            0,
            H_e,
            W_e,
        )
        plot_weights_individual(
            weights[:st, st:ex], H, W, N_x, "ST->EX", dir="incoming"
        )

    if plot_weights_ee:
        create_3D_weights_plot(
            weights[st:ex, st:ex],
            "EX->EX Outgoing Weights",
            "excitatory neuron",
            "excitatory neuron",
            "connectivity strength",
            1,
            H_e,
            W_e,
        )
        create_3D_weights_plot(
            weights[st:ex, st:ex],
            "EX->EX Incoming Weights",
            "excitatory neuron",
            "inhibitory neuron",
            "connectivity strength",
            0,
            H_e,
            W_e,
        )
        plot_weights_individual(
            weights[st:ex, st:ex], H_e, W_e, N_exc, "EX->EX", dir="incoming"
        )
    if plot_weights_ei:
        create_3D_weights_plot(
            weights[st:ex, ex:ih],
            "E->I Outgoing Weights",
            "excitatory neuron",
            "excitatory neuron",
            "connectivity strength",
            1,
            H_e,
            W_e,
        )
        create_3D_weights_plot(
            weights[st:ex, ex:ih],
            "E->I Incoming Weights",
            "inhibitory neuron",
            "inhibitory neuron",
            "connectivity strength",
            0,
            H_i,
            W_i,
        )
        plot_weights_individual(
            weights[st:ex, ex:ih], H_e, W_e, N_inh, "E->I", dir="incoming"
        )
    if plot_weights_ie:
        create_3D_weights_plot(
            np.abs(weights[ex:ih, st:ex]),
            "I->E Outgoing Weights",
            "inhibitory neuron",
            "inhibitory neuron",
            "connectivity strength",
            1,
            H_i,
            W_i,
        )
        create_3D_weights_plot(
            np.abs(weights[ex:ih, st:ex]),
            "I->E Incoming Weights",
            "excitatory neuron",
            "excitatory neuron",
            "connectivity strength",
            0,
            H_e,
            W_e,
        )
        plot_weights_individual(
            np.abs(weights[ex:ih, st:ex]), H_e, W_e, N_inh, "I->E", dir="outgoing"
        )
    if plot_single_ee:
        plot_single_neuron_weights(
            weights,
            st,
            ex,
            H,
            W,
            H_e,
            W_e,
            id_=1400,
        )

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
        exc_in_per_E = W_ee.sum(axis=0)

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

    membrane_potential_train = np.zeros(ih - st)
    if total_time_train > 0:
        membrane_potential_train[:] = resting_membrane

    membrane_potential_test = np.zeros(ih - st)
    if total_time_test > 0:
        membrane_potential_test[:] = resting_membrane

    spikes_train = np.zeros((total_time_train, N), dtype=np.int8)
    if data_train is not None and total_time_train > 0:
        spikes_train[:, :st] = data_train

    spikes_test = np.zeros((total_time_test, N), dtype=np.int8)
    if data_test is not None and total_time_test > 0:
        spikes_test[:, :st] = data_test

    # create spike traces for each neuron
    spike_trace = np.zeros(N - N_inh, dtype=np.float32)

    return (
        membrane_potential_train,
        membrane_potential_test,
        spikes_train,
        spikes_test,
        spike_trace,
    )
