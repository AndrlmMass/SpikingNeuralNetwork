import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from dataclasses import dataclass
import numpy as np
import matplotlib

from src.plot.weights import (
    create_3D_weights_plot,
    plot_weights_individual,
    plot_single_neuron_weights,
)

matplotlib.use("Agg")


@dataclass
class WeightFactory:
    N: int
    N_x: int
    N_exc: int
    N_inh: int
    rng: np.random.Generator
    # density params
    w_dense_se: float
    w_dense_ee: float
    w_dense_ei: float
    w_dense_ie: float
    # peak params
    se_weights: float
    ee_weights: float
    ei_weights: float
    ie_weights: float
    random_weights: bool
    rf_scale: float = 1.0

    def __post_init__(self):
        self.H = int(np.sqrt(self.N_x))
        self.W = self.H
        self.H_e = int(np.sqrt(self.N_exc))
        self.W_e = self.H_e
        self.st = self.N_x
        self.ex = self.st + self.N_exc
        self.ih = self.ex + self.N_inh
        self.weights = np.zeros((self.N, self.N))
        # scale factors
        if not self.random_weights:
            self.ref_x = np.sqrt(self.H * self.W)
            self.ref_e = np.sqrt(self.H_e * self.W_e)
            self.sigma_se = self.rf_scale * (1.0 / self.ref_x) * self.ref_x
            self.sigma_ee = self.rf_scale * (1.0 / self.ref_e) * self.ref_e
            self.sigma_ei = self.rf_scale * (1.0 / self.ref_e) * self.ref_e
            self.r0 = self.rf_scale * (1.5 / self.ref_e) * self.ref_e
            self.sigma_r = self.rf_scale * (1.0 / self.ref_e) * self.ref_e
            self.local_r = self.rf_scale * (2.0 / self.ref_e) * self.ref_e
            self._fse = 1.0 / self.ref_x
            self._fee = 1.0 / self.ref_e
            self._fei = 1.0 / self.ref_e
            self._fr0 = 1.5 / self.ref_e
            self._fsr = 1.0 / self.ref_e
            self._flr = 2.0 / self.ref_e

    def build(self):
        if self.random_weights:
            self._fill_random_weights()
        else:
            self._fill_receptive_fields()
        return self.weights

    def sparse_indices(self, weights: np.ndarray) -> dict:
        """Compute all nonzero index arrays needed by Trainer — call after build()."""
        nz_rows_se, nz_cols_se = np.nonzero(weights[: self.st, self.st : self.ex])
        nz_rows_ee, nz_cols_ee = np.nonzero(
            weights[self.st : self.ex, self.st : self.ex]
        )
        nz_rows_exc, nz_cols_exc = np.nonzero(weights[: self.ex, self.st : self.ex])
        nz_cols_exc = nz_cols_exc + self.st  # → global
        nonzero_pre_idx = [
            np.nonzero(weights[: self.ex, i])[0] for i in range(self.st, self.ex)
        ]
        return dict(
            nz_rows_se=nz_rows_se,
            nz_cols_se=nz_cols_se,
            nz_rows_ee=nz_rows_ee,
            nz_cols_ee=nz_cols_ee,
            nz_rows_exc=nz_rows_exc,
            nz_cols_exc=nz_cols_exc,
            nonzero_pre_idx=nonzero_pre_idx,
        )

    def initial_sums(self, weights: np.ndarray, reg_mode: str) -> tuple:
        """Compute initial_sums_se/ee for the regularizer."""
        if reg_mode == "static":
            return np.zeros(1), np.zeros(1)
        elif reg_mode == "post":
            return (
                weights[: self.st, self.st : self.ex].sum(axis=0),
                weights[self.st : self.ex, self.st : self.ex].sum(axis=0),
            )
        elif reg_mode == "layer":
            return (
                weights[: self.st, self.st : self.ex].sum(),
                weights[self.st : self.ex, self.st : self.ex].sum(),
            )

    def _fill_receptive_fields(self):
        W_se = gaussian_se_weights(
            self.N_x,
            self.N_exc,
            self.H,
            self.W,
            self.H_e,
            self.W_e,
            rng=self.rng,
            sigma=self.rf_scale * self._fse * self.ref_x,
            peak=self.se_weights,
            fraction=self.w_dense_se,
            torus=True,
        )
        self.weights[: self.st, self.st : self.ex] = W_se

        W_ee = gaussian_ee_weights(
            self.N_exc,
            self.H_e,
            self.W_e,
            sigma=self.rf_scale * self._fee * self.ref_e,
            peak=self.ee_weights,
            frac=self.w_dense_ee,
        )
        self.weights[self.st : self.ex, self.st : self.ex] = W_ee

        # --- E->I local pooling ---
        (
            W_ei,
            _,
            inh_centers,
        ) = gaussian_ei_local(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            H_e=self.H_e,
            W_e=self.W_e,
            sigma=self.rf_scale * self._fei * self.ref_e,
            peak=self.ei_weights,
            frac=self.w_dense_ei,
        )
        self.weights[self.st : self.ex, self.ex : self.ih] = W_ei

        # --- I->E far inhibition (center-surround / mexican hat) ---
        W_ie = mexican_hat_ie_far(
            H_e=self.H_e,
            W_e=self.W_e,
            inh_centers=inh_centers,
            peak=self.ie_weights,
            frac=self.w_dense_ie,
            rng=self.rng,
            r0=self.rf_scale * self._fr0 * self.ref_e,
            sigma_r=self.rf_scale * self._fsr * self.ref_e,
            local_r=self.rf_scale * self._flr * self.ref_e,
            N_exc=self.N_exc,
        )

        self.weights[self.ex : self.ih, self.st : self.ex] = W_ie

    def _fill_random_weights(self):
        # Create weights based on affinity rates
        mask_ee = self.rng.random((self.N_exc, self.N_exc)) < self.w_dense_ee
        mask_ei = self.rng.random((self.N_exc, self.N_inh)) < self.w_dense_ei
        mask_ie = self.rng.random((self.N_inh, self.N_exc)) < self.w_dense_ie
        mask_se = self.rng.random((self.N_x, self.N_exc)) < self.w_dense_se

        # input poisson weights
        self.weights[: self.st, self.st : self.ex][mask_se] = self.se_weights

        # hidden excitatory weights
        self.weights[self.st : self.ex, self.st : self.ex][mask_ee] = self.ee_weights
        # remove excitatory self-connecting (diagonal) weights
        np.fill_diagonal(self.weights[self.st : self.ex, self.st : self.ex], 0)

        self.weights[self.st : self.ex, self.ex : self.ih][mask_ei] = self.ei_weights

        # hidden inhibitory weights
        self.weights[self.ex : self.ih, self.st : self.ex][mask_ie] = self.ie_weights

        # remove recurrent connections from exc to inh
        inh_mask = self.weights[self.st : self.ex, self.ex : self.ih].T != 0
        self.weights[self.ex : self.ih, self.st : self.ex][inh_mask] = 0

    def plot(
        self,
        plot_weights_se=False,
        plot_weights_ee=False,
        plot_weights_ei=False,
        plot_weights_ie=False,
        plot_single_ee=False,
        plot_weights=False,
    ):
        H_i = W_i = int(np.sqrt(self.N_inh))
        w = self.weights  # shorthand

        if plot_weights_se:
            create_3D_weights_plot(
                w[: self.st, self.st : self.ex],
                "ST->EX Outgoing Weights",
                "input neuron",
                "input neuron",
                "connectivity strength",
                1,
                self.H,
                self.W,
            )
            create_3D_weights_plot(
                w[: self.st, self.st : self.ex],
                "ST->EX Incoming Weights",
                "excitatory neuron",
                "excitatory neuron",
                "connectivity strength",
                0,
                self.H_e,
                self.W_e,
            )
            plot_weights_individual(
                w[: self.st, self.st : self.ex],
                self.H,
                self.W,
                self.N_x,
                "ST->EX",
                dir="incoming",
            )

        if plot_weights_ee:
            create_3D_weights_plot(
                w[self.st : self.ex, self.st : self.ex],
                "EX->EX Outgoing Weights",
                "excitatory neuron",
                "excitatory neuron",
                "connectivity strength",
                1,
                self.H_e,
                self.W_e,
            )
            create_3D_weights_plot(
                w[self.st : self.ex, self.st : self.ex],
                "EX->EX Incoming Weights",
                "excitatory neuron",
                "inhibitory neuron",
                "connectivity strength",
                0,
                self.H_e,
                self.W_e,
            )
            plot_weights_individual(
                w[self.st : self.ex, self.st : self.ex],
                self.H_e,
                self.W_e,
                self.N_exc,
                "EX->EX",
                dir="incoming",
            )

        if plot_weights_ei:
            create_3D_weights_plot(
                w[self.st : self.ex, self.ex : self.ih],
                "E->I Outgoing Weights",
                "excitatory neuron",
                "excitatory neuron",
                "connectivity strength",
                1,
                self.H_e,
                self.W_e,
            )
            create_3D_weights_plot(
                w[self.st : self.ex, self.ex : self.ih],
                "E->I Incoming Weights",
                "inhibitory neuron",
                "inhibitory neuron",
                "connectivity strength",
                0,
                H_i,
                W_i,
            )
            plot_weights_individual(
                w[self.st : self.ex, self.ex : self.ih],
                self.H_e,
                self.W_e,
                self.N_inh,
                "E->I",
                dir="incoming",
            )

        if plot_weights_ie:
            create_3D_weights_plot(
                np.abs(w[self.ex : self.ih, self.st : self.ex]),
                "I->E Outgoing Weights",
                "inhibitory neuron",
                "inhibitory neuron",
                "connectivity strength",
                1,
                H_i,
                W_i,
            )
            create_3D_weights_plot(
                np.abs(w[self.ex : self.ih, self.st : self.ex]),
                "I->E Incoming Weights",
                "excitatory neuron",
                "excitatory neuron",
                "connectivity strength",
                0,
                self.H_e,
                self.W_e,
            )
            plot_weights_individual(
                np.abs(w[self.ex : self.ih, self.st : self.ex]),
                self.H_e,
                self.W_e,
                self.N_inh,
                "I->E",
                dir="outgoing",
            )

        if plot_single_ee:
            plot_single_neuron_weights(
                w, self.st, self.ex, self.H, self.W, self.H_e, self.W_e, id_=1400
            )

        if plot_weights:
            boundaries = [np.min(w), -0.001, 0.001, np.max(w)]
            cmap = ListedColormap(["red", "white", "green"])
            norm = BoundaryNorm(boundaries, ncolors=cmap.N)
            plt.imshow(w, cmap=cmap, norm=norm)
            plt.gca().invert_yaxis()
            plt.title("Weights")
            plt.show()

            W_ie = w[self.ex : self.ih, self.st : self.ex]
            W_ee = w[self.st : self.ex, self.st : self.ex]
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
                "mean inh/exc ratio =",
                inh_in_per_E.mean() / (exc_in_per_E.mean() + 1e-12),
            )
            print(
                "max  inh/exc ratio =",
                inh_in_per_E.max() / (exc_in_per_E.max() + 1e-12),
            )


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

    # remove small weights
    W_ei[W_ei < 0.01] = 0.0

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
    rng,
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
        centers += rng.uniform(-jitter, jitter, size=centers.shape)
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

    # remove small weights
    W_ie[W_ie > -0.01] = 0.0

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
    rng,
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
        centers += rng.uniform(-jitter, jitter, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], 0, H - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, W - 1)

    # Input coords
    inp_coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)

    # Distances: (N_exc, N_x)
    if torus:
        d2 = torus_d2(centers, inp_coords, H, W)
    else:
        d2 = ((centers[:, None, :] - inp_coords[None, :, :]) ** 2).sum(axis=2)

    sigmas = rng.normal(loc=sigma, scale=0.3, size=(N_exc, 1))
    sigmas = np.clip(sigmas, 1.0, None)  # enforce positive + not-too-small
    G = np.exp(-d2 / (2 * sigmas**2))

    # pre x post
    W_se = G.T

    # remove small weights
    W_se[W_se < 0.01] = 0.0

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

    # remove small weights
    W[W < 0.01] = 0.0

    # ensure weight strength complies with the sum of compliant weights
    W = weight_compliance(frac=frac, N=N_exc, weights=W, peak=peak, type="W_ee")

    return W


@dataclass
class Network:
    N: int
    N_exc: int
    N_inh: int
    st: int
    resting_membrane: float
    spike_threshold_default: float
    time: int
    data: np.ndarray

    def __post_init__(self):
        if self.time == 0:
            raise ValueError("total_time_test must be > 0 to create test arrays")

    def build(self):
        # create membrane potential
        membrane_potential = np.full(self.N - self.st, self.resting_membrane)

        # create spikes array
        spikes = np.zeros((self.time, self.N), dtype=np.int8)
        spikes[:, : self.st] = self.data

        # create spike traces for each neuron
        spike_trace = np.zeros(self.N - self.N_inh, dtype=np.float32)

        # create missing arrays
        I_syn_exc = np.zeros(self.N_exc)
        I_syn_inh = np.zeros(self.N_inh)
        a = np.zeros(self.N_exc + self.N_inh)

        # create spike threshold array
        spike_threshold = np.full(
            shape=(self.N - self.st),
            fill_value=self.spike_threshold_default,
            dtype=float,
        )

        # create pre-targets for STDP
        x_tar_se = np.zeros(self.N_x)
        x_tar_ee = np.zeros(self.N_exc)

        return (
            membrane_potential,
            spikes,
            spike_trace,
            I_syn_exc,
            I_syn_inh,
            a,
            spike_threshold,
            x_tar_se,
            x_tar_ee,
        )
