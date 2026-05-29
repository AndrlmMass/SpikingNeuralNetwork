import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from dataclasses import dataclass
import numpy as np
import matplotlib

from neurosnn._plot.weights import (
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
    w_dense_se: float
    w_dense_ee: float
    w_dense_ei: float
    w_dense_ie: float
    se_weights: float
    ee_weights: float
    ei_weights: float
    ie_weights: float
    random_weights: bool
    rf_scale: float = 1.0
    oriented_rf: bool = False
    sigma_x: float = 3.0
    gamma: float = 0.4
    n_orientations: int = 4
    r_cut_factor: float = 3.0
    sigma_x_lognormal_std: float = 0.0
    sigma_x_lognormal_max: float = 0.0   # 0 = no upper clip
    orientation_mode: str = "block"      # "block" or "interleaved"
    sigma_ee_mean: float = 0.0          # 0 = auto-compute from rf_scale
    sigma_ee_lognormal_std: float = 0.0  # 0 = disabled (fixed sigma_ee)
    sigma_se_mean: float = 0.0          # 0 = auto-compute from rf_scale
    sigma_se_lognormal_std: float = 0.0  # 0 = disabled (fixed sigma_se)

    def __post_init__(self):
        self.H = int(np.sqrt(self.N_x))
        self.W = self.H
        self.H_e = int(np.sqrt(self.N_exc))
        self.W_e = self.H_e
        self.st = self.N_x
        self.ex = self.st + self.N_exc
        self.ih = self.ex + self.N_inh
        self.weights = np.zeros((self.N, self.N))
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
        nz_rows_se, nz_cols_se = np.nonzero(weights[: self.st, self.st : self.ex])
        nz_rows_ee, nz_cols_ee = np.nonzero(
            weights[self.st : self.ex, self.st : self.ex]
        )
        nz_rows_exc, nz_cols_exc = np.nonzero(weights[: self.ex, self.st : self.ex])
        nz_cols_exc = nz_cols_exc + self.st
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
        if reg_mode == "static":
            return (
                weights[: self.st, self.st : self.ex].sum(),
                weights[self.st : self.ex, self.st : self.ex].sum(),
            )
        elif reg_mode == "neuron":
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
        if self.oriented_rf:
            W_se = oriented_gaussian_se_weights(
                N_x=self.N_x,
                N_exc=self.N_exc,
                input_size=self.H,
                sigma_x=self.sigma_x,
                gamma=self.gamma,
                n_orientations=self.n_orientations,
                r_cut_factor=self.r_cut_factor,
                sigma_x_lognormal_std=self.sigma_x_lognormal_std,
                sigma_x_lognormal_max=self.sigma_x_lognormal_max,
                orientation_mode=self.orientation_mode,
                peak=self.se_weights,
                fraction=self.w_dense_se,
                rng=self.rng,
            )
        else:
            sigma_se = (
                self.sigma_se_mean
                if self.sigma_se_mean > 0.0
                else self.rf_scale * self._fse * self.ref_x
            )
            W_se = gaussian_se_weights(
                self.N_x,
                self.N_exc,
                self.H,
                self.W,
                self.H_e,
                self.W_e,
                rng=self.rng,
                sigma=sigma_se,
                peak=self.se_weights,
                fraction=self.w_dense_se,
                torus=True,
                sigma_se_lognormal_std=self.sigma_se_lognormal_std,
            )
        self.weights[: self.st, self.st : self.ex] = W_se

        sigma_ee = (
            self.sigma_ee_mean
            if self.sigma_ee_mean > 0.0
            else self.rf_scale * self._fee * self.ref_e
        )
        W_ee = gaussian_ee_weights(
            self.N_exc,
            self.H_e,
            self.W_e,
            sigma=sigma_ee,
            peak=self.ee_weights,
            frac=self.w_dense_ee,
            sigma_lognormal_std=self.sigma_ee_lognormal_std,
            rng=self.rng,
        )
        self.weights[self.st : self.ex, self.st : self.ex] = W_ee

        W_ei, _, inh_centers = gaussian_ei_local(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            H_e=self.H_e,
            W_e=self.W_e,
            sigma=self.rf_scale * self._fei * self.ref_e,
            peak=self.ei_weights,
            frac=self.w_dense_ei,
        )
        self.weights[self.st : self.ex, self.ex : self.ih] = W_ei

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
        mask_ee = self.rng.random((self.N_exc, self.N_exc)) < self.w_dense_ee
        mask_ei = self.rng.random((self.N_exc, self.N_inh)) < self.w_dense_ei
        mask_ie = self.rng.random((self.N_inh, self.N_exc)) < self.w_dense_ie
        mask_se = self.rng.random((self.N_x, self.N_exc)) < self.w_dense_se

        self.weights[: self.st, self.st : self.ex][mask_se] = self.se_weights
        self.weights[self.st : self.ex, self.st : self.ex][mask_ee] = self.ee_weights
        np.fill_diagonal(self.weights[self.st : self.ex, self.st : self.ex], 0)
        self.weights[self.st : self.ex, self.ex : self.ih][mask_ei] = self.ei_weights
        self.weights[self.ex : self.ih, self.st : self.ex][mask_ie] = self.ie_weights

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
        w = self.weights

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
    current_sum = np.where(current_sum == 0, 1e-12, current_sum)
    optimal_sum = frac * N * peak
    ratio = optimal_sum / current_sum
    weights = weights * ratio[:, None]
    weights_abs = abs(weights)
    structural = (weights_abs > 0).mean(axis=1).mean()
    effective = weights_abs.sum(axis=1).mean() / (N * peak)
    print(f"{type}: structural = {structural} and effective = {effective}")
    return weights


def grid_home_indices(N_inh, H_e, W_e):
    H_i, W_i = inh_grid_shape(N_inh, H_e, W_e)
    rs = np.linspace(0, H_e - 1, H_i)
    cs = np.linspace(0, W_e - 1, W_i)
    homes_rc = np.array([(r, c) for r in rs for c in cs], dtype=float)[:N_inh]
    r_idx = np.clip(np.rint(homes_rc[:, 0]).astype(int), 0, H_e - 1)
    c_idx = np.clip(np.rint(homes_rc[:, 1]).astype(int), 0, W_e - 1)
    home_idx = r_idx * W_e + c_idx
    return home_idx


def gaussian_ei_local(
    N_exc, N_inh, H_e, W_e, sigma=1.5, peak=1.0, frac=None, torus=True
):
    assert H_e * W_e == N_exc
    coords = exc_coords(H_e, W_e)

    H_i, W_i = inh_grid_shape(N_inh, H_e, W_e)
    rs = np.linspace(0, H_e - 1, H_i)
    cs = np.linspace(0, W_e - 1, W_i)
    centers = np.array([(r, c) for r in rs for c in cs], dtype=float)[:N_inh]

    if torus:
        d2 = torus_d2(centers, coords, H_e, W_e)
    else:
        d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    W_ei = np.exp(-d2 / (2 * sigma**2))
    W_ei[W_ei < 0.01] = 0.0
    W_ei /= W_ei.max(axis=0, keepdims=True) + 1e-12
    W_ei *= peak
    W_ei[W_ei < 0.01] = 0.0
    W_ei = weight_compliance(frac=frac, N=N_exc, weights=W_ei, peak=peak, type="W_ei")
    W_ei = W_ei.T

    return W_ei, d2.argmin(axis=1), centers


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
    coords = exc_coords(H_e, W_e)
    centers = inh_centers

    if jitter is not None and jitter > 0:
        centers += rng.uniform(-jitter, jitter, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], 0, H_e - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, W_e - 1)

    if torus:
        d2 = torus_d2(centers, coords, H_e, W_e)
    else:
        d2 = ((centers[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    d = np.sqrt(d2)
    ring = np.exp(-((d - r0) ** 2) / (2 * sigma_r**2))
    center_suppress = 1.0 - np.exp(-(d**2) / (2 * local_r**2))
    ring = ring * center_suppress
    ring /= ring.max(axis=1, keepdims=True) + 1e-12
    W_ie = peak * ring
    W_ie[np.abs(W_ie) < 0.01] = 0.0
    W_ie = weight_compliance(frac=frac, N=N_exc, weights=W_ie, peak=peak, type="W_ie")

    return W_ie


def torus_d2(centers, coords, H, W):
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
    rng,
    sigma=2.0,
    peak=1.0,
    fraction=0.1,
    torus=True,
    jitter=0.25,
    sigma_se_lognormal_std=0.0,
):
    assert H * W == N_x
    assert H_e * W_e == N_exc

    coords_e = exc_coords(H_e, W_e)
    centers = np.empty_like(coords_e)
    centers[:, 0] = coords_e[:, 0] * (H - 1) / max(H_e - 1, 1)
    centers[:, 1] = coords_e[:, 1] * (W - 1) / max(W_e - 1, 1)

    if jitter is not None and jitter > 0:
        centers += rng.uniform(-jitter, jitter, size=centers.shape)
        centers[:, 0] = np.clip(centers[:, 0], 0, H - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, W - 1)

    inp_coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=float)

    if torus:
        d2 = torus_d2(centers, inp_coords, H, W)
    else:
        d2 = ((centers[:, None, :] - inp_coords[None, :, :]) ** 2).sum(axis=2)

    if sigma_se_lognormal_std > 0.0:
        sigmas = rng.lognormal(
            mean=np.log(sigma),
            sigma=sigma_se_lognormal_std / sigma,
            size=(N_exc, 1),
        )
        sigmas = np.clip(sigmas, 0.5, None)
    else:
        sigmas = rng.normal(loc=sigma, scale=0.3, size=(N_exc, 1))
        sigmas = np.clip(sigmas, 1.0, None)
    G = np.exp(-d2 / (2 * sigmas**2))
    W_se = G.T
    W_se[W_se < 0.01] = 0.0
    W_se = weight_compliance(
        frac=fraction, N=N_exc, weights=W_se, peak=peak, type="W_se"
    )

    return W_se


def oriented_gaussian_se_weights(
    N_x: int,
    N_exc: int,
    input_size: int,
    sigma_x: float = 3.0,
    gamma: float = 0.4,
    n_orientations: int = 4,
    r_cut_factor: float = 3.0,
    sigma_x_lognormal_std: float = 0.0,
    sigma_x_lognormal_max: float = 0.0,
    orientation_mode: str = "block",
    peak: float = 1.0,
    fraction: float = 0.05,
    rng=None,
) -> np.ndarray:
    """Oriented elliptical Gaussian W_se, shape (N_x, N_exc).

    Each E neuron gets an elongated RF whose preferred orientation cycles
    through n_orientations evenly across the population (V1-like).
    sigma_y = gamma * sigma_x (short axis); hard cutoff at r_cut_factor * sigma_x.

    When sigma_x_lognormal_std > 0, per-neuron sigma_x is drawn from a
    log-normal distribution (mean=sigma_x), giving V1-like size diversity.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    W = np.zeros((N_exc, N_x), dtype=np.float32)

    n_side = int(np.ceil(np.sqrt(N_exc)))
    neuron_xs = np.linspace(0, input_size - 1, n_side)
    neuron_ys = np.linspace(0, input_size - 1, n_side)

    px, py = np.meshgrid(np.arange(input_size), np.arange(input_size), indexing="ij")
    px = px.ravel().astype(np.float32)
    py = py.ravel().astype(np.float32)

    orientations = [np.pi * k / n_orientations for k in range(n_orientations)]

    if sigma_x_lognormal_std > 0.0:
        sigma_x_arr = rng.lognormal(
            mean=np.log(sigma_x),
            sigma=sigma_x_lognormal_std / sigma_x,
            size=N_exc,
        ).astype(np.float32)
        upper = sigma_x_lognormal_max if sigma_x_lognormal_max > 0.0 else None
        sigma_x_arr = np.clip(sigma_x_arr, 0.5, upper)
    else:
        sigma_x_arr = np.full(N_exc, sigma_x, dtype=np.float32)

    for i in range(N_exc):
        gx = i // n_side
        gy = i % n_side
        if gx >= len(neuron_xs) or gy >= len(neuron_ys):
            continue

        cx = neuron_xs[gx]
        cy = neuron_ys[gy]
        if orientation_mode == "interleaved":
            theta = orientations[gy % n_orientations]
        else:
            block_size = max(1, n_side // n_orientations)
            theta = orientations[min(gx // block_size, n_orientations - 1)]
        sx = sigma_x_arr[i]
        sy = gamma * sx

        dx = px - cx
        dy = py - cy

        ct, st = np.cos(theta), np.sin(theta)
        x_t = dx * ct + dy * st
        y_t = -dx * st + dy * ct

        w = np.exp(-(x_t**2 / (2 * sx**2) + y_t**2 / (2 * sy**2)))
        # Elliptical cutoff in the rotated frame: cuts at r_cut_factor sigma
        # in each axis independently, matching the Gaussian footprint.
        # ellipse_dist = np.sqrt((x_t / sx) ** 2 + (y_t / sy) ** 2)
        ell = np.sqrt((x_t / sx) ** 2 + (y_t / sy) ** 2)
        w[ell > r_cut_factor] = 0.0
        s = w.sum()
        if s > 0:
            w = (w / s) * peak
        # w[w < 0.01] = 0.0
        W[i] = w

    # Column-wise compliance: scale each E neuron's total incoming weight to
    # fraction * N_x * peak, matching the drive level of the isotropic path.
    # Row-wise compliance (used by gaussian_se_weights) would destroy oriented
    # structure by massively inflating pixels that fall in few RFs.
    W = weight_compliance(
        frac=fraction, N=N_x, weights=W, peak=peak, type="W_se_oriented"
    )

    return W.T  # (N_x, N_exc) — matches gaussian_se_weights convention


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
    sigma_lognormal_std=0.0,
    rng=None,
):
    assert H_e * W_e == N_exc

    coords = np.array([(r, c) for r in range(H_e) for c in range(W_e)], dtype=float)

    if torus:
        d2 = torus_d2(coords, coords, H_e, W_e)
    else:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)

    if sigma_lognormal_std > 0.0 and rng is not None:
        sigmas = rng.lognormal(
            mean=np.log(sigma),
            sigma=sigma_lognormal_std / sigma,
            size=N_exc,
        )
        sigmas = np.clip(sigmas, 0.3, None)
    else:
        sigmas = np.full(N_exc, sigma)

    W = np.exp(-d2 / (2 * sigmas[:, None] ** 2))
    W /= W.max(axis=1, keepdims=True) + 1e-12
    W *= peak

    if self_zero:
        np.fill_diagonal(W, 0.0)

    W[W < 0.01] = 0.0
    W = weight_compliance(frac=frac, N=N_exc, weights=W, peak=peak, type="W_ee")

    return W
