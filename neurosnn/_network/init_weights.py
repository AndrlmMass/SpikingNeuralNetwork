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
    wta_inhibition: bool = False
    ablate_ee: bool = False             # zero the recurrent E->E block (causal test)
    ablate_ie: bool = False             # zero the I->E inhibition block (causal test)

    grouped_inhibition: bool = False    # block-diagonal W_ie by group
    n_groups: int = 0                   # 0 = auto (10); ignored unless grouped_inhibition
    group_layout: str = "interleaved"   # "interleaved" | "block"
    tiled_centers: bool = False         # per-class tiled RF centers (block layout, full-input
                                        # coverage per class); allows non-square N_exc = n_groups*k^2

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
        # Causal-ablation switches: zero a recurrent block *after* construction.
        # Because the plastic synapse set (nonzero_pre_idx) is read from this
        # final matrix, a zeroed block carries no current AND is excluded from
        # STDP — so it stays ablated for the whole run, isolating its effect on
        # RF collapse.
        if self.ablate_ee:
            self.weights[self.st : self.ex, self.st : self.ex] = 0.0
        if self.ablate_ie:
            self.weights[self.ex : self.ih, self.st : self.ex] = 0.0
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
        result = dict(
            nz_rows_se=nz_rows_se,
            nz_cols_se=nz_cols_se,
            nz_rows_ee=nz_rows_ee,
            nz_cols_ee=nz_cols_ee,
            nz_rows_exc=nz_rows_exc,
            nz_cols_exc=nz_cols_exc,
            nonzero_pre_idx=nonzero_pre_idx,
        )
        if self.grouped_inhibition and hasattr(self, "_group_assignment"):
            ie_block = weights[self.ex : self.ih, self.st : self.ex]
            result["ie_struct_mask"] = ie_block != 0.0
            result["group_assignment"] = self._group_assignment.copy()
        else:
            result["ie_struct_mask"] = None
            result["group_assignment"] = None
        return result

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
        if self.tiled_centers:
            self._fill_grouped_tiled()
            return
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

        if self.wta_inhibition:
            W_ei = wta_ei_weights(self.N_exc, self.N_inh, peak=self.ei_weights)
            self.weights[self.st : self.ex, self.ex : self.ih] = W_ei
            if self.grouped_inhibition:
                n_grp = self.n_groups if self.n_groups > 0 else 10
                self._group_assignment = make_group_assignment(self.N_exc, n_grp, self.group_layout)
                W_ie = grouped_wta_ie_weights(self.N_exc, self._group_assignment, peak=self.ie_weights)
            else:
                W_ie = wta_ie_weights(
                    self.N_inh, self.N_exc, peak=self.ie_weights, frac=self.w_dense_ie
                )
            self.weights[self.ex : self.ih, self.st : self.ex] = W_ie
        else:
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

    def _fill_grouped_tiled(self):
        """Grouped-tiled SE: each class = a contiguous block whose RF centers tile the
        full input on a regular torus grid (block layout + full coverage). Feedforward
        only (W_ee = 0), 1:1 WTA hitman + intra-group (block) I->E inhibition. Works for
        non-square N_exc = n_groups * k^2 (bypasses the sheet-grid square asserts)."""
        n_grp = self.n_groups if self.n_groups > 0 else 10
        centers = group_tiled_centers(self.N_exc, n_grp, self.H)  # (N_exc, 2) input coords

        if self.oriented_rf:
            W_se = oriented_gaussian_se_weights(
                N_x=self.N_x, N_exc=self.N_exc, input_size=self.H,
                sigma_x=self.sigma_x, gamma=self.gamma, n_orientations=self.n_orientations,
                r_cut_factor=self.r_cut_factor, orientation_mode=self.orientation_mode,
                peak=self.se_weights, fraction=self.w_dense_se, rng=self.rng,
                centers=centers,
            )
        else:
            sigma_se = (
                self.sigma_se_mean if self.sigma_se_mean > 0.0
                else self.rf_scale * self._fse * self.ref_x
            )
            W_se = gaussian_se_weights(
                self.N_x, self.N_exc, self.H, self.W, self.H_e, self.W_e, rng=self.rng,
                sigma=sigma_se, peak=self.se_weights, fraction=self.w_dense_se, torus=True,
                jitter=0.0, sigma_se_lognormal_std=self.sigma_se_lognormal_std,
                centers=centers,
            )
        self.weights[: self.st, self.st : self.ex] = W_se

        # W_ee stays zero (feedforward reward architecture): skip gaussian_ee_weights,
        # which asserts a square exc sheet and would crash at non-square N_exc.

        # Inhibition: 1:1 WTA hitman (E->I identity) + intra-group block I->E.
        self._group_assignment = np.repeat(np.arange(n_grp), self.N_exc // n_grp)
        self.weights[self.st : self.ex, self.ex : self.ih] = wta_ei_weights(
            self.N_exc, self.N_inh, peak=self.ei_weights
        )
        self.weights[self.ex : self.ih, self.st : self.ex] = grouped_wta_ie_weights(
            self.N_exc, self._group_assignment, peak=self.ie_weights
        )

    def _fill_random_weights(self):
        mask_ee = self.rng.random((self.N_exc, self.N_exc)) < self.w_dense_ee
        mask_se = self.rng.random((self.N_x, self.N_exc)) < self.w_dense_se

        self.weights[: self.st, self.st : self.ex][mask_se] = self.se_weights
        self.weights[self.st : self.ex, self.st : self.ex][mask_ee] = self.ee_weights
        np.fill_diagonal(self.weights[self.st : self.ex, self.st : self.ex], 0)

        if self.wta_inhibition:
            W_ei = wta_ei_weights(self.N_exc, self.N_inh, peak=self.ei_weights)
            self.weights[self.st : self.ex, self.ex : self.ih] = W_ei
            if self.grouped_inhibition:
                n_grp = self.n_groups if self.n_groups > 0 else 10
                self._group_assignment = make_group_assignment(self.N_exc, n_grp, self.group_layout)
                W_ie = grouped_wta_ie_weights(self.N_exc, self._group_assignment, peak=self.ie_weights)
            else:
                W_ie = wta_ie_weights(
                    self.N_inh, self.N_exc, peak=self.ie_weights, frac=self.w_dense_ie
                )
            self.weights[self.ex : self.ih, self.st : self.ex] = W_ie
        else:
            mask_ei = self.rng.random((self.N_exc, self.N_inh)) < self.w_dense_ei
            mask_ie = self.rng.random((self.N_inh, self.N_exc)) < self.w_dense_ie
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


def wta_ei_weights(N_exc, N_inh, peak=1.0):
    """WTA E→I, one-to-one: E neuron e drives its own dedicated interneuron e (an
    inhibitory 'hitman'). Requires N_inh == N_exc so every E cell has exactly one
    private interneuron.

    The old round-robin (``e % N_inh``) mapped ~N_exc/N_inh excitatory cells onto
    each interneuron; combined with a global I→E projection that degenerated into
    plain uniform inhibition with no per-neuron competition. One-to-one restores a
    genuine winner-take-all: only the interneurons of E cells that fired become
    active.
    """
    if N_inh != N_exc:
        raise ValueError(
            f"one-to-one WTA requires N_inh == N_exc (got N_inh={N_inh}, "
            f"N_exc={N_exc}). Set the layer's N_inh equal to N_exc for WTA."
        )
    return np.eye(N_exc, dtype=float) * peak


def wta_ie_weights(N_inh, N_exc, peak=-0.2, frac=None):
    """WTA I→E: interneuron e inhibits every OTHER E cell (global minus self), so a
    winning E's hitman suppresses all its competitors but never the winner itself
    (zero diagonal).

    If ``frac`` is given, each interneuron's total outgoing inhibition is scaled by
    ``weight_compliance`` to ``frac * N_exc * peak`` — i.e. weak per-synapse
    (≈ peak·frac each) but broadcast to all competitors. Requires N_inh == N_exc.
    """
    if N_inh != N_exc:
        raise ValueError(
            f"one-to-one WTA requires N_inh == N_exc (got N_inh={N_inh}, "
            f"N_exc={N_exc}). Set the layer's N_inh equal to N_exc for WTA."
        )
    W_ie = np.full((N_inh, N_exc), peak, dtype=float)
    np.fill_diagonal(W_ie, 0.0)  # the hitman spares its own driver E cell
    if frac is not None:
        W_ie = weight_compliance(
            frac=frac, N=N_exc, weights=W_ie, peak=peak, type="W_ie_wta"
        )
    return W_ie


def make_group_assignment(N_exc: int, n_groups: int, layout: str = "interleaved") -> np.ndarray:
    """Return (N_exc,) int array: group index in [0, n_groups) per excitatory neuron.

    interleaved: group[i] = i % n_groups
        Stride-n_groups sampling of the 2D excitatory grid visits every region,
        so each group naturally tiles the full input with the existing retinotopic mapping.
    block: group[i] = chunk index
        Contiguous blocks; each block covers only a sub-region of the grid unless
        RF centers are explicitly remapped (not done here; interleaved is preferred).
    """
    if layout == "interleaved":
        return np.arange(N_exc, dtype=int) % n_groups
    sizes = [N_exc // n_groups + (1 if i < N_exc % n_groups else 0) for i in range(n_groups)]
    a = np.empty(N_exc, dtype=int)
    start = 0
    for g, sz in enumerate(sizes):
        a[start : start + sz] = g
        start += sz
    return a


def group_tiled_centers(N_exc: int, n_groups: int, input_size: int) -> np.ndarray:
    """Per-class-group RF centers that tile the full input on a regular torus grid.

    Each group gets g = N_exc // n_groups neurons whose centers form a k x k grid
    over the input, with the SAME grid for every group, so every class tiles the
    whole image. Centers use offset spacing ((i+0.5)*input/k), not linspace
    endpoints, so the torus wrap spacing is uniform — this maximizes the *minimum*
    inter-center spacing (even coverage), rather than maximizing total pairwise
    distance (which would clump centers at the edges).

    Returned in block order (neuron n -> group n // g), matching the block
    assignment np.repeat(arange(n_groups), g). Shape (N_exc, 2), (row, col) coords.
    """
    if N_exc % n_groups != 0:
        raise ValueError(f"N_exc ({N_exc}) must be divisible by n_groups ({n_groups})")
    g = N_exc // n_groups
    k = int(round(np.sqrt(g)))
    if k * k != g:
        raise ValueError(
            f"neurons per group ({g}) must be a perfect square for a k x k tile; "
            f"got sqrt={np.sqrt(g):.3f}. Try N_exc = n_groups * k^2 (e.g. 10*100=1000)."
        )
    step = input_size / k
    coords = (np.arange(k) + 0.5) * step
    grid = np.array([(r, c) for r in coords for c in coords], dtype=float)  # (g, 2)
    return np.tile(grid, (n_groups, 1))  # (N_exc, 2) in block order


def grouped_wta_ie_weights(N_exc: int, group_assignment: np.ndarray, peak: float = -2.0) -> np.ndarray:
    """Block-diagonal W_ie for intra-group WTA inhibition.

    Hitman i suppresses only the neurons in the same group as i, excluding itself.
    N_inh == N_exc (enforced by wta_ei_weights upstream).

    weight_compliance is intentionally NOT applied: cross-group sparsity already
    reduces total inhibitory drive by ~(group_size / N_exc) vs global WTA.
    Rescaling would multiply per-synapse weight by ~n_groups, catastrophically
    over-inhibiting all groups at initialization.
    """
    W_ie = np.zeros((N_exc, N_exc), dtype=float)
    for i in range(N_exc):
        same_group = group_assignment == group_assignment[i]
        same_group[i] = False
        W_ie[i, same_group] = peak
    return W_ie


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
    centers=None,
):
    assert H * W == N_x

    # centers=None: derive from the retinotopic sheet grid (requires N_exc square).
    # centers given (N_exc, 2): use them directly (tiled / grouped path) and skip the
    # square assumption entirely.
    if centers is None:
        assert H_e * W_e == N_exc
        coords_e = exc_coords(H_e, W_e)
        centers = np.empty_like(coords_e)
        centers[:, 0] = coords_e[:, 0] * (H - 1) / max(H_e - 1, 1)
        centers[:, 1] = coords_e[:, 1] * (W - 1) / max(W_e - 1, 1)
    else:
        centers = np.array(centers, dtype=float)  # copy: never mutate caller's array
        assert centers.shape == (N_exc, 2), f"centers must be ({N_exc}, 2), got {centers.shape}"

    if jitter is not None and jitter > 0:
        centers = centers + rng.uniform(-jitter, jitter, size=centers.shape)
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


def oriented_rf_assignment(N_exc, n_orientations, input_size, orientation_mode="block"):
    """Map each excitatory neuron to a RF center + orientation from its position
    on the H_e x W_e sheet (row-major: row = i // W_e, col = i % W_e).

    Single source of truth for the index -> (center, orientation) mapping, shared
    by ``oriented_gaussian_se_weights`` and the RF diagnostic plots so they can
    never drift apart.

    Both modes keep every orientation tiling the FULL image (V1 intent); they
    differ only in how sheet-neighbors relate, which is what the torus-distance
    W_ee / W_ei / W_ie builders assume about adjacency:

      interleaved : sqrt(n_or) x sqrt(n_or) hypercolumn patches. Neighbors inside
                    a patch share a location and span all orientations -> learns
                    multi-orientation co-activation.
      block       : n_or quadrants (grid_shape(n_or)). Neighbors inside a quadrant
                    share an orientation and tile the image retinotopically ->
                    learns oriented-contour continuity.

    Returns
    -------
    centers : (N_exc, 2) float32 array of (cx, cy) image-space RF centers.
    orientation_idx : (N_exc,) int64 array in [0, n_orientations).
    """
    H_e = W_e = int(np.sqrt(N_exc))
    assert H_e * W_e == N_exc, "N_exc must be a perfect square"
    i = np.arange(N_exc)
    r, c = i // W_e, i % W_e

    if orientation_mode == "interleaved":
        p = int(round(np.sqrt(n_orientations)))
        if p * p != n_orientations:
            raise ValueError(
                "interleaved layout needs a perfect-square n_orientations, "
                f"got {n_orientations}"
            )
        if H_e % p != 0 or W_e % p != 0:
            raise ValueError(
                f"sheet {H_e}x{W_e} not divisible by patch side {p} "
                f"(n_orientations={n_orientations})"
            )
        Sr, Sc = H_e // p, W_e // p  # spatial-location grid (one neuron/ori per slot)
        xs = np.linspace(0, input_size - 1, Sr)
        ys = np.linspace(0, input_size - 1, Sc)
        orientation_idx = (r % p) * p + (c % p)
        centers = np.stack([xs[r // p], ys[c // p]], axis=1)
    else:  # block = quadrants
        qr, qc = grid_shape(n_orientations)
        if H_e % qr != 0 or W_e % qc != 0:
            raise ValueError(
                f"sheet {H_e}x{W_e} not divisible by quadrant grid {qr}x{qc} "
                f"(n_orientations={n_orientations})"
            )
        Qh, Qw = H_e // qr, W_e // qc  # neurons per quadrant == per-ori center grid
        xs = np.linspace(0, input_size - 1, Qh)
        ys = np.linspace(0, input_size - 1, Qw)
        orientation_idx = (r // Qh) * qc + (c // Qw)
        centers = np.stack([xs[r % Qh], ys[c % Qw]], axis=1)

    return centers.astype(np.float32), orientation_idx.astype(np.int64)


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
    centers=None,
    orientation_idx=None,
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

    # Guard against sigma_x <= 0 (e.g. a 0.0 CLI default flowing through):
    # mean=np.log(sigma_x) would be -inf and the lognormal shape param below
    # would divide by zero, producing NaN RFs. Fall back to the nominal size.
    if sigma_x <= 0.0:
        sigma_x = 3.0

    W = np.zeros((N_exc, N_x), dtype=np.float32)

    # RF centre + orientation per neuron come from the shared sheet-position
    # assignment, so the RF geometry and the W_ee/W_ei/W_ie sheet adjacency are
    # consistent (sheet-neighbors code related features). See
    # oriented_rf_assignment for the block/interleaved layout semantics.
    # centers/orientation_idx=None: derive from the retinotopic sheet (requires N_exc
    # square). Provided (tiled path): use them and skip oriented_rf_assignment's assert.
    if centers is None:
        centers, orientation_idx = oriented_rf_assignment(
            N_exc, n_orientations, input_size, orientation_mode
        )
    else:
        centers = np.asarray(centers, dtype=np.float32)
        assert centers.shape == (N_exc, 2), f"centers must be ({N_exc}, 2), got {centers.shape}"
        if orientation_idx is None:
            orientation_idx = np.arange(N_exc) % n_orientations
        orientation_idx = np.asarray(orientation_idx).astype(np.int64)

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
        # Clip RF sizes to a sane range around sigma_x: a lower bound (hard
        # 1 px floor, since sub-pixel RFs collapse to dead single-pixel
        # neurons) and an upper cap (default 3x sigma_x) to tame the heavy
        # lognormal tail that would otherwise yield a few image-spanning,
        # non-selective RFs. sigma_x_lognormal_max overrides the upper cap.
        lower = max(0.75 * sigma_x, 1.0)
        upper = (
            sigma_x_lognormal_max if sigma_x_lognormal_max > 0.0 else 3.0 * sigma_x
        )
        sigma_x_arr = np.clip(sigma_x_arr, lower, upper)
    else:
        sigma_x_arr = np.full(N_exc, sigma_x, dtype=np.float32)

    for i in range(N_exc):
        # Center + orientation are read from the shared sheet-position assignment;
        # the modes differ only in how (location, orientation) map onto the sheet
        # (see oriented_rf_assignment).
        cx, cy = centers[i]
        theta = orientations[orientation_idx[i]]
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
        # Floor must keep the nearest torus neighbour (d=1) above the 0.01 prune
        # below: exp(-1/(2 sigma^2)) > 0.01 needs sigma > 0.33, else the whole
        # off-diagonal row is pruned and that E neuron gets zero outgoing EE
        # weight (isolated dead neuron). 0.5 leaves a safe margin (nn ~ 0.135).
        sigmas = np.clip(sigmas, 0.5, None)
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
