from dataclasses import dataclass


@dataclass
class WeightsSpec:
    """Internal weight configuration. Use receptive_fields() or random() to construct."""

    density_se: float = 0.05
    density_ee: float = 0.01
    density_ei: float = 0.05
    density_ie: float = 0.05

    peak_se: float = 0.1
    peak_ee: float = 0.3
    peak_ei: float = 0.3
    peak_ie: float = -0.2

    rf_scale: float = 1.0

    _random: bool = False
    _oriented: bool = False

    sigma_x: float = 3.0
    gamma: float = 0.4
    n_orientations: int = 4
    r_cut_factor: float = 3.0
    sigma_x_lognormal_std: float = 0.0
    sigma_x_lognormal_max: float = 0.0   # 0 = no upper clip
    orientation_mode: str = "block"      # "block" or "interleaved"

    sigma_ee_mean: float = 0.0          # 0 = auto-compute from rf_scale
    sigma_ee_lognormal_std: float = 0.0  # 0 = disabled (fixed sigma_ee)
    sigma_se_mean: float = 0.0          # 0 = auto-compute from rf_scale (rf mode only)
    sigma_se_lognormal_std: float = 0.0  # 0 = disabled (fixed sigma_se, rf mode only)

    wta_inhibition: bool = False
    ablate_ee: bool = False             # zero E->E recurrence (causal collapse test)
    ablate_ie: bool = False             # zero I->E inhibition (causal collapse test)

    grouped_inhibition: bool = False    # block-diagonal W_ie by group
    n_groups: int = 0                   # 0 = auto (uses 10); ignored unless grouped_inhibition
    group_layout: str = "interleaved"   # "interleaved" | "block"
    tiled_centers: bool = False         # per-class tiled RF centers (block + full coverage)

    def _to_factory_kwargs(self) -> dict:
        return dict(
            w_dense_se=self.density_se,
            w_dense_ee=self.density_ee,
            w_dense_ei=self.density_ei,
            w_dense_ie=self.density_ie,
            se_weights=self.peak_se,
            ee_weights=self.peak_ee,
            ei_weights=self.peak_ei,
            ie_weights=self.peak_ie,
            rf_scale=self.rf_scale,
            random_weights=self._random,
            oriented_rf=self._oriented,
            sigma_x=self.sigma_x,
            gamma=self.gamma,
            n_orientations=self.n_orientations,
            r_cut_factor=self.r_cut_factor,
            sigma_x_lognormal_std=self.sigma_x_lognormal_std,
            sigma_x_lognormal_max=self.sigma_x_lognormal_max,
            orientation_mode=self.orientation_mode,
            sigma_ee_mean=self.sigma_ee_mean,
            sigma_ee_lognormal_std=self.sigma_ee_lognormal_std,
            sigma_se_mean=self.sigma_se_mean,
            sigma_se_lognormal_std=self.sigma_se_lognormal_std,
            wta_inhibition=self.wta_inhibition,
            ablate_ee=self.ablate_ee,
            ablate_ie=self.ablate_ie,
            grouped_inhibition=self.grouped_inhibition,
            n_groups=self.n_groups,
            group_layout=self.group_layout,
            tiled_centers=self.tiled_centers,
        )


def receptive_fields(
    density_se: float = 0.05,
    density_ee: float = 0.01,
    density_ei: float = 0.05,
    density_ie: float = 0.05,
    peak_se: float = 0.1,
    peak_ee: float = 0.3,
    peak_ei: float = 0.3,
    peak_ie: float = -0.2,
    rf_scale: float = 1.0,
    sigma_ee_mean: float = 0.0,
    sigma_ee_lognormal_std: float = 0.0,
    sigma_se_mean: float = 0.0,
    sigma_se_lognormal_std: float = 0.0,
    wta_inhibition: bool = False,
    ablate_ee: bool = False,
    ablate_ie: bool = False,
) -> WeightsSpec:
    """Gaussian / Mexican-hat structured receptive fields (topographic connectivity)."""
    return WeightsSpec(
        density_se=density_se,
        density_ee=density_ee,
        density_ei=density_ei,
        density_ie=density_ie,
        peak_se=peak_se,
        peak_ee=peak_ee,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
        rf_scale=rf_scale,
        _random=False,
        sigma_ee_mean=sigma_ee_mean,
        sigma_ee_lognormal_std=sigma_ee_lognormal_std,
        sigma_se_mean=sigma_se_mean,
        sigma_se_lognormal_std=sigma_se_lognormal_std,
        wta_inhibition=wta_inhibition,
        ablate_ee=ablate_ee,
        ablate_ie=ablate_ie,
    )


def oriented_receptive_fields(
    density_se: float = 0.05,
    density_ee: float = 0.01,
    density_ei: float = 0.05,
    density_ie: float = 0.05,
    peak_se: float = 0.1,
    peak_ee: float = 0.3,
    peak_ei: float = 0.3,
    peak_ie: float = -0.2,
    rf_scale: float = 1.0,
    sigma_x: float = 3.0,
    gamma: float = 0.4,
    n_orientations: int = 4,
    r_cut_factor: float = 3.0,
    sigma_x_lognormal_std: float = 0.0,
    sigma_x_lognormal_max: float = 0.0,
    orientation_mode: str = "block",
    sigma_ee_mean: float = 0.0,
    sigma_ee_lognormal_std: float = 0.0,
    wta_inhibition: bool = False,
    ablate_ee: bool = False,
    ablate_ie: bool = False,
) -> WeightsSpec:
    """Oriented elliptical Gaussian RFs for W_se; isotropic Gaussians for W_ee/W_ei/W_ie."""
    return WeightsSpec(
        density_se=density_se,
        density_ee=density_ee,
        density_ei=density_ei,
        density_ie=density_ie,
        peak_se=peak_se,
        peak_ee=peak_ee,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
        rf_scale=rf_scale,
        _random=False,
        _oriented=True,
        sigma_x=sigma_x,
        gamma=gamma,
        n_orientations=n_orientations,
        r_cut_factor=r_cut_factor,
        sigma_x_lognormal_std=sigma_x_lognormal_std,
        sigma_x_lognormal_max=sigma_x_lognormal_max,
        orientation_mode=orientation_mode,
        sigma_ee_mean=sigma_ee_mean,
        sigma_ee_lognormal_std=sigma_ee_lognormal_std,
        wta_inhibition=wta_inhibition,
        ablate_ee=ablate_ee,
        ablate_ie=ablate_ie,
    )


def grouped_excitatory(
    n_groups: int = 10,
    group_layout: str = "interleaved",
    tiled: bool = False,
    oriented: bool = True,
    density_se: float = 0.01,
    density_ee: float = 0.0,
    density_ei: float = 0.03,
    density_ie: float = 0.05,
    peak_se: float = 4.0,
    peak_ee: float = 1.0,
    peak_ei: float = 20.0,
    peak_ie: float = -2.0,
    n_orientations: int = 4,
    orientation_mode: str = "block",
    sigma_x: float = 3.0,
    gamma: float = 0.4,
    r_cut_factor: float = 3.0,
    ablate_ee: bool = False,
    ablate_ie: bool = False,
) -> WeightsSpec:
    """Grouped excitatory architecture: N_exc divided into n_groups class groups,
    each independently tiling the full input, with intra-group WTA inhibition
    (block-diagonal W_ie). Designed for use with reward-STDP.

    oriented=True  — elliptical Gabor-style RFs (default)
    oriented=False — isotropic 2D Gaussian RFs; orientation params are ignored
    """
    return WeightsSpec(
        density_se=density_se,
        density_ee=density_ee,
        density_ei=density_ei,
        density_ie=density_ie,
        peak_se=peak_se,
        peak_ee=peak_ee,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
        _random=False,
        _oriented=oriented,
        sigma_x=sigma_x,
        gamma=gamma,
        n_orientations=n_orientations,
        r_cut_factor=r_cut_factor,
        orientation_mode=orientation_mode,
        # When isotropic, use sigma_x as the Gaussian radius so RF size is
        # consistent between oriented and isotropic modes (auto-compute gives sigma=1px).
        sigma_se_mean=0.0 if oriented else sigma_x,
        wta_inhibition=True,
        grouped_inhibition=True,
        n_groups=n_groups,
        # tiled forces block layout (contiguous class blocks) + tiled RF centers
        group_layout="block" if tiled else group_layout,
        tiled_centers=tiled,
        ablate_ee=ablate_ee,
        ablate_ie=ablate_ie,
    )


def random(
    density_se: float = 0.05,
    density_ee: float = 0.01,
    density_ei: float = 0.05,
    density_ie: float = 0.05,
    peak_se: float = 0.1,
    peak_ee: float = 0.3,
    peak_ei: float = 0.3,
    peak_ie: float = -0.2,
    wta_inhibition: bool = False,
) -> WeightsSpec:
    """Uniformly random sparse connectivity."""
    return WeightsSpec(
        density_se=density_se,
        density_ee=density_ee,
        density_ei=density_ei,
        density_ie=density_ie,
        peak_se=peak_se,
        peak_ee=peak_ee,
        peak_ei=peak_ei,
        peak_ie=peak_ie,
        _random=True,
        wta_inhibition=wta_inhibition,
    )
