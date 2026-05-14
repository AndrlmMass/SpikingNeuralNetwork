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
    )
