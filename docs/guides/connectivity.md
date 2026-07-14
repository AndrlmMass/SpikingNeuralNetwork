# Connectivity & weights

Connectivity is described by a **`WeightsSpec`**, built through one of three factory
functions in `snn.weights`. The spec sets, per pathway, how *dense* the connections are and
how *strong* they start.

## E/I populations and the four pathways

A network has three groups of neurons: **stimulus** inputs (count from
`Model(input_size=...)`), **excitatory** neurons (`Layer(N_exc=...)`), and **inhibitory**
neurons (`Layer(N_inh=...)`). Connections between them fall into four pathways:

| Pathway | Meaning | Sign | Plastic? |
|---------|---------|------|----------|
| **SE** | stimulus → excitatory | + | learned via STDP |
| **EE** | excitatory → excitatory | + | learned via STDP |
| **EI** | excitatory → inhibitory | + | fixed |
| **IE** | inhibitory → excitatory | − | fixed |

The SE and EE pathways carry the representation and are shaped by
[learning](learning-stdp.md). The EI/IE pathways implement fixed feedback inhibition.

## Shared parameters

Every factory accepts the same density and peak parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `density_se` | `0.05` | Connection probability, stimulus → excitatory |
| `density_ee` | `0.01` | Connection probability, excitatory → excitatory |
| `density_ei` | `0.05` | Connection probability, excitatory → inhibitory |
| `density_ie` | `0.05` | Connection probability, inhibitory → excitatory |
| `peak_se` | `0.1` | Peak initial weight, SE |
| `peak_ee` | `0.3` | Peak initial weight, EE |
| `peak_ei` | `0.3` | Peak initial weight, EI |
| `peak_ie` | `-0.2` | Peak initial weight, IE (negative — inhibitory) |

## The three factories

### `weights.random(...)`

Uniformly sparse random connectivity with no spatial structure. The simplest choice and a
good baseline.

```python
weights = snn.weights.random(density_se=0.05, peak_se=1.0)
```

### `weights.receptive_fields(...)`

Topographically structured connectivity: Gaussian receptive fields on the SE pathway and a
Mexican-hat (centre–surround) profile on EE. This is the default for `Layer`.

| Extra parameter | Default | Description |
|-----------------|---------|-------------|
| `rf_scale` | `1.0` | Global scale factor for all spatial sigmas |
| `sigma_ee_mean` | `0.0` | EE sigma; `0` = auto-compute from `rf_scale` |
| `sigma_ee_lognormal_std` | `0.0` | Log-normal spread of EE sigmas; `0` = fixed |
| `sigma_se_mean` | `0.0` | SE sigma; `0` = auto-compute from `rf_scale` |
| `sigma_se_lognormal_std` | `0.0` | Log-normal spread of SE sigmas; `0` = fixed |

### `weights.oriented_receptive_fields(...)`

Oriented elliptical Gaussian receptive fields on the SE pathway (V1 simple-cell-like),
isotropic Gaussians elsewhere.

| Extra parameter | Default | Description |
|-----------------|---------|-------------|
| `sigma_x` | `3.0` | Major-axis sigma of each RF (pixels) |
| `gamma` | `0.4` | Aspect ratio σ_y / σ_x; `1.0` = isotropic |
| `n_orientations` | `4` | Number of orientation groups cycled across E neurons |
| `r_cut_factor` | `3.0` | Hard elliptical cutoff at `r_cut_factor × σ` |
| `sigma_x_lognormal_std` | `0.0` | Log-normal RF-size diversity; `0` = uniform |
| `sigma_x_lognormal_max` | `0.0` | Upper clip on RF size; `0` = no clip |
| `orientation_mode` | `"block"` | `"block"` or `"interleaved"` assignment of orientations |

??? note "Receptive-field geometry"
    A receptive field assigns each connection a weight that falls off with distance from the
    field centre. For `receptive_fields`, SE uses an isotropic Gaussian and EE uses a
    difference-of-Gaussians (Mexican hat) giving local excitation and broader surround
    suppression. For `oriented_receptive_fields`, the SE Gaussian is elongated along an
    orientation axis with aspect ratio `gamma`, and successive excitatory neurons are
    assigned one of `n_orientations` angles — `"block"` assigns them in contiguous groups,
    `"interleaved"` cycles them neuron-by-neuron. Construction lives in
    `neurosnn/_network/init_weights.py`.

## See also

- [Learning (Trace STDP)](learning-stdp.md) — how SE/EE weights evolve during training.
- [Sleep & homeostasis](sleep-homeostasis.md) — how weights are periodically rescaled.
