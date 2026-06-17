# Data & input encoding

`neurosnn` turns images into **Poisson spike trains** that drive the stimulus neurons.
Dataset choice and encoding parameters are set on the `Model`.

## What it does

Images are loaded (and cached in RAM), optionally Gabor-filtered, then converted to spikes:
each pixel's intensity sets a per-timestep firing probability, and an independent random
draw per pixel per timestep produces the spikes. Each image is presented for `num_steps`
timesteps.

```python
import neurosnn as snn

model = snn.Model(
    image_dataset="mnist",
    input_size=784,        # flattened pixels
    num_steps=350,         # ms each image is shown
    max_rate_hz=90.0,      # peak Poisson rate at full intensity
    gabor=False,           # set True to apply oriented Gabor filtering
)
```

## Encoding parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_dataset` | `"mnist"` | Dataset key (see [Supported datasets](../reference/datasets.md)) |
| `input_size` | `225` | Flattened input pixels (`784` for 28×28 MNIST) |
| `num_steps` | `350` | Timesteps each image is presented (ms) |
| `max_rate_hz` | `90.0` | Peak Poisson firing rate at full pixel intensity |
| `gain` | `1.0` | Global multiplicative gain on inputs |
| `gabor` | `False` | Apply oriented Gabor filtering before encoding |

## Poisson rate coding

A pixel of intensity `I ∈ [0, 1]` produces a spike each timestep with probability
proportional to `I × max_rate_hz × dt`. Higher intensity → higher firing rate. The draw is
independent per pixel and per timestep, which approximates a Poisson process.

## Gabor filtering

With `gabor=True`, each image is convolved with four oriented Gabor filters (0°, 90°, 45°,
135°) before encoding, and the four responses are packed into the quadrants of a same-size
image. This injects orientation-selective edge structure reminiscent of V1 simple cells
while preserving the input dimensionality.

??? note "The encoding maths and pipeline"
    Spike probability per timestep is

    $$
    p = \text{clip}\bigl(I \cdot r_\text{max} \cdot \tfrac{dt}{1000},\ 0,\ 1\bigr)
    $$

    with `dt = 1 ms`, so at full intensity and `max_rate_hz = 90` the per-step spike
    probability is `0.09`. A spike occurs where `rand() < p`. The batch is reshaped from
    `(T, B, N_pixels)` to `(B·T, N_pixels)` so the simulation receives one row per timestep.
    The loaders and encoder live in `neurosnn/_data/get_data.py` (`ImageDataStreamer`,
    `_convert_images_to_spikes`, `gabor_pack_quadrants`).

## See also

- [Supported datasets](../reference/datasets.md) — the full list of dataset keys.
- [Neuron dynamics](neuron-dynamics.md) — how the resulting spikes drive membrane potentials.
