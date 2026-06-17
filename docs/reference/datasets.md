# Supported datasets

Set the dataset with `Model(image_dataset="...")`. Standard image datasets are downloaded
automatically on first use via `torchvision`.

| Key | Dataset | Notes |
|-----|---------|-------|
| `"mnist"` | MNIST handwritten digits | 28×28; use `input_size=784` |
| `"kmnist"` | Kuzushiji-MNIST | Cursive Japanese characters |
| `"fmnist"` / `"fashionmnist"` / `"fashion"` | Fashion-MNIST | Clothing items |
| `"notmnist"` | notMNIST letters A–J | Requires the optional `deeplake` package |
| `"geomfig"` | Geometric figures | Toy dataset; fast for local testing |
| `"fcx1"` | Cortical spike recordings | Real pre-encoded spike data (100 input neurons) |

## Notes per dataset

- **MNIST / KMNIST / Fashion-MNIST** — loaded through `torchvision`, cached under
  `data/torchvision/`. Images are grayscaled, resized to `pixel_size × pixel_size`, and
  converted to Poisson spikes (see [Data & input encoding](../guides/data-encoding.md)).
- **notMNIST** — fetched via [Deeplake](https://github.com/activeloopai/deeplake). Install
  with `pip install deeplake`. If the source omits a train/test split, a seeded 90/10 split
  is created automatically.
- **geomfig** — a small synthetic set of geometric figures, handy for quick local runs and
  smoke tests without downloading anything large.
- **fcx1** — real cortical spike data supplied as pre-encoded spike arrays (loaded from
  `data/fcx1/`); the first 100 input channels are used. No Poisson encoding step is applied
  since the data is already spiking.

## See also

- [Data & input encoding](../guides/data-encoding.md) — how images become spike trains.
