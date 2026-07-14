# Installation

## Clone and install

`neurosnn` is not yet published to PyPI. Install it directly from the repository:

```bash
git clone https://github.com/AndrlmMass/SpikingNeuralNetwork.git
cd SpikingNeuralNetwork
pip install -e .
```

`neurosnn` requires **Python ≥ 3.12**. The core scientific dependencies — `numpy`, `numba`,
`scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `psutil`, and `pillow`
— are declared in `pyproject.toml` and installed automatically.

## Data dependencies

Image datasets are loaded through **`torch`** / **`torchvision`** and downloaded on first
use. These are not declared as hard dependencies because the right PyTorch build varies by
platform. Install them explicitly before running:

```bash
pip install torch torchvision
```

The `"notmnist"` dataset additionally requires the optional `deeplake` package:

```bash
pip install deeplake
```

## Documentation tooling (optional)

To build this documentation site locally, install the `docs` extra:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open <http://127.0.0.1:8000>. This installs `mkdocs-material` and `pymdown-extensions`.

## Verifying the install

```python
import neurosnn as snn
print(snn.__all__)
# ['Model', 'Layer', 'TrainResult', 'EvalResult', 'membrane', 'weights', 'learner', 'regularizer']
```

If that prints without error, you are ready for the [Quickstart](quickstart.md).
