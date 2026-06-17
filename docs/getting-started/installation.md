# Installation

## From PyPI

```bash
pip install neurosnn
```

`neurosnn` requires **Python ≥ 3.12**. The core scientific dependencies — `numpy`, `numba`,
`scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `psutil`, and `pillow`
— are installed automatically.

## Data dependencies

Image datasets are loaded through **`torch`** / **`torchvision`** and downloaded on first
use. Depending on your platform you may want to install these explicitly first (for example
to select a CPU-only PyTorch build):

```bash
pip install torch torchvision
pip install neurosnn
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
