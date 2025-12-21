# Import Best Practices

## Overview

This project uses **absolute imports with the `src.` prefix** for cross-package imports. This ensures imports work consistently whether you run scripts from the project root or from within the `src` directory.

## Import Rules

### ✅ DO: Use absolute imports for cross-package imports
```python
# In src/models/SNN_sleepy/snn.py
from src.datasets.load import DataStreamer
from src.plot.plot import spike_plot
from src.evaluation.classifiers import pca_logistic_regression
```

### ✅ DO: Use relative imports within the same package
```python
# In src/models/SNN_sleepy/snn.py (within SNN_sleepy package)
from .train import train_network
from .dynamics import create_learning_bounds
from .layers import create_weights
```

### ❌ DON'T: Use bare imports without src prefix
```python
# WRONG - won't work when run from project root
from datasets.load import DataStreamer
from plot.plot import spike_plot
```

### ❌ DON'T: Use relative imports for cross-package imports
```python
# WRONG - evaluation is not a subpackage of SNN_sleepy
from .evaluation.classifiers import pca_logistic_regression
```

## Setup Options

### Option 1: Install in Development Mode (Recommended)

Install the package in development mode so Python can find it from anywhere:

```bash
pip install -e .
```

This installs the package in "editable" mode, so imports work regardless of where you run scripts from.

### Option 2: Add Project Root to PYTHONPATH

When running scripts, add the project root to PYTHONPATH:

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "C:\Users\Andreas\Documents\Github\SpikingNeuralNetwork"
python test_geomfig.py
```

**Linux/Mac:**
```bash
export PYTHONPATH=/path/to/SpikingNeuralNetwork:$PYTHONPATH
python test_geomfig.py
```

### Option 3: Add to sys.path in Scripts

For standalone scripts, add the project root to `sys.path`:

```python
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import PAPER_GEOMFIG_EXPERIMENT
```

## Current Import Structure

```
src/
├── models/
│   └── SNN_sleepy/
│       ├── snn.py          # Uses: from src.datasets..., from src.plot...
│       ├── train.py        # Uses: from .plasticity... (relative)
│       └── plasticity.py
├── datasets/
│   ├── load.py             # Uses: from src.datasets.datasets...
│   └── datasets.py
├── plot/
│   └── plot.py
└── evaluation/
    └── classifiers.py
```

## Testing

After fixing imports, test with:

```bash
# From project root
python test_geomfig.py --quick

# Or after pip install -e .
python -m src.experiments.mnist_family --snn-sleepy-only
```

