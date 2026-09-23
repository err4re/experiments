# Experiments

Measurement-control and analysis code for a superconducting-circuits lab,
used interactively from Jupyter notebooks: build a `Config`, run it through
an `Experiment` (flux map or two-tone spectroscopy), get back an HDF5 file
and a live plot, then post-process with the tools in `analysis.py`.

See [`experiments_overview.md`](experiments_overview.md) for how the code is
structured, [`experiments_improvements.md`](experiments_improvements.md) for
a critical read of its current strengths/weaknesses, and
[`experiments_refactor.md`](experiments_refactor.md) for a concrete
step-by-step cleanup plan.

## Requirements

- Python >= 3.10 (`experiment_data.py` uses `@dataclass(kw_only=True)`).
- This lab's own **`instruments`** driver package (VNA/signal-generator/
  current-source drivers: https://github.com/err4re/instruments). It's not
  published on PyPI, so it has to be installed separately before this
  package will actually import.
- A machine that can open Qt windows (PyQt6) and, for the interactive
  analysis widgets, a running Jupyter kernel (they use IPython's
  `%matplotlib` magic and `get_ipython()`).

## Installation

```bash
# 1. install the instrument-driver package first
pip install -e git+https://github.com/err4re/instruments.git#egg=instruments

# 2. install this package
pip install -e .

# 3. for the interactive notebook workflow
pip install jupyterlab ipywidgets
```

For a known-working set of exact versions (captured from a working install),
use [`requirements-lock.txt`](requirements-lock.txt) instead of step 2:
`pip install -r requirements-lock.txt && pip install -e .`

## Quick usage

```python
from experiments.flux_map_experiment import FluxMapExperiment
from experiments.experiment_config import FluxMapConfig
from instruments.configs.yoko7651_config import YokoCurrSweepConfig
from instruments.configs.znb_config import ZnbLinConfig

config = FluxMapConfig(yoko=YokoCurrSweepConfig(...), vna=ZnbLinConfig(...))
exp = FluxMapExperiment(sample_name="Q1", sample_code="Q1", file_directory="./data", config=config)
exp.flux_map(live_plotting=True)   # runs, live-plots, saves an HDF5 file on completion
exp.plot_results()                  # final annotated figure
```

## Status

No `LICENSE` file is checked in yet, so `setup.py` doesn't claim one either
-- add one (and a matching classifier) once the license is decided.
