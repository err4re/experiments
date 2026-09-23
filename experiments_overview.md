# Experiments — Codebase Overview

This repository is the measurement-control layer for a superconducting-circuits
lab. It runs on top of a separate, external `instruments` package (VNA, signal
generator, current source drivers — not part of this repo) and is used
interactively from Jupyter notebooks: a physicist builds a `*Config`, hands it
to an `*Experiment`, calls a measurement method, and gets back an HDF5 file
plus a live-updating plot.

There is no test suite, no CI, and no `requirements.txt`. The code is meant to
be read top to bottom in a notebook session, not deployed as a service.

## 1. Big picture

```
Config dataclass  ──┐
                     ├──>  Experiment subclass  ──>  ExperimentData dataclass ──> HDF5 file
Instrument drivers ──┘             │
                                    └──> Plotter (live view + final figure)
```

- **Config** (`experiment_config.py`) — plain-data description of how to drive
  each instrument for one run (frequency ranges, powers, sweep points, bias
  currents/voltages...). Built once per measurement, usually in the notebook.
- **Experiment** (`experiment.py`, `flux_map_experiment.py`,
  `two_tone_experiment.py`) — owns the instrument handles, runs the sweep
  described by the config, and fills in a Data object as it goes.
- **ExperimentData** (`experiment_data.py`) — plain-data container for
  everything measured plus metadata (start/end time, state). This is exactly
  what gets serialized to disk.
- **Plotter** (`plotter.py`) — turns Data into matplotlib figures, both live
  (during acquisition, via `IPython.display`) and as a final annotated plot.
- **analysis.py** — offline, interactive post-processing of saved
  `FluxMapData`: converting flux-map voltage/current axes into flux quanta,
  fitting resonator dips, measuring level spacing — using Qt slider dialogs
  layered on top of the plots.

## 2. Base classes (`experiment.py`, `experiment_config.py`, `experiment_data.py`)

### `Experiment` (dataclass, `experiment.py`)

Holds `sample_name`, `sample_code`, `measurement_code`, `config`,
`file_directory`/`file_name`/`file_path`, the instrument handles
(`vna`, `ana`, `yoko`), the `data` object, and a `plotter`.

Key pieces:

- **`experiment_method` decorator** — wraps a measurement method so every
  concrete measurement automatically gets: `start_experiment()` (records
  start time, sets state to `RUNNING`, initializes instruments) before the
  method body, and `end_experiment()` (records end time/state, releases
  instruments, closes plots, saves to HDF5) after — including on
  `KeyboardInterrupt`, which is caught, marks the run `INTERRUPTED`, and
  re-raised after cleanup. This is the main place where "boilerplate" for a
  measurement lives, so subclasses' measurement methods can just describe the
  physics loop.
- **`safe_initializer` decorator** — wraps `initialize_vna/ana/yoko`; if an
  instrument fails to connect it prints a warning and leaves the attribute
  `None` instead of crashing the whole experiment. Useful when developing
  away from the lab with only some instruments present.
- **HDF5 persistence** (`save_dataclass_to_hdf5` / `load_dataclass_from_hdf5`)
  — generic: it walks `dataclasses.fields()` of whatever `ExperimentData`
  subclass is attached, and stores each field either as an HDF5 dataset
  (numpy arrays), a group (dicts, recursively), or an attribute (scalars,
  `datetime`, the `State` enum). Loading looks up the dataclass type by name
  from `experiment_data` and reconstructs it the same way. Any new
  `ExperimentData` subclass gets save/load for free as long as its fields are
  arrays/dicts/scalars.
- **`save_data_npz`** — older, simpler saving path via `data_dict` (a plain
  dict); kept for backward compatibility, HDF5 is the current path.

### Config classes (`experiment_config.py`)

`ExperimentConfig` is an empty marker base. `FluxMapConfig` and
`TwoToneConfig` are dataclasses that just bundle instrument-specific config
objects imported from the external `instruments` package (`ZnbLinConfig`,
`YokoCurrSweepConfig`, `AnaFreqSweepConfig`, ...). `__post_init__` does simple
required-field validation (raise `ValueError` if a sub-config wasn't
supplied).

### Data classes (`experiment_data.py`)

`ExperimentData` is the base (start/end time, `State` enum, measurement
function name). `FluxMapData` and `TwoToneData` add the numpy arrays specific
to each experiment (frequencies, S-parameters, bias values, metadata dicts).
All fields default to `None`/`np.nan`-filled arrays and are populated by the
`Experiment` subclass as the sweep runs.

## 3. `FluxMapExperiment` (`flux_map_experiment.py`)

Sweeps a VNA frequency window while stepping a bias (current or voltage) on
the Yoko source, to map how a resonator frequency shifts with flux. Public
measurement methods (all decorated with `@Experiment.experiment_method`):

- `flux_map` — bias = current, one VNA trace per bias point.
- `flux_map_voltage_sweep` — same, but bias = voltage.
- `flux_map_constant_voltage` — voltage held fixed, still loops the "sweep"
  (essentially a repeated-trace/averaging mode; note it never changes the
  bias inside the loop).
- `power_sweep` — bias = current outer loop, VNA power inner loop.

All four repeat the same shape: ramp the source to its starting value, turn
it on, configure the VNA, loop over bias points setting the value, sleeping,
sweeping, storing into `self.data`, and optionally live-plotting via
`Plotter.update_full_imshow`.

At the bottom of the file, `flux_map_power_dependence`,
`initialize_flux_map_adaptive_data`, and `flux_map_adaptive` are legacy /
unfinished code: `flux_map_power_dependence` calls `self.instruments[...]`
and `self.configs_dict[...]`, an older dict-based API that no longer exists
on `Experiment`, so it cannot run; `flux_map_adaptive` simply
`raise NotImplementedError`.

## 4. `TwoToneExperiment` (`two_tone_experiment.py`, ~3100 lines)

Two-tone spectroscopy: the VNA probes a fixed resonator tone (`f1`) while a
second generator (Anapico, `f2`) sweeps looking for a qubit transition that
shifts the resonator response. This is by far the largest and most
repetitive file in the repo (see `experiments_improvements.md` for the
duplication analysis) — the summary here groups methods by role rather than
listing all of them.

**Setup / calibration helpers**

- `initialize_data` — allocates the `TwoToneData` arrays based on whichever
  axis is being swept (current, voltage, or a power sweep) and whether
  flux-tracking is configured.
- `initialize_f1_to_f2_tracking` — pulls tracking parameters (spline fits for
  upper/lower resonator branches, qubit `Ec`/`Ej`) out of
  `config.tracking_parameters` for use by `f1_to_f2`/`f1_to_flux` later.
- `calibrate_f1`, `calibrate_f1_z`, `calibrate_f1_z_short`,
  `calibrate_f1_z_short_masked`, `calibrate_f1_masked`, `calibrate_f1_z_masked`
  — six variants of "take a VNA trace, find the resonance dip, zoom in and
  re-measure to get a precise center frequency," differing only in whether
  they mask out spurious device frequencies and what they return. These are
  near-duplicates of each other (see improvements doc).
- `mask_devices` — boolean mask excluding known spurious frequency bands
  (duplicates `utils.generate_mask`).
- `calibrate_f1_and_flux_map` — rough+fine VNA sweep to re-derive the flux
  map around the current bias point, for keeping the probe tone on resonance
  as flux drifts.

**Signal processing helpers**

- `phase_shift_signal` / `unwrapped_phase_shift_signal` / `full_S_parameter_signal`
  — convert a raw `S(f2)` trace and a reference trace into the scalar
  "signal" plotted against `f2` (unwraps phase, subtracts reference, scales).
- `f2_to_power`, `f2_to_span`, `f2_power_compensate_filter` — optional
  frequency-dependent power/span shaping for the second tone, used when
  `config.f2_to_power` / `config.f2_to_span` are set.
- `f1_to_f2`, `f1_to_flux`, `f1_to_flux_spline`, `f2_tracking` — convert
  between resonator frequency, flux and qubit frequency using the spline fits
  from `initialize_f1_to_f2_tracking`, to keep `f2`'s search window centered
  on the expected qubit frequency as the bias sweeps.
- `find_parabola_vertex_form` — 3-point parabola fit, used by the
  "tracking_parabola" measurement variants to re-center the next point's
  search window on the qubit frequency inferred from the last 3 points.

**Measurement methods** (all `@Experiment.experiment_method`)

These form a grid across a few independent axes, and the file contains one
method per combination actually needed historically, rather than one method
parameterized by the axes:

| axis | values found in the method names |
|---|---|
| swept bias | `_current` (implicit/default) vs `_voltage` |
| acquisition mode | single trace, `_average`, `_cw` (continuous-wave/single-shot style), `_single_shot`, `_movie`, `_tracking_parabola` |
| what's swept besides bias | `_f2_power_sweep`, `_f1_power_sweep` |

Representative members: `two_tone_optimized`, `two_tone_optimized_average`,
`two_tone_optimized_average_current`, `two_tone_optimized_average_voltage`,
`two_tone_f2_power_sweep`, `two_tone_cw_f2_power_sweep`,
`two_tone_cw_f2_power_sweep_voltage`, `two_tone_cw_f1_power_sweep_voltage`,
`two_tone_cw`, `two_tone_cw_voltage`, `two_tone_movie_cw_voltage`,
`two_tone_single_shot_cw_voltage`, `two_tone_cw_tracking_parabola`,
`two_tone_cw_voltage_tracking_parabola`, plus older non-decorated methods
`two_tone_rough`, `two_tone_power`, `two_tone`, `two_tone_tracking`,
`two_tone_and_flux_map` (earlier iterations, kept alongside the newer
`_optimized`/`_cw` ones).

Every method follows the same shape: ramp/turn on the bias source, loop over
bias points, at each point calibrate `f1`, take a reference trace, sweep `f2`
(directly or via the VNA's segmented/triggered sweep), compute `signal`,
store into `self.data`, live-plot via `self.plotter.update_twotone_imshow*`.

## 5. `Plotter` (`plotter.py`)

Two distinct responsibilities live in one class:

1. **Static/staticmethod "final figure" plots** — `plot_flux_map`,
   `plot_flux_map_current`, `plot_flux_map_voltage`, `plot_two_tone`,
   `plot_trace_mag`, `plot_trace_phase`, `plot_trace_mag_phase[_stacked]`,
   `plot_flat_pcolormesh`. These take an `ExperimentData` instance (or raw
   arrays), build a fresh matplotlib figure, and optionally annotate it with
   a text box summarizing the sweep (`generate_flux_map_comments_*`,
   `generate_two_tone_comments`, `generate_power_sweep_comments`).
2. **Stateful "live view" plots** — `update_imshow`, `update_full_imshow`,
   `update_pcolormesh`, `update_twotone_imshow_voltage` (and its
   `_single_shot_voltage` / `_movie_voltage` / `_power` siblings). These keep
   figure/axis/artist handles as instance attributes, and on each call either
   create the figure (first call) or push new data into the existing artist
   and re-`display()` it — this is what makes a sweep "watchable" in a
   Jupyter cell while it runs.

`generate_flat_mesh` builds an irregular pcolormesh grid (handles unevenly
spaced y-values per row) — used by `plot_flat_pcolormesh` and
`update_pcolormesh`.

## 6. `analysis.py` — offline post-processing

Used after a flux map has been saved, typically to turn the raw bias axis
into physical flux and to characterize the resonator:

- `find_voltage_to_flux` / `find_voltage_to_flux_manual` — locate the
  symmetry points of the flux map (either automatically via
  `find_separating_line(s)`, or by hand with a PyQt6 slider dialog,
  `flux_period_qt_widget`) and fit a linear voltage→flux conversion.
- `find_spacing_manual` / `find_spacing` — measure the frequency gap between
  the resonator branches at zero and half-flux, either by dragging two
  horizontal lines (`frequency_spacing_qt_widget`) or automatically by fitting
  resonance dips in user-picked voltage windows (`flux_interval_qt_widget` +
  `fit_resonances_voltage_intervals`, using `resonator.shunt.LinearShuntFitter`
  and a parabola fit `fit_parabola_extremum` to find each branch's extremum).
- `hamiltonian_sym` / `hamiltonian_asym` / `numerical_solution_sym` /
  `numerical_solution_asym` — build and diagonalize a SQUID Hamiltonian
  (via `qutip`) to compute the theoretical qubit frequency vs. flux, for
  comparison against measured data.

The three `*_qt_widget` functions (`flux_period_qt_widget`,
`frequency_spacing_qt_widget`, `flux_interval_qt_widget`) are near-duplicate
PyQt6 dialogs (draggable vertical/horizontal lines over a matplotlib figure,
"Remove" buttons, a "Done" button) — see the improvements doc.

## 7. Supporting / standalone files

- **`utils.py`** — small, well-documented, dependency-free numeric helpers:
  `S_to_dBm`, `round_to_significant_digits`, `is_linearly_spaced`,
  `highest_average_triplet`/`_nlet` (index of the strongest local peak/dip —
  duplicated as a static method inside `TwoToneExperiment`), `background`
  (1D interpolation), `flatten_background_horizontally`/`_vertically`,
  `generate_mask` (duplicated as `TwoToneExperiment.mask_devices`). These are
  the most reusable, best-isolated pieces of the codebase.
- **`hysteresis.py`** — polynomial hysteresis-correction math
  (`hystersis_system_of_equations`, `current_to_flux`, `plot_current_to_flux`)
  for coil current → flux conversion when the coil shows hysteresis. Not
  imported anywhere else in the package; appears to be a standalone/legacy
  tool.
- **`data_acquisition.py`**, **`data_plotting.py`**, **`file_management.py`**
  — earlier, pre-`Experiment`-class, function-based prototypes of flux-map
  acquisition, trace plotting, and file saving. Not imported by anything else
  in `experiments/`, and `data_acquisition.py` in particular references many
  names (`time`, `os`, `self`, `frequencies`, `nb_segments`, `trace_name`,
  `average`, `bw`, `comment`, `N_pointss`) that are never defined or imported
  in that file, so it cannot run as-is. These three files are effectively
  dead code left over from before the current class-based design.

## 8. Typical usage (as the code implies it)

```python
config = FluxMapConfig(yoko=YokoCurrSweepConfig(...), vna=ZnbLinConfig(...))
exp = FluxMapExperiment(sample_name="Q1", sample_code="Q1", file_directory="./data", config=config)
exp.flux_map(live_plotting=True)          # runs, live-plots, saves HDF5 on completion
exp.plot_results()                         # final annotated figure

# later, offline:
data = Experiment.load_dataclass_from_hdf5("./data/xxxxx.h5")
period, zero_flux, slope = find_voltage_to_flux(data)   # analysis.py, interactive
```

## 9. File map

| File | Role | Notes |
|---|---|---|
| `experiment.py` | Base `Experiment` class: lifecycle, instrument init/release, HDF5/NPZ persistence | Core, actively used |
| `experiment_config.py` | `FluxMapConfig`, `TwoToneConfig` | Core, small |
| `experiment_data.py` | `ExperimentData`, `FluxMapData`, `TwoToneData`, `State` | Core, small |
| `flux_map_experiment.py` | `FluxMapExperiment`: resonator-vs-flux sweeps | Core; legacy dead code at the bottom |
| `two_tone_experiment.py` | `TwoToneExperiment`: two-tone spectroscopy | Core but very large/duplicated |
| `plotter.py` | `Plotter`: live + final plots for both experiment types | Core |
| `analysis.py` | Interactive offline flux/resonance analysis (Qt + fitting) | Core for post-processing, uses `qutip`, `PyQt6`, `resonator`, `lmfit` |
| `utils.py` | Small numeric helpers | Reusable, well-documented |
| `hysteresis.py` | Coil hysteresis correction | Standalone, not imported elsewhere |
| `data_acquisition.py` | Old function-based flux-map scripts | Dead code, does not run |
| `data_plotting.py` | Old trace-plotting functions | Dead code, unused |
| `file_management.py` | Old filename/save helpers | Dead code, unused, also buggy (`os` not imported, uses `self` outside a class) |
| `setup.py` | Packaging metadata | No `install_requires` |
