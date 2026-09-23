# Experiments — Code Grading & Improvement Ideas

Read this alongside `experiments_overview.md`. The grading criterion used here
is the one that matters for this repo: **is it simple, easy to read, easy to
trust, and easy to extend for the next measurement idea?** — not "is it a
well-architected framework." Heavy abstraction would be a *worse* grade here,
even if it removed duplication, if it made a single measurement harder to
read top-to-bottom in a notebook. The ideas below aim for the smallest change
that removes real duplication/risk without adding layers.

## How this was graded

I read every `.py` file, diffed several of the near-identical methods in
`two_tone_experiment.py` against each other, and grepped for cross-file
references (e.g. does `Plotter` actually define every method
`TwoToneExperiment` calls on it). Findings below are backed by specific
line/file evidence, not general impressions.

## Weak points (most important first)

### 1. `two_tone_experiment.py` is ~3100 lines because of copy-paste, not because the problem is that big

Concrete evidence: `two_tone_optimized_average` (line 970) and
`two_tone_optimized_average_current` (line 1105) are **byte-for-byte
identical except for the method name** (135 lines, 1 line of diff). The
`_voltage` sibling (line 1241) differs from the same method in exactly 6
places, all "current → voltage" substitutions (`set_source_current_sweep` →
`set_source_voltage_sweep`, `ramp_current` → `ramp_voltage`,
`self.data.currents` → `self.data.voltages`, etc.).

This current/voltage duplication runs through the whole file (`two_tone_cw`
vs `two_tone_cw_voltage`, `two_tone_cw_f2_power_sweep` vs
`two_tone_cw_f2_power_sweep_voltage`, `two_tone_cw_tracking_parabola` vs
`two_tone_cw_voltage_tracking_parabola`, ...), multiplied again by the
cw/average/single-shot/movie/tracking axis, producing ~20 top-level
measurement methods where the actual number of distinct *behaviors* is much
smaller. Six `calibrate_f1*` variants (lines 163–272) show the same pattern
at a smaller scale — several of them are identical apart from a masking step
or a return signature.

**Why this matters more than anything else here:** a bug fixed in one
variant silently stays broken in the other 3-5 copies. A new sweep type
(e.g. "cw + voltage + f1 power sweep") means writing a 130-line method by
copy-pasting the nearest existing one and hoping every substitution was
caught — which is exactly how issue #3 below happened.

### 2. No shared "take one VNA trace" primitive — the same 3-4 lines are copy-pasted 40+ times

`self.vna.set_sweep(config)` followed by `f, z = self.vna.sweep()` (usually
followed immediately by `meta = self.vna.get_meta(); self.data.vna_meta = meta`)
appears, essentially unchanged, more than 40 times in `two_tone_experiment.py`
(e.g. around lines 166–177, 648–667, 884–950, 1015–1086, 1975–2012, 2495–2537,
2813–2856 — grep the file for `self.vna.set_sweep(` for the full list) plus 4
more times in `flux_map_experiment.py`. This is the finer-grained, more
foundational version of issue #1: it's the actual repeated snippet that makes
each copy-pasted method-variant possible in the first place.

The base `Experiment` class already establishes exactly the right precedent
for fixing this: `initialize_vna`/`release_vna` (and the `ana`/`yoko`
equivalents) already exist as one-line shared primitives on the base class so
subclasses don't each reimplement instrument setup/teardown. "Acquire one VNA
trace with this config" is the same kind of primitive, at the same level, and
it simply hasn't been pulled up there yet — this isn't a new pattern for the
codebase, just an incomplete application of one it already uses successfully.
The same is true of the "ramp the yoko to its start value and turn the output
on" sequence (`set_source_*_sweep(...)`; `output(True)`; `ramp_*(...)`),
which opens nearly every measurement method in both experiment files with
the same three lines.

### 3. A concrete, reproducible bug: `Plotter.update_twotone_imshow` doesn't exist

`two_tone_experiment.py` calls `self.plotter.update_twotone_imshow(...)` in
12 places (e.g. lines 670, 746, 822, 954, 1090, 1225, 1497, 1625, 2016, 2722,
2987, 3107) — this is the call used by every *current*-sweep two-tone
method. But `plotter.py` only defines `update_twotone_imshow_voltage`,
`update_twotone_imshow_single_shot_voltage`, `update_twotone_imshow_movie_voltage`,
and `update_twotone_imshow_power` — there is no plain `update_twotone_imshow`.
Any current-sweep two-tone measurement run with `live_plotting=True` (the
default in most of these methods) will raise `AttributeError` the first time
it tries to plot.

This is a direct consequence of #1: the voltage-sweep variant was added by
copying the current-sweep one and renaming the plotting call to
`_voltage`, but the base method's own call was apparently never fixed/tested
afterward, or `Plotter` was refactored and only the `_voltage` call sites
were updated.

### 4. Dead code and legacy code are shipped mixed in with live code

- `data_acquisition.py` references `time`, `os`, `self`, `frequencies`,
  `nb_segments`, `N_pointss`, `trace_name`, `average`, `bw`, `comment` —
  none of which are defined or imported in that file. It cannot execute.
- `data_plotting.py` and `file_management.py` are unused by any other file
  in the package (confirmed by grep) and `file_management.save_data` also
  references undefined `self`/`os` outside of a class.
- `flux_map_experiment.py`'s `flux_map_power_dependence` (bottom of file)
  calls `self.instruments[...]` / `self.configs_dict[...]`, a dict-based API
  that no longer exists on `Experiment` (which now uses `self.vna`/`self.yoko`
  directly and a typed `config`). It cannot execute either.
- `hysteresis.py` is self-contained and not imported by anything else in
  `experiments/`.

None of this is harmful by itself, but a newcomer cannot tell "legacy,
kept for reference" apart from "current, supported" without reading every
line, and a search for "how do I save data" or "how do I plot a trace" will
surface both the live and the dead version.

### 5. No tests, no CI, no pinned dependencies

There is no test file anywhere, and `setup.py` declares no
`install_requires` (the code imports `h5py`, `numpy`, `matplotlib`, `qutip`,
`PyQt6`, `scikit-learn`, `lmfit`, `resonator`, `tqdm`, `IPython` — none of
which are declared). Most of this codebase does need real instruments to
test end-to-end, but a good third of it doesn't: `utils.py`,
`Plotter.generate_flat_mesh`, the HDF5 round-trip in `experiment.py`, the
fitting/geometry helpers in `analysis.py` (`find_separating_line(s)`,
`fit_parabola_extremum`, `parabola`). None of that is currently tested, so a
refactor (including the ones suggested below) has no safety net.

### 6. `Plotter` conflates two jobs with two different lifetimes — this is more than a style nit

`Plotter` currently does two unrelated things in one class:

**(a) Stateless, `@staticmethod` "turn this saved `Data` object into a
finished figure" functions** — `plot_flux_map`, `plot_two_tone`,
`plot_trace_mag_phase`, the `generate_*_comments` helpers, etc. These take a
`Data` object as an explicit argument and touch no instance state at all;
`analysis.py` already calls several of them directly as
`Plotter.plot_flux_map(data)` on data freshly loaded from an HDF5 file, with
no `Experiment` in sight. **This half is doing the right thing**: it's
decoupled from any live instrument session, so "replot a measurement from six
months ago" or "compare two saved runs side by side" only needs the `Data`
object.

**(b) Stateful "keep one open figure updated as new rows of data arrive"
instance methods** — `update_full_imshow`, `update_pcolormesh`,
`update_twotone_imshow_*` — which hold mutable per-run state (`imshow_z`,
`pcolor_x/y/z`, `twotone_fig`, ...) as *class*-level attributes, populated
lazily on first write because `Plotter.__init__` is just `pass`. This state
should live and die with exactly one measurement run, but nothing currently
enforces that — there's no `reset()`/`close()` that
`start_experiment()`/`end_experiment()` call, so "this window's data" and
"this experiment run" are two independent lifecycles that happen to usually
line up only because each notebook cell tends to create one `Experiment` and
one `Plotter`. The one place this mismatch already leaks into `Experiment`
is `end_experiment()`'s `plt.close('all')` — a blunt, global "close every
matplotlib figure that exists anywhere" call standing in for what should be
"tell my plotter this run is over," because `Plotter` doesn't expose such a
method.

**Should live plotting move into `Experiment`, then?** No — merging (a) into
`Experiment` would be a step backward: it would tie "can I replot this data"
to "do I currently have a live instrument session," which is exactly the
coupling that *isn't* a problem today and that `analysis.py` deliberately
relies on not having. It would also scatter matplotlib details (`imshow`,
colorbars, `extent=`, ...) into the measurement methods themselves, undoing
the exact separation the `experiment_method` decorator earns (strong point
#2 below): a measurement method should read as the physics sequence, not as
physics-sequence-interleaved-with-plotting-code.

But (b) genuinely *is* state that belongs to one `Experiment` run, and today
it's modeled as an object with no lifecycle tie to that run at all — that's
the real bug, not "it's a separate class." The fix is to give it an explicit,
owned lifecycle rather than to fold it into `Experiment`: see the
corresponding improvement idea below for the concrete split (stateless
plotting stays a separate, freely-reusable module; the stateful live view
becomes a small object that `start_experiment()` creates fresh and
`end_experiment()` explicitly closes).

Separately, `plot_trace_mag_phase`/`plot_trace_mag_phase_stacked` are
implemented nearly identically a third time in the (dead) `data_plotting.py`
— resolved automatically once #4 above (delete dead code) happens.

### 7. Scattered `print()` debugging instead of a single verbosity switch

`two_tone_experiment.py`'s measurement methods are full of ad hoc
`print(time.time() - start)` / `print("configured")` / `print(f'f shape: ...')`
style debug prints (e.g. lines 1007–1057). They're useful during development
but there's no way to turn them off for a routine run, and they'll be
duplicated N times over thanks to issue #1.

### 8. Minor: inconsistent dB math

`utils.S_to_dBm` correctly uses `20 * np.log10(...)`. In `two_tone_experiment.py`
and once in `flux_map_experiment.py`, dip-finding uses `20*np.log(np.abs(...))`
(natural log, missing the `/np.log(10)` factor) about 25 times. In every case
found this is only used to pick the *index* of the strongest dip
(`find_highest_average_triplet`), where a constant scale factor doesn't
change the answer — so this looks harmless today, but it's a landmine if any
of these values are ever logged, plotted, or compared against a real dB
threshold. Worth a quick audit and a switch to the existing `S_to_dBm` helper
for consistency.

### 9. Minor: hardcoded instrument addresses as code defaults

`experiment.py`'s `initialize_vna`/`initialize_ana`/`initialize_yoko` have
lab-specific IP addresses/GPIB addresses as default argument values. Fine for
a single-setup lab, but it means "point this at a different rig" requires
editing library code rather than the notebook/config.

## Strong points (most important first)

### 1. The Config / Data / Experiment split is the right shape for this problem

`*Config` (what to tell the instruments), `*Data` (what came back, plus
run metadata) and `Experiment` (the verb: how to get from one to the other)
map cleanly onto how a physicist actually thinks about a measurement, and
the split is consistently applied across `FluxMapExperiment` and
`TwoToneExperiment`. This is a genuinely simple, low-abstraction design —
worth explicitly preserving in any refactor.

### 2. `experiment_method` centralizes the lifecycle well

Timing, state tracking, instrument init/release, `KeyboardInterrupt` →
`INTERRUPTED` state, and the final HDF5 save are all handled in one decorator
in `experiment.py`. Every measurement method benefits without repeating this
logic, and a physicist reading a measurement method sees only the physics
loop, not bookkeeping.

### 3. `safe_initializer` gives graceful degradation for free

Being able to run/develop against a subset of connected instruments (rather
than crashing on the first missing one) is a genuinely useful property for
lab code, and it's implemented as a two-line decorator rather than scattered
try/except blocks.

### 4. Generic HDF5 (de)serialization via dataclass reflection

`save_dataclass_to_hdf5`/`load_dataclass_from_hdf5` work for *any*
`ExperimentData` subclass without per-class boilerplate — adding a new field
to `TwoToneData` doesn't require touching the save/load code. This is a good
example of exactly the right amount of generality: general enough to avoid
repetition, not so general it needs a plugin system.

### 5. Live plotting fits the actual workflow

Driving matplotlib through `IPython.display`/`clear_output` so a sweep can be
watched live in a notebook cell is a good match for how these measurements
are actually run and monitored, and `update_full_imshow`/`update_pcolormesh`
handle the "data arrives one row at a time" case cleanly.

### 6. `utils.py` and parts of `analysis.py` are well-documented and self-contained

Docstrings with parameter/return descriptions and examples exist for most of
`utils.py` and several `analysis.py` functions, and these modules have no
side effects and few dependencies — they're the easiest parts of the repo to
read, reuse, and (if it comes to it) test.

## Improvement ideas & strategy, ordered by importance

### 1. Fix the `update_twotone_imshow` crash first

This is a one-line, zero-risk fix and it's an active bug blocking a whole
class of measurements. Simplest fix in the spirit of "small diff, no new
abstraction": rename the "current" call sites to use
`update_twotone_imshow_voltage` with a generic axis argument (it already
takes the x-values as a parameter — it isn't voltage-specific except in
name), or add a thin `update_twotone_imshow` alias. Do this before anything
else below so nobody hits it in the meantime.

### 2. Extract a shared `acquire_vna_trace` primitive onto `Experiment`

Do this before #3/#4 below — it removes one axis of "what's slightly
different between these two methods" before the merges start, which makes
each merge smaller and easier to review. Add one method to the `Experiment`
base class, next to `initialize_vna`/`release_vna`:

```python
def acquire_vna_trace(self, config) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Configure and take one VNA trace; caller decides where to store it."""
    self.vna.set_sweep(config)
    f, z = self.vna.sweep()
    meta = self.vna.get_meta()
    return f, z, meta
```

and replace the 40+ call sites of `self.vna.set_sweep(...)` / `self.vna.sweep()`
(often followed by `self.vna.get_meta()`) with a call to it. It deliberately
returns the trace and metadata rather than writing into `self.data` itself —
different call sites store the result into `self.data.f`/`S` vs.
`self.data.f_ref`/`S_ref`, so a primitive that guesses the field name would
just reintroduce a different kind of special-casing. Do the same for the
"ramp the yoko to its start value and turn the output on" sequence that opens
nearly every measurement method (`set_source_*_sweep(...)`; `output(True)`;
`ramp_*(...)`).

### 3. Collapse the current/voltage duplication with one small helper, not a class hierarchy

The current vs. voltage variants differ in exactly 4-6 lines each time:
which `set_source_*_sweep`/`ramp_*`/`output` calls to make, which
`self.data.currents`/`voltages` array to fill, and the axis label passed to
the plotter. Rather than introducing a "SweepAxis" class hierarchy (too much
machinery for 4 lines of difference), the simplest fix that removes the
duplication is:

- Add a tiny helper on `Experiment` (or a 10-line free function) that, given
  the `yoko` config, returns `(values, set_value_fn, ramp_fn, data_field_name)`
  — e.g. `_yoko_sweep_axis(self)` returning
  `(self.config.yoko.currents, self.yoko.current, self.yoko.ramp_current, 'currents')`
  or the voltage equivalents based on `isinstance(self.config.yoko, YokoCurrSweepConfig)`.
- Every `_current`/`_voltage` pair becomes one method that calls this helper
  once at the top and otherwise reads identically. This alone should remove
  roughly half of `two_tone_experiment.py` without changing behavior or
  adding a new concept a reader has to learn — it's the same pattern
  `flux_map_experiment.py`'s methods already almost share.

### 4. Then collapse the acquisition-mode axis (average / cw / single-shot / movie / tracking) the same way

Once #3 is done, look at what's actually different between e.g.
`two_tone_optimized_average` and `two_tone_cw`: mostly the inner
per-bias-point acquisition (how `f2` is swept and how `signal` is computed),
not the outer bias loop. Pull the outer loop (ramp, iterate bias points, save
per-point metadata, call the plotter) into one private method that takes a
small callback / strategy object for "acquire and return signal for this
bias point." This turns ~10 near-identical 100+ line methods into ~10 short
methods that each define only their specific acquisition callback, sharing
one loop. Keep the callback as a plain function, not a new class hierarchy —
this is a case where "extract the repeated 80 lines into a helper" is enough,
no framework needed.

Do #2, #3 and #4 as separate, reviewable changes, and lean on real saved data
(compare before/after HDF5 output on a rerun with mocked instruments, or at
minimum a dry run) rather than doing them all at once — the size of this file
makes an all-at-once rewrite risky to review.

### 5. Delete the dead code

`data_acquisition.py`, `data_plotting.py`, `file_management.py`, and the
non-functional legacy methods (`flux_map_power_dependence`,
`initialize_flux_map_adaptive_data`/`flux_map_adaptive`) can simply be
deleted — they don't run today and nothing imports them. If any of the ideas
in them are still wanted (e.g. adaptive flux maps), track that as a task
rather than keeping broken code around "for reference." This is the cheapest
possible win: it reduces the surface area newcomers have to read with zero
risk, since none of it currently executes.

### 6. Add a small, fast test layer around the pure-logic pieces — before refactoring #3/#4

Before touching `two_tone_experiment.py`, pin down the parts that don't need
hardware with simple tests: `utils.py` functions (most already have clear
docstring examples that would make ready-made test cases), `generate_flat_mesh`,
`Experiment.save_dataclass_to_hdf5`/`load_dataclass_from_hdf5` round-tripping
a small fake `FluxMapData`, and `analysis.find_separating_line(s)` /
`fit_parabola_extremum` / `parabola`. This is maybe half a day of work and
gives a real safety net for the duplication cleanup above, without requiring
any instrument mocking framework.

### 7. Add a `requirements.txt` (or `pyproject.toml` dependencies) and fill in `install_requires`

List `numpy`, `h5py`, `matplotlib`, `qutip`, `PyQt6`, `scikit-learn`, `lmfit`,
`resonator`, `tqdm`, `ipython`, plus the external `instruments` package
(however it's installed today — presumably `pip install -e` from a sibling
repo). This is low-effort and prevents "works on my machine" setup pain.

### 8. Consolidate the three Qt slider-dialog functions in `analysis.py`

`flux_period_qt_widget`, `frequency_spacing_qt_widget`, and
`flux_interval_qt_widget` are ~90% identical PyQt6 boilerplate (build a
dialog, embed the existing figure, add draggable vertical and/or horizontal
lines with sliders and remove buttons, a Done button, run the event loop,
return final positions). One shared "draggable line dialog" function
parameterized by which lines to draw (N vertical pairs, 2 horizontal, or
both) would cut this from ~350 lines to under 150 and mean a bug in the
slider/remove logic only needs fixing once.

### 9. Route debug output through one place

Replace the scattered `print(...)` calls in `two_tone_experiment.py` with
either a module-level `verbose` flag checked before printing, or Python's
`logging` module at `DEBUG` level. Either is a small change; the goal is just
"one switch to silence routine runs," not a logging framework.

### 10. Split `Plotter` into a stateless plotting module and a per-run live view with an explicit lifecycle

Two small, independent changes — not a new abstraction layer:

- Move the `@staticmethod` "final figure" functions (`plot_flux_map*`,
  `plot_two_tone`, `plot_trace_mag_phase*`, the `generate_*_comments`
  helpers) out to plain module-level functions in `plotter.py`, or keep them
  on `Plotter` as a pure namespace — they don't use `self` today, so this is
  mechanical either way. Callers don't need to change if you keep the
  namespace: `analysis.py` and `plot_results()` already call these the same
  way a static, stateless function would be called.
- Turn the stateful live-view methods (`update_full_imshow`, `update_pcolormesh`,
  `update_twotone_imshow_*`) into a small `LiveView` object that
  `start_experiment()` creates fresh for that run (instead of relying on
  `Plotter.__init__` being a no-op and state being created lazily on first
  write) and that `end_experiment()` explicitly closes via a real
  `self.live_view.close()`, replacing today's blanket `plt.close('all')`.

This directly answers "should live plotting move into `Experiment`?" with
*no, but its lifecycle should be owned by `Experiment`* — `start_experiment`/
`end_experiment` decide *when* a live view exists, the (separate, reusable)
plotting code decides *how* it's drawn. It also sets up Step 9 of
`experiments_refactor.md` (the Qt popup window for live plotting) to be a
self-contained swap of `LiveView`'s internals, since callers only ever see
`update(...)`/`close()`.

### 11. Move hardcoded instrument addresses to the config/notebook layer

Make `initialize_vna`/`initialize_ana`/`initialize_yoko` require the address
(or read it from the `Config`/environment) rather than defaulting to
lab-specific addresses baked into `experiment.py`. Keeps the library
rig-agnostic without adding any real complexity.
