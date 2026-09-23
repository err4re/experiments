# Experiments — Step-by-Step Refactor Plan

This is an actionable, ordered plan to go from the current state (documented
in `experiments_overview.md`, graded in `experiments_improvements.md`) to a
cleaner codebase, plus a dedicated deep-dive on two things explicitly asked
for: the style of the `experiment_method` decorator, and moving live plotting
out of the notebook into a popup PyQt window for speed.

**Ground rule for every step below:** land it as its own small, reviewable
change, run/read the affected experiment top-to-bottom afterward, and don't
start the next step until the previous one is committed. Nothing here should
add a new concept a reader has to learn — every step either removes code,
inlines duplication into one place, or swaps an implementation behind an
unchanged interface (the calls `FluxMapExperiment`/`TwoToneExperiment` make
into `Plotter` stay the same throughout steps 10–11, for instance).

## Step 0 — re-read findings that changed since the last pass

Reading the decorator and the flux-map methods together turned up one more
concrete, previously-unreported bug, described in detail in Step 2. Worth
fixing alongside the decorator cleanup since it's the same code region.

## Step 1 — fix the two known crashes/bugs first (no refactor yet)

Do this before anything else so nobody hits them while steps 2+ are in
progress.

1. `Plotter` has no `update_twotone_imshow` method, but
   `two_tone_experiment.py` calls it 12 times (the "current sweep" live-plot
   call). Either add the method (call through to
   `update_twotone_imshow_voltage` with a generic axis argument — it isn't
   voltage-specific except in name) or rename the call sites. This is
   `experiments_improvements.md` item #3.
2. **New finding**: in `flux_map_experiment.py`, `flux_map`,
   `flux_map_voltage_sweep`, `flux_map_constant_voltage`, and `power_sweep`
   all start with `self.data = self.initialize_flux_map_data()` (or a
   sibling) as their *first line*. But the `experiment_method` decorator
   (see Step 2) has already set `self.data.measurement_function` and
   `self.data.state` on the *old* `self.data` object one line earlier, in
   `start_experiment()`/the wrapper. Reassigning `self.data` throws that
   away — the freshly-constructed `FluxMapData(...)` doesn't carry
   `measurement_function` or `state` forward, and `FluxMapData` defaults
   both to `None`. Net effect: **every flux-map HDF5 file has
   `measurement_function = None`** (state happens to come out right only
   because `end_experiment()` sets it again right before saving, after the
   method body has already run). Fix: either have these methods fill in the
   pre-existing arrays on `self.data` instead of replacing the whole object
   (matches how `TwoToneExperiment` already does it — see Step 2's proposed
   decorator contract), or have the decorator set
   `measurement_function`/re-stamp `state` *after* the method body runs
   instead of before.

## Step 2 — clean up `experiment.py` style issues (small, isolated, high value)

This file is small (~410 lines) and is the one every experiment depends on,
so getting its style right pays off everywhere else.

1. **`experiment_method` is not marked `@staticmethod`.** It's defined
   inside the `Experiment` dataclass body and used as `@Experiment.experiment_method`
   from subclasses — that works today (Python 3 returns a plain function
   when you access it off the class rather than an instance), but it reads
   as if it were an instance method that happens to take `method` as `self`,
   which is confusing on first read. Add `@staticmethod` — zero behavior
   change, removes the "wait, why does this work?" moment for the next
   reader. See the dedicated deep-dive below for the full before/after.
2. **Document (or enforce) the decorator's contract.** Right now nothing
   states "the decorated method should only *fill in* `self.data`'s fields,
   never replace `self.data` itself" — which is exactly the assumption Step
   1.2's bug violated. Either add that as a one-line docstring on
   `experiment_method`, or make the decorator itself responsible for
   allocating `self.data` (call a new `self.allocate_data()` hook right
   before invoking the method, so the method body never touches `self.data`
   identity at all — only the pre-existing `initialize_data()` currently
   called in `__init__` needs to also be callable per-run). The second
   option is slightly more work but removes an entire category of "forgot to
   carry a field forward" bugs.
3. **`InstrumentDict` and `ConfigurationDict`** (lines 26–76, both `TypedDict`
   with long docstrings) are never referenced anywhere in the codebase
   (confirmed by grep). Delete them, or if they were meant to type
   `Experiment.vna`/`ana`/`yoko`/`config`, actually wire them in — as dead
   code they're pure reading overhead.
4. **`data: Optional[ExperimentData] = None` vs `data_dict: Dict[str, Any] = field(default_factory=dict) #legacy stuff to handle .npz`** —
   both live on `Experiment` side by side. `data_dict`/`save_data_npz` is
   marked "legacy" in its own comment; if nothing still calls
   `save_data_npz` in current notebooks, delete `data_dict` and
   `save_data_npz` together and keep only the HDF5 path. If it's still
   needed for one workflow, say so in a comment instead of "legacy stuff."
5. **Optional typing is inconsistent with runtime defaults.** `plotter: Plotter = None`,
   `f: np.ndarray[np.float64] = None`, etc. default to `None` but aren't
   typed `Optional[...]`. Cheap, mechanical fix (add `Optional[...]`
   everywhere a field defaults to `None`) — worth doing in the same pass as
   #3/#4 since you're already touching these dataclasses.
6. **`pause_experiment(self): pass`** is a stub that implies paused-state
   support exists. Either implement it minimally (set `self.data.state` to a
   new `PAUSED` `State` value) or remove it until it's needed — a no-op
   public method is a promise the code doesn't keep.
7. **Dataclass-vs-manual-`__init__` mismatch.** `Experiment` is `@dataclass`
   (auto-generated `__init__`), but both `FluxMapExperiment` and
   `TwoToneExperiment` are plain classes that hand-write their own `__init__`
   and call `super().__init__(...)` with a subset of fields, then manually
   set `self.data`/`self.plotter` afterward. This works, but it means the
   `@dataclass` decorator on `Experiment` buys almost nothing (no subclass
   actually gets a generated `__init__`) — pick one style: either drop
   `@dataclass` from `Experiment` and treat it as a plain base class (matches
   what's actually happening), or make the subclasses proper dataclasses too
   (`@dataclass` + `__post_init__` for the `initialize_data()`/`Plotter()`
   setup) so the pattern is consistent top to bottom.

## Step 3 — delete dead code (cheap, zero risk)

Per `experiments_improvements.md` #4/#5: delete `data_acquisition.py`,
`data_plotting.py`, `file_management.py` (none of them are imported anywhere
and `data_acquisition.py` can't even run — it references undefined names),
and the non-functional legacy methods in `flux_map_experiment.py`
(`flux_map_power_dependence`, `initialize_flux_map_adaptive_data`,
`flux_map_adaptive`). If `hysteresis.py` isn't used from any current
notebook either, confirm with whoever owns that workflow before removing it
— it's self-contained and undamaging to keep, but it's also currently
undiscoverable (not imported, not documented) so it's worth a decision either
way rather than leaving it ambiguous.

## Step 4 — extract a shared `acquire_vna_trace` primitive (and see the deep-dive below for where primitives like this belong)

Re-reading `two_tone_experiment.py` and `flux_map_experiment.py` side by
side turns up a duplication finer-grained than Step 5/6 below, and worth
fixing first: `self.vna.set_sweep(config); f, z = self.vna.sweep()`
(usually immediately followed by `meta = self.vna.get_meta(); self.data.vna_meta = meta`)
appears essentially unchanged more than 40 times across the two files. Add
one method to the `Experiment` base class, right next to
`initialize_vna`/`release_vna` (which already establish exactly this "shared
primitive lives on the base class" pattern for instrument setup/teardown):

```python
def acquire_vna_trace(self, config) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Configure and take one VNA trace; caller decides where to store it."""
    self.vna.set_sweep(config)
    f, z = self.vna.sweep()
    meta = self.vna.get_meta()
    return f, z, meta
```

It returns the trace and metadata rather than writing into `self.data`
itself, since different call sites store the result into `self.data.f`/`S`
vs. `self.data.f_ref`/`S_ref` — a primitive that guesses the field name would
just reintroduce a different kind of special-casing. Do the same for the
"ramp the yoko to its start value and turn the output on" sequence
(`set_source_*_sweep(...)`; `output(True)`; `ramp_*(...)`) that opens nearly
every measurement method. Doing this before Step 5 shrinks the diff Step 5
has to review, since one more axis of "what varies between these two
methods" is gone before the merge starts.

## Step 5 — collapse the current/voltage duplication

The single biggest size reduction available. In both `flux_map_experiment.py`
and `two_tone_experiment.py`, the `_current` and `_voltage` variants of each
method differ only in: which `set_source_*_sweep`/`ramp_*`/`current`/`voltage`
calls to make, which `self.data.currents`/`voltages` array to fill, and the
axis label passed to the plotter.

Add one small helper (no new class hierarchy — this is 4-6 lines of
variation, not a new abstraction layer):

```python
def _bias_axis(self) -> tuple[np.ndarray, Callable, Callable, str]:
    """Returns (values, set_fn, ramp_fn, data_field_name) for whichever
    bias (current or voltage) this experiment's yoko config sweeps."""
    if isinstance(self.config.yoko, YokoCurrSweepConfig):
        return self.config.yoko.currents, self.yoko.current, self.yoko.ramp_current, "currents"
    return self.config.yoko.voltages, self.yoko.voltage, self.yoko.ramp_voltage, "voltages"
```

Every `_current`/`_voltage` pair becomes one method that calls this once at
the top and is otherwise identical. Do `flux_map_experiment.py` first (4
methods, low risk, good rehearsal) before touching `two_tone_experiment.py`
(same pattern, ~10 pairs, higher risk because the file is larger and less
tested).

## Step 6 — collapse the acquisition-mode duplication in `two_tone_experiment.py`

After Step 5, look at what's left differing between e.g.
`two_tone_optimized_average` and `two_tone_cw`: the outer bias loop is now
shared, but the *inner* per-bias-point acquisition (how `f2` is swept, how
many averages, how `signal` is computed) still differs. Extract that inner
step into a private method or a plain callback, and give each public
`two_tone_*` method a short body that just wires bias-loop (Step 5's helper)
+ its specific acquisition callback together. This should turn ~10
near-identical 100–150 line methods into ~10 short methods (10–20 lines each)
sharing one loop implementation. Do this as its own commit, separate from
Step 5, and compare a saved HDF5 file before/after on a real (or mocked)
sweep to make sure nothing silently changed.

## Step 7 — consolidate the `calibrate_f1*` family

Six methods (`calibrate_f1`, `calibrate_f1_z`, `calibrate_f1_z_short`,
`calibrate_f1_z_short_masked`, `calibrate_f1_masked`, `calibrate_f1_z_masked`)
reduce to one function with two boolean-ish parameters: `masked: bool` and
`return_trace: bool` (or just always return the trace and let callers ignore
what they don't need — simpler still, since discarding a tuple element costs
nothing). Also fold `TwoToneExperiment.mask_devices` into
`utils.generate_mask`, which already does the same thing.

## Step 8 — consolidate the three Qt slider dialogs in `analysis.py`

`flux_period_qt_widget`, `frequency_spacing_qt_widget`, and
`flux_interval_qt_widget` share ~90% of their code (build dialog, embed
figure, add draggable line(s) with sliders + remove buttons, Done button,
run event loop, read back positions). Write one
`_draggable_line_dialog(fig, vertical_labels, horizontal_labels, ...)` and
have the three current functions become thin wrappers that call it with
different label sets. This is independent of Steps 5–7 and can be done in
any order relative to them.

## Step 9 — add a small test layer for the pure-logic code

Before Steps 5–8 land (or at least before merging them), add tests for the
parts that don't need real hardware: `utils.py` (most functions already have
worked examples in their docstrings — turn those into test cases directly),
`Plotter.generate_flat_mesh`, `Experiment.save_dataclass_to_hdf5`/
`load_dataclass_from_hdf5` round-tripping a small fake `FluxMapData`, and
`analysis.find_separating_line(s)`/`fit_parabola_extremum`/`parabola`. This
gives Steps 5–7 (which touch the riskiest, least-tested file in the repo) a
real safety net instead of "read the diff carefully and hope."

## Step 10 — split `Plotter` into a stateless module and a per-run live view with an explicit lifecycle

Full analysis in the deep-dive below (it directly answers "is `Plotter`
implemented well, and should it move into `Experiment`?"). In short: pull the
`@staticmethod` "final figure from saved data" functions out into either
plain module functions or a slimmer namespace (no behavior change, they
don't use `self` today), and turn the stateful "keep updating this open
figure" methods into a small `LiveView` object that `start_experiment()`
creates fresh and `end_experiment()` explicitly closes — replacing today's
blanket `plt.close('all')` with a real `self.live_view.close()`. Do this
before Step 11: once the live view has an explicit `update(...)`/`close()`
lifecycle, swapping what happens inside it (notebook inline → native Qt
window) becomes a self-contained change that doesn't touch
`FluxMapExperiment`/`TwoToneExperiment` at all.

## Step 11 — swap notebook live-plotting for a popup Qt window

This is the other explicit ask — full analysis and design below. Doing Steps
5/6 first means there are far fewer `self.plotter.update_*`/`self.live_view.update(...)`
call sites to have ever touched (from ~20 down to a handful), and doing Step
10 first means this step only has to change `LiveView`'s internals, not any
code in `FluxMapExperiment`/`TwoToneExperiment`.

## Step 12 — packaging (`requirements.txt`) and CI

Once the above is stable, add a `requirements.txt`/`pyproject.toml`
dependency list (`numpy`, `h5py`, `matplotlib`, `qutip`, `PyQt6`,
`scikit-learn`, `lmfit`, `resonator`, `tqdm`, `ipython`) and a CI job that at
minimum runs the Step 9 tests. Last because it's the lowest-risk, most
mechanical step and benefits from the codebase already being smaller.

---

## Deep dive: the `experiment_method` decorator

Current code (`experiment.py`):

```python
class Experiment:
    ...
    # decorator for methods that execute an experiment
    def experiment_method(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self: Experiment, *args, **kwargs):
            self.start_experiment()
            self.data.measurement_function = method.__name__
            interrupted = False
            try:
                result = method(self, *args, **kwargs)
            except KeyboardInterrupt:
                interrupted = True
                print("Experiment interrupted by user or kernel.")
                raise
            finally:
                self.end_experiment(interrupted=interrupted)
            return result
        return wrapper
```

What's good here, worth keeping exactly as-is: the try/except/finally shape
is correct and idiomatic — `KeyboardInterrupt` is caught just to flag
`interrupted=True`, then re-raised, and `finally` guarantees
`end_experiment()` (release instruments, save data) always runs. That's the
right way to write a "no matter what happens, clean up" decorator in Python.

What's stylistically off, and the minimal fix for each:

1. **Not `@staticmethod`.** Defined as a plain function inside the class
   body, used via `Experiment.experiment_method` (class access, not instance
   access) as a decorator in subclasses. This *works* — Python 3 only wraps
   a function in a bound method when you access it through an *instance*,
   not through the class — but it looks like it should need `self` and
   doesn't, which is exactly the kind of "why does this work?" a reader
   shouldn't have to puzzle out. One-line fix:

   ```python
   @staticmethod
   def experiment_method(method: Callable[..., Any]) -> Callable[..., Any]:
       ...
   ```

   No behavior change (`@staticmethod` on a function that already never uses
   `self` is a no-op at runtime, purely a readability fix), and it now matches
   `safe_initializer` just below it, which *is* already `@staticmethod` —
   right now the two decorators in the same file are written in two
   different styles for no reason.

2. **Implicit, undocumented contract with the method it wraps.** As found in
   Step 1.2, the decorator assumes the wrapped method never reassigns
   `self.data`. That assumption isn't written down anywhere and one whole
   subclass (`FluxMapExperiment`) violates it. A one-line docstring fixes the
   documentation gap immediately; the more thorough fix (decorator owns data
   allocation, method only fills fields) removes the possibility of the bug
   recurring. Suggested version:

   ```python
   @staticmethod
   def experiment_method(method: Callable[..., Any]) -> Callable[..., Any]:
       """Wrap a measurement method with start/end lifecycle handling.

       The wrapped method must not reassign `self.data`; it should only
       mutate the fields of the `ExperimentData` instance already set by
       `initialize_data()`, so that `state`/`measurement_function`/
       `start_time` (set here and in `start_experiment`) survive.
       """
       @functools.wraps(method)
       def wrapper(self: "Experiment", *args, **kwargs):
           self.start_experiment()
           self.data.measurement_function = method.__name__
           interrupted = False
           try:
               result = method(self, *args, **kwargs)
           except KeyboardInterrupt:
               interrupted = True
               print("Experiment interrupted by user or kernel.")
               raise
           finally:
               self.end_experiment(interrupted=interrupted)
           return result
       return wrapper
   ```

3. **Minor naming nit, not worth its own step:** `method` as the parameter
   name is fine, but note it shadows the outer `experiment_method`'s own name
   conceptually (a "method that decorates methods" called `method` inside
   itself) — purely cosmetic, skip unless touching this code anyway.

Nothing else about this decorator needs to change — it's one of the better
pieces of the codebase (see `experiments_improvements.md`, strong point #2),
this is a polish pass, not a rewrite.

---

## Deep dive: is `Plotter` implemented well? Where should a "take one VNA trace" method live?

### Is `Plotter` smart today?

Half of it, yes. `Plotter` currently holds two things that have nothing to
do with each other:

- **Stateless functions that turn a saved `Data` object into a figure** —
  `plot_flux_map`, `plot_two_tone`, `plot_trace_mag_phase`, the
  `generate_*_comments` helpers. These are `@staticmethod`, take a `Data`
  instance as an explicit argument, and touch no instance state. This part
  is genuinely well designed: `analysis.py` already calls
  `Plotter.plot_flux_map(data)` directly on data loaded back from an HDF5
  file, with no `Experiment` involved at all. That's the right shape —
  plotting a saved measurement shouldn't require re-creating a live
  instrument session.
- **Stateful methods that keep one open figure updated as new data arrives**
  — `update_full_imshow`, `update_pcolormesh`, `update_twotone_imshow_*`.
  These hold mutable state (`imshow_z`, `pcolor_x/y/z`, `twotone_fig`, ...)
  as *class*-level attributes (`Plotter.__init__` is just `pass`, so nothing
  ever resets them per-instance). This half is where the design is weaker:
  the state's *natural* lifetime is "one running measurement," but nothing
  ties it to that — there's no `Plotter.reset()`/`close()` that
  `start_experiment()`/`end_experiment()` call. The clearest symptom is
  `end_experiment()` reaching past `Plotter` entirely and calling the global
  `plt.close('all')`, because `Plotter` has no method to ask for "this run's
  view is done."

### Should live plotting move inside `Experiment`?

No. Folding the stateless half into `Experiment` would tie "can I look at
this data" to "do I have a live instrument session right now" — the opposite
of what `analysis.py` relies on today (replotting/comparing saved runs with
no instruments attached at all). Folding the stateful half in would scatter
`imshow`/colorbar/`extent=` details into `flux_map`/`two_tone_*`'s bodies,
undoing exactly the separation `experiment_method` earns elsewhere: those
methods should read as the measurement sequence, not as
measurement-interleaved-with-matplotlib-calls.

What *should* change is narrower: give the stateful half an explicit
lifecycle owned by `Experiment`, without moving its code there. Concretely
(this is Step 10 above): `start_experiment()` creates a fresh `LiveView`
object for the run (instead of `Plotter`'s class-level attributes being
implicitly reused/created on first write), and `end_experiment()` calls
`self.live_view.close()` instead of `plt.close('all')`. `Experiment` decides
*when* a live view exists; `plotter.py` still decides *how* it's drawn.

### Where does a simple primitive like "acquire one VNA trace" belong?

On `Experiment` — and the codebase already shows why, one layer down.
`initialize_vna`, `release_vna` (and the `ana`/`yoko` equivalents) are
already exactly this kind of thing: small, shared, instrument-touching
primitives that live on the base class so `FluxMapExperiment` and
`TwoToneExperiment` don't each reimplement instrument setup. "Take one VNA
trace with this config" is the same category of primitive, just missing —
right now `self.vna.set_sweep(config); f, z = self.vna.sweep()` (often
followed by `meta = self.vna.get_meta()`) is copy-pasted more than 40 times
across `flux_map_experiment.py` and `two_tone_experiment.py` instead of
being pulled up next to `initialize_vna`.

The distinguishing question that decides "does this belong on `Experiment`,
or decoupled from it like `Plotter`'s static half?" is: **does it need a
live instrument, or does it only ever consume a `Data` object?**

- Needs `self.vna`/`self.ana`/`self.yoko` to do anything → belongs on
  `Experiment` (or a free function that takes the instrument handle
  explicitly, if you want it usable/testable without a full `Experiment`
  instance — either is fine, the point is it's instrument-facing code, not
  data-facing code). `acquire_vna_trace`, the yoko ramp-and-arm sequence, and
  `calibrate_f1` all fall here.
- Only ever consumes an already-acquired `Data` object → belongs outside
  `Experiment`, as it already mostly does with `Plotter`'s static half and
  `analysis.py`. Nothing here needs a live instrument, and coupling it to
  `Experiment` would only make it harder to reuse on saved data later.

Concretely, add to `Experiment`:

```python
def acquire_vna_trace(self, config) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Configure and take one VNA trace; caller decides where to store it."""
    self.vna.set_sweep(config)
    f, z = self.vna.sweep()
    meta = self.vna.get_meta()
    return f, z, meta
```

so that, for example, `FluxMapExperiment.flux_map`'s inner loop goes from:

```python
self.vna.set_sweep(self.config.vna)
f, z = self.vna.sweep()
self.data.f = f
self.data.S[i] = z
```

to:

```python
f, z, meta = self.acquire_vna_trace(self.config.vna)
self.data.f = f
self.data.S[i] = z
self.data.vna_meta = meta
```

— one line shorter here, but the real win is the other 40-odd call sites
that currently each hand-roll this and would otherwise each need fixing
separately the next time, say, `get_meta()`'s return shape changes.

---

## Deep dive: live plotting in a popup Qt window instead of inline

### Why the current approach is slow

Every live-plot call in `plotter.py` (`update_imshow`, `update_full_imshow`,
`update_pcolormesh`, `update_twotone_imshow_voltage` and its siblings) ends
the same way:

```python
clear_output(wait=True)
display(self.imshow_fig)   # or pcolor_fig / twotone_fig
```

With the notebook's default inline backend, `display(fig)` doesn't show a
persistent window — it re-renders the *entire* figure to a fresh PNG (or SVG)
image, base64-encodes it, and sends it as a new output message over the
kernel's IOPub channel to the browser, which then decodes it and repaints the
output cell. `clear_output(wait=True)` additionally tells the frontend to
drop the previous image first. All of this happens synchronously in the same
Python thread that's running the measurement loop — the VNA/Yoko calls and
the plotting call share one thread, so the measurement genuinely waits for
the encode+transmit step to finish before continuing.

Two things make this worse than a fixed per-call cost:

- **The image being encoded grows over the course of the sweep.**
  `update_full_imshow`/`update_pcolormesh` accumulate one more row into
  `imshow_z`/`pcolor_z` every call, so the PNG being generated and shipped to
  the browser gets larger as the measurement progresses — meaning plotting
  overhead isn't constant, it *increases* over a long sweep, right when
  you'd want it to matter least.
- **It happens once per bias point (or once per average),** so for a flux
  map or two-tone sweep with hundreds of points, this overhead is paid
  hundreds of times, on top of whatever the instrument communication itself
  costs.

So yes — for sweeps with many points, this can plausibly become comparable
to, or larger than, the actual instrument acquisition time, exactly as
described. It's a well-known Jupyter/matplotlib-inline pain point, not
specific to this codebase.

### Proposed design: a persistent, native Qt window

The codebase already has everything needed for this — `analysis.py`
already builds `PyQt6` dialogs with an embedded matplotlib canvas
(`FigureCanvasQTAgg`) for its interactive widgets, so this isn't a new
dependency, just a new (simpler) use of one already in `requirements`.

Key idea: **keep the same `Plotter` method names and signatures** (so
`FluxMapExperiment`/`TwoToneExperiment` call sites don't change at all — this
is purely an internal swap), but change what happens at the very end of each
`update_*` method from "encode + display inline" to "update the existing
artist + repaint a native window that's already open":

```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class LiveFigureWindow(QMainWindow):
    """A single always-on-top window holding one matplotlib Figure,
    used instead of notebook inline display for live plotting."""

    def __init__(self, title: str = "Live plot"):
        # Reuse a running QApplication if one exists (e.g. from analysis.py's
        # dialogs); otherwise create one. No IPython magic, no global
        # backend switch — this window is independent of whatever backend
        # the rest of the notebook uses for its own inline plots.
        self.app = QApplication.instance() or QApplication(sys.argv)
        super().__init__()
        self.setWindowTitle(title)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        self.resize(900, 600)
        self.show()
        self.app.processEvents()   # make sure the window actually appears now

    def refresh(self):
        """Call after mutating self.figure's artists (set_data/set_clim/...)."""
        self.canvas.draw_idle()
        self.app.processEvents()   # pump the Qt event loop without blocking
```

And `Plotter.update_full_imshow` becomes (only the last few lines change):

```python
def update_full_imshow(self, x, y, z):
    if self.imshow_fig is None:
        self.imshow_x, self.imshow_y = x, y
        self.imshow_z = np.full((len(x), len(y)), np.nan)
        for i, row in enumerate(z):
            self.imshow_z[i] = row

        self._live_window = LiveFigureWindow("Flux map (live)")
        self.imshow_fig = self._live_window.figure
        self.imshow_ax = self.imshow_fig.add_subplot(111)
        self.imshow_im = self.imshow_ax.imshow(
            np.flip(self.imshow_z.T, axis=0),
            extent=[x.flat[0], x.flat[-1], y.min(), y.max()],
            cmap='viridis', aspect='auto', interpolation='none')
    else:
        for i, row in enumerate(z):
            self.imshow_z[i] = row
        self.imshow_im.set_data(np.flip(self.imshow_z.T, axis=0))
        self.imshow_ax.relim()
        self.imshow_ax.autoscale_view()

    self._live_window.refresh()   # replaces clear_output(wait=True) + display(...)
```

Everything about *what* gets computed and drawn (`set_data`, `relim`,
`autoscale_view`) is unchanged — `update_full_imshow` already does the
efficient thing of mutating the existing `AxesImage` rather than rebuilding
it (this part of the code was already good). The only thing being replaced
is the "publish this frame" step, from a notebook-display round trip to a
direct repaint of a window that's already on screen. `update_pcolormesh` is
the one exception worth fixing at the same time — it currently calls
`self.pcolor_ax.pcolormesh(...)` fresh on every update (rebuilding the mesh
artist and the colorbar from scratch) instead of mutating in place; that's
worth switching to `set_array`/keeping a persistent `QuadMesh` regardless of
the plotting backend, since it's wasteful either way.

### Will this actually be faster?

Yes, and by a large margin for anything beyond a handful of points, for two
independent reasons:

1. **No image encoding/transport.** `canvas.draw_idle()` repaints directly
   into the Qt window's own backing store, in-process — there's no PNG
   encode, no base64, no IPC message to a browser, no browser-side decode
   and DOM update. `QApplication.processEvents()` just pumps pending Qt
   events (including the paint event) without blocking on anything external.
2. **No growth penalty tied to notebook transport.** matplotlib still has to
   redraw the (growing) image data either way, but that in-process redraw is
   the cheap part; it's the notebook display protocol that scales badly with
   image size. Removing that removes the "gets slower as the sweep
   progresses" effect.

As a rule of thumb, this class of fix (swap `clear_output`+`display` for a
native-window `draw_idle`/`processEvents`) commonly turns a 50–500ms-per-frame
inline update into low tens-of-ms or less, but the exact number depends on
image size, sweep length, and machine — **measure it**: wrap the existing
call and the new call each in a `time.perf_counter()` delta on the same
sweep and compare, rather than trusting an estimate. If per-frame cost is
still non-trivial after this change, the cheapest next lever, independent of
backend, is throttling: only call `refresh()` every _k_-th bias point, or
rate-limit to a fixed frequency (e.g. skip the refresh if less than 100ms
have passed since the last one) — plotting a running sweep at 10–20 Hz looks
just as "live" to a human as plotting every single point, and this is a
one-line change wherever `refresh()`/`update_*` is called from the
measurement loop.

### Optional, bigger step: fully decouple plotting from acquisition

The design above still runs plotting on the *same thread* as the measurement
loop — it's much cheaper per call now, but the measurement still technically
pauses for that cheaper call. If profiling after the above shows plotting is
still a meaningful fraction of total time, the next lever is moving plotting
off the acquisition thread entirely:

- Run the acquisition loop (VNA/Yoko calls) in the main thread as today, but
  instead of calling `plotter.update_*(...)` directly, push the new row onto
  a `queue.Queue` (a `put_nowait` call costs microseconds).
- Run a `QTimer` on a background Qt event loop (or keep Qt's loop on the main
  thread and run acquisition in a `threading.Thread` — Qt widgets must only
  ever be touched from the thread that owns them, so pick one of these two
  arrangements deliberately, don't mix) that periodically drains the queue
  and calls `refresh()` at its own pace, e.g. every 100ms regardless of how
  fast data is arriving.
- This guarantees the acquisition loop is never blocked by rendering, at the
  cost of real concurrency: a queue, a decision about which thread owns the
  Qt event loop, and correspondingly more to reason about when something
  goes wrong.

Treat this as a follow-up only if Step 11's simpler fix isn't enough in
practice — it trades a meaningful amount of simplicity for a further speedup,
which is the wrong trade to make preemptively. Measure after Step 11 before
deciding whether Step 11b is worth it.
