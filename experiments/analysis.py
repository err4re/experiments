from typing import Type, Tuple, Union, List, Optional

import numpy as np
from qutip import Qobj, tensor, qeye, destroy

from scipy.optimize import curve_fit, minimize_scalar
from scipy.constants import hbar, elementary_charge, h

import re

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QSlider, QPushButton,
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from IPython import get_ipython
import matplotlib.pyplot as plt

import time
from tqdm.notebook import tqdm


from experiments.experiment_data import FluxMapData, TwoToneData
from experiments.plotter import Plotter
from experiments.utils import S_to_dBm

from resonator import shunt, see, background
import lmfit


### Hamiltonians

def hamiltonian_sym(Ec : float, Ej : float, phi_x : float, N : int, ng : float) -> Type[Qobj]:
    """
    Calculates the Hamiltonian for a symmetric SQUID (Superconducting Quantum Interference Device) 
    and returns it as a Qobj instance, which is suitable for quantum mechanical calculations.

    Parameters:
    - Ec (float): Charging energy.
    - Ej (float): Josephson energy.
    - phi_x (float): External magnetic flux through the SQUID, in units of the flux quantum (phi_0).
    - N (int): Number of charge states to include on either side of the charge neutrality point.
    - ng (float): Offset charge, controlling the effective charge of the superconducting island.

    Returns:
    - Qobj: Quantum object representing the Hamiltonian of the symmetric SQUID.

    Note:
    The Hamiltonian includes capacitive energy terms and Josephson junction energy terms, 
    considering the effects of an applied magnetic flux.
    """
    mc = np.diag(Ec * (np.arange(-N, N + 1) - ng) ** 2)
    mj =  0.5 * Ej * np.cos(phi_x/2) * (np.diag(-np.ones(2 * N), 1) + np.diag(-np.ones(2 * N), -1))

    m = mc + mj

    return Qobj(m)


def hamiltonian_asym(Ec : float, Ej : float, d : float, phi_x : float, N : int, ng : float) -> Type[Qobj]:
    """
    Calculates the Hamiltonian for an asymmetric SQUID (Superconducting Quantum Interference Device) 
    and returns it as a Qobj instance. The asymmetry is introduced via the parameter 'd', 
    which differentiates the Josephson junction energies.

    Parameters:
    - Ec (float): Charging energy.
    - Ej (float): Josephson energy.
    - d (float): Asymmetry parameter, defining the difference in Josephson energies between the two junctions.
    - phi_x (float): External magnetic flux through the SQUID, in units of the flux quantum (phi_0).
    - N (int): Number of charge states to include on either side of the charge neutrality point.
    - ng (float): Offset charge, controlling the effective charge of the superconducting island.

    Returns:
    - Qobj: Quantum object representing the Hamiltonian of the asymmetric SQUID.

    Note:
    The Hamiltonian includes capacitive energy terms, symmetric Josephson junction energy terms,
    and asymmetric terms that account for the difference in Josephson energies, 
    all influenced by an applied magnetic flux.
    """
    mc = np.diag(Ec * (np.arange(-N, N + 1) - ng) ** 2)
    mj =  0.5 * Ej * np.cos(phi_x/2) * (np.diag(-np.ones(2 * N), 1) + np.diag(-np.ones(2 * N), -1)) + 0.5 * d * Ej * np.sin(phi_x/2) * (-1j)* (np.diag(-np.ones(2 * N), 1) - np.diag(-np.ones(2 * N), -1))

    m = mc + mj

    return Qobj(m)

### Qutip solvers

def numerical_solution_sym(phi_x_values: Union[float, np.ndarray, List[float]], Ec: float, Ej: float, ng: float = 0, N: int = 20) -> np.ndarray:
    r"""
    Calculates the energy difference between the first two eigenstates of a sym. Squid for a given range or single value of external flux.
    
    Parameters:
    - phi_x_values (Union[float, np.ndarray, List[float]]): A single external flux value or an array/list of external flux (\(\phi_x\)) values.
    - Ec (float): The charging energy.
    - Ej (float): The Josephson energy.
    - ng (float): The offset charge (default is 0).
    - N (int): The number of charge states to consider (default is 20).
    
    Returns:
    - np.ndarray: An array of energy differences (\(E_1 - E_0\)) for each \(\phi_x\) value.
    """
    phi_x_array = np.atleast_1d(phi_x_values)  # Ensure phi_x_values is treated as an array
    
    # Initialize an array for the energy differences
    energy_differences = np.empty(phi_x_array.shape)
    
    # Compute the energy difference once for each phi_x value
    for i, phi_x in enumerate(phi_x_array):
        hamiltonian = hamiltonian_sym(Ec, Ej, phi_x, N, ng)
        energies = hamiltonian.eigenenergies()
        energy_diff = energies[1] - energies[0]
        energy_differences[i] = energy_diff
    
    return energy_differences

def numerical_solution_asym(phi_x_values: Union[float, np.ndarray, List[float]], Ec: float, Ej: float, d: float, ng: float = 0, N: int = 20) -> np.ndarray:
    r"""
    Calculates the energy difference between the first two eigenstates of an asym. Squid for a given range or single value of external flux.
    
    Parameters:
    - phi_x_values (Union[float, np.ndarray, List[float]]): A single external flux value or an array/list of external flux (\(\phi_x\)) values.
    - Ec (float): The charging energy.
    - Ej (float): The Josephson energy.
    - d (float): Squid asymmetry.
    - ng (float): The offset charge (default is 0).
    - N (int): The number of charge states to consider (default is 20).
    
    Returns:
    - np.ndarray: An array of energy differences (\(E_1 - E_0\)) for each \(\phi_x\) value.
    """
    phi_x_array = np.atleast_1d(phi_x_values)  # Ensure phi_x_values is treated as an array
    
    # Initialize an array for the energy differences
    energy_differences = np.empty(phi_x_array.shape)
    
    # Compute the energy difference once for each phi_x value
    for i, phi_x in enumerate(phi_x_array):
        hamiltonian = hamiltonian_asym(Ec, Ej, d, phi_x, N, ng)
        energies = hamiltonian.eigenenergies()
        energy_diff = energies[1] - energies[0]
        energy_differences[i] = energy_diff
    
    return energy_differences



### Tools for analysis

def find_separating_line(y_values):
    # Sort y values
    y_sorted = np.sort(y_values)
    
    # Compute differences between consecutive sorted y values
    diffs = np.diff(y_sorted)
    
    # Find index of the maximum gap
    max_gap_index = np.argmax(diffs)
    
    # Compute the y-value of the horizontal separating line
    y1 = y_sorted[max_gap_index]
    y2 = y_sorted[max_gap_index + 1]
    separating_y = (y1 + y2) / 2

    return separating_y

def find_separating_lines(y_values, drop_threshold=0.5):
    """
    Finds separating lines at midpoints of the largest vertical gaps in y_values.
    Stops when the next gap is significantly smaller than the previous (based on drop_threshold).
    
    Args:
        y_values: List or array of y-values (e.g., min S values in dBm).
        drop_threshold: Fractional drop between consecutive gap sizes (default = 0.5).
    
    Returns:
        List of y-values where horizontal lines should be placed.
    """
    y_sorted = np.sort(y_values)
    diffs = np.diff(y_sorted)

    # Pair gap size with index
    indexed_gaps = [(i, gap) for i, gap in enumerate(diffs)]
    # Sort by gap size descending
    indexed_gaps.sort(key=lambda x: x[1], reverse=True)

    separating_lines = []
    previous_gap = None

    for i, gap in indexed_gaps:
        if previous_gap is not None:
            ratio = gap / previous_gap
            if ratio < drop_threshold:
                break
        mid = (y_sorted[i] + y_sorted[i + 1]) / 2
        separating_lines.append(mid)
        previous_gap = gap

    # Sort lines for consistency
    separating_lines = np.sort(separating_lines)
    return separating_lines

def flux_period_qt_widget(
    fig: Figure = None,
    init_positions: Optional[dict] = None,
    resolution_fraction: float = 0.00001
) -> dict:
    """
    Interactive Qt dialog to adjust seven flux‐period vertical lines,
    with the ability to remove any you don’t need and to zoom/pan.

    Params
    ------
    fig : matplotlib.figure.Figure, optional
        A Figure with at least one Axes.  Either fig or ax must be provided.
    resolution_fraction : float
        Slider step = this fraction of the axis’ x‐span.

    Returns
    -------
    dict[str, float]
        Final x‐positions of the remaining lines, keyed by label:
        '-1.5 phi_0', '-1 phi_0', '-0.5 phi_0', '0 phi_0', '0.5 phi_0', '1 phi_0', '1.5 phi_0'.
    """

     # Ensure fig has an Axes
    if not fig.axes:
        raise ValueError("Figure must contain at least one Axes")
    ax = fig.axes[0]

    # get limits & compute scale
    xmin, xmax = ax.get_xlim()
    span = xmax - xmin
    step = span * resolution_fraction
    scale = 1.0 / step

    # define labels & positions
    # multipliers = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    if init_positions is None:
        multipliers = [-1.5, -0.5, 0.5, 1.5]
        labels      = [f"{m} phi_0" for m in multipliers]
        init_pos    = np.linspace(xmin, xmax, len(labels))
    else:
        init_pos = []
        labels = []
        for key, value in init_positions.items():
            init_pos.append(value)
            labels.append(key)

                                  

    # build dialog
    dlg = QDialog()
    dlg.setWindowTitle("Adjust & Remove Flux Lines")
    dlg._labels         = {}
    dlg._sliders        = {}
    dlg._remove_buttons = {}

    main_layout = QVBoxLayout(dlg)

    # Resize the dialog to 1200×700 pixels
    dlg.resize(1200, 700)

    # toolbar + canvas for zoom/pan
    canvas  = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    # draw lines
    lines = {}
    for lbl, pos in zip(labels, init_pos):
        lines[lbl] = ax.axvline(pos, label=lbl, lw=1, linestyle='dotted')
    # ax.legend(loc="upper right")

    # helper to remove a line
    def _remove(lbl):
        dlg._labels[lbl].hide()
        dlg._sliders[lbl].hide()
        dlg._remove_buttons[lbl].hide()
        lines[lbl].remove()
        canvas.draw_idle()
        dlg._labels        .pop(lbl)
        dlg._sliders       .pop(lbl)
        dlg._remove_buttons.pop(lbl)
        lines.pop(lbl)

    # sliders + remove buttons
    slider_row = QHBoxLayout()
    for lbl, pos in zip(labels, init_pos):
        lw = QLabel(lbl, parent=dlg)
        sl = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        sl.setRange(int(np.floor(xmin * scale)), int(np.ceil(xmax * scale)))
        sl.setValue(int(pos * scale))
        def _make_updater(name):
            def _upd(val):
                x0 = val / scale
                lines[name].set_xdata([x0, x0])
                canvas.draw_idle()
            return _upd
        sl.valueChanged.connect(_make_updater(lbl))

        rb = QPushButton("Remove", parent=dlg)
        rb.clicked.connect(lambda _, name=lbl: _remove(name))

        dlg._labels[lbl]         = lw
        dlg._sliders[lbl]        = sl
        dlg._remove_buttons[lbl] = rb

        col = QVBoxLayout()
        col.addWidget(lw)
        col.addWidget(sl)
        col.addWidget(rb)
        slider_row.addLayout(col)

    main_layout.addLayout(slider_row)

    # Done button
    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    # run
    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    # return remaining positions
    return {lbl: s.value() / scale for lbl, s in dlg._sliders.items()}


def find_voltage_to_flux_manual(
    data: FluxMapData, 
    init_positions: Optional[dict] = None,
    resolution_fraction: float = 0.00001
) -> dict :

    ### generalise for current or voltage on x-axis at some point

    fig, ax = Plotter.plot_flux_map(data, comment=False)


    ip = get_ipython()

    # 1) Close existing figures (important!)
    plt.close('all')

    # 2) Switch to external Qt windows
    ip.run_line_magic('matplotlib', 'qt')
    # get dictonary with flux symmetry points and their corresponding voltages
    flux_dict = flux_period_qt_widget(fig, init_positions, resolution_fraction)

    # 4) Close any Qt figures
    plt.close('all')

    # 5) Switch BACK to inline plotting
    ip.run_line_magic('matplotlib', 'inline')


    # lists to store extracted points and fit voltage to flux converion to
    fluxes = []
    voltages = []

    for key,value in flux_dict.items():
        #find first float with sign (or integer if no float is found)
        #flux in units of flux quanta
        match = re.search(r'[-+]?\d*\.\d+|\d+', key)

        if match:
            flux = float(match.group())
            if 'mV' in ax.get_xlabel():
                voltage = value*1e-3
            else:
                voltage = value

        else:
            raise Exception(f'Flux not found for key: {key}')
        
        fluxes.append(flux)
        voltages.append(voltage)

    # Fit linear function: flux = a * voltage + b
    def linear_func(v, a, b):
        return a * v + b

    popt, _ = curve_fit(linear_func, voltages, fluxes)
    a, b = popt  # a = slope (flux per voltage), b = offset

    voltage_to_flux_slope = a
    zero_flux = (0 - b) / a  # solve 0 = a * V + b → V = -b/a
    period = 1 / abs(a)      # ΔV for ΔΦ = 1

    data.fluxes = linear_func(data.voltages, *popt)

    Plotter.plot_flux_map_fluxes(data)

    print(f'Period: {period/1e-3} mV')
    print(f'Zero flux: {zero_flux/1e-3} mV')
    print(f'Voltage to flux slope: {voltage_to_flux_slope}')

    return period, zero_flux, voltage_to_flux_slope

def find_voltage_to_flux(data: FluxMapData, resolution_fraction: float = 0.0001, drop_threshold: float = 0.5):

    minima = [data.f[np.argmin(S_to_dBm(s))] for s in data.S]

    # frequency in the middle of the gap between upper and lower branches
    separating_freq = find_separating_line(minima)

    upper_branches_voltages = data.voltages[minima > separating_freq]
    lower_branches_voltages = data.voltages[minima < separating_freq]

    # drop_threshold controls how much smaller a gap may be than the previous
    # one and still count as a real cluster boundary (see find_separating_lines).
    # Lower it if a genuine flux-symmetry point near the edge of your sweep
    # is being dropped because its gap is smaller than the others.
    zero_flux_lines = find_separating_lines(upper_branches_voltages, drop_threshold=drop_threshold)
    pi_flux_lines = find_separating_lines(lower_branches_voltages, drop_threshold=drop_threshold)

    # smallest voltage value larger than 0 will be set as pi flux
    pi_flux_index = np.where(pi_flux_lines > 0, pi_flux_lines, np.inf).argmin()
    pi_flux = pi_flux_lines[pi_flux_index]

    multipliers = [0.5 + (i-pi_flux_index) for i, pi_flux_line in enumerate(pi_flux_lines)]
    labels = [f"{m} phi_0" for m in multipliers]

    init_positions = dict(zip(labels, pi_flux_lines/1e-3))

    # load computed initial positions and check manually
    return find_voltage_to_flux_manual(data, init_positions, resolution_fraction)

def frequency_spacing_qt_widget(
    fig: Figure,
    initial_positions: Optional[list] = None,
    resolution_fraction: float = 0.01
) -> dict:
    """
    Interactive Qt dialog to adjust a pair of horizontal lines on an existing Matplotlib Figure,
    with zoom/pan support.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A Figure with at least one Axes on which to overlay the lines.
    resolution_fraction : float, default=0.01
        Slider step as a fraction of the axis span (both x and y).

    Returns
    -------
    positions : dict[str, float]
        Final positions of the remaining lines, keyed by:
        'H bottom', 'H top'.
    """
 

    # Ensure fig has an Axes
    if not fig.axes:
        raise ValueError("Figure must contain at least one Axes")
    ax = fig.axes[0]

    # Determine axis limits and scales
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin
    span_y = ymax - ymin
    step_x = span_x * resolution_fraction
    step_y = span_y * resolution_fraction
    scale_x = 1.0 / step_x
    scale_y = 1.0 / step_y

    # Initial positions
    if initial_positions is not None:
        h_positions = (initial_positions[0], initial_positions[1])
    else:
        h_positions = (ymin, ymax)    

    # Labels
    h_labels = ["H bottom", "H top"]

    # Prepare dialog
    dlg = QDialog()
    dlg.setWindowTitle("Adjust Flux Intervals")
    dlg._sliders = {}
    dlg._labels = {}
    dlg._remove_pair_btns = {}

    main_layout = QVBoxLayout(dlg)

    # Resize the dialog to 1200×700 pixels
    dlg.resize(1200, 700)

    # Toolbar and canvas
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    # Plot initial lines
    h_lines = {
        "H bottom": ax.axhline(h_positions[0], lw=1, label="H bottom", color='red', linestyle='dotted'),
        "H top":    ax.axhline(h_positions[1], lw=1, label="H top", color='red', linestyle='dotted'),
    }
    # ax.legend(loc='upper right')

    # Sliders layout
    slider_row = QHBoxLayout()

    # Horizontal sliders
    for lbl, pos in zip(h_labels, h_positions):
        label_widget = QLabel(lbl, parent=dlg)
        slider = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        slider.setRange(int(np.floor(ymin * scale_y)), int(np.ceil(ymax * scale_y)))
        slider.setValue(int(pos * scale_y))
        slider.valueChanged.connect(lambda val, name=lbl: (
            h_lines[name].set_ydata([val/scale_y, val/scale_y]), canvas.draw_idle()
        ))
        dlg._labels[lbl] = label_widget
        dlg._sliders[lbl] = slider
        col = QVBoxLayout()
        col.addWidget(label_widget)
        col.addWidget(slider)
        slider_row.addLayout(col)

    main_layout.addLayout(slider_row)

    # Done button
    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    # Execute dialog
    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    # Collect positions (of horizontal lines)
    results = {}
    for lbl, slider in dlg._sliders.items():
        results[lbl] = slider.value() / scale_y

    return results

def find_spacing_manual(
    data: FluxMapData,
    initial_positions: Optional[list] = None,
    resolution_fraction: float = 0.00001
) -> float:
    """
    Manual frequency spacing selection: pops up a Qt widget to set two horizontal lines,
    computes the spacing, then re-plots the data inline with the lines and
    annotates the spacing on the graph with a double-headed arrow.
    Returns spacing in Hz.
    """
    # 1) Get the interval via Qt widget
    fig, ax = Plotter.plot_flux_map(data, comment=False)
    ip = get_ipython()
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    freq_dict = frequency_spacing_qt_widget(
        fig=fig,
        initial_positions=initial_positions,
        resolution_fraction=resolution_fraction
    )
    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    # 2) Read horizontal (frequency) band in Hz
    hbot_hz = freq_dict["H bottom"] * 1e9
    htop_hz = freq_dict["H top"] * 1e9

    # 3) Compute spacing
    spacing_hz = abs(htop_hz - hbot_hz)
    spacing_mhz = spacing_hz / 1e6
    print(f"Spacing: {spacing_mhz} MHz")

    # 4) Re-plot inline and annotate
    fig, ax = Plotter.plot_flux_map(data, title='Spacing Manual' ,comment=False)

    # draw horizontal lines (in GHz)
    hbot_ghz = freq_dict["H bottom"]
    htop_ghz = freq_dict["H top"]
    ax.axhline(hbot_ghz, lw=1, color='red', linestyle='dotted')
    ax.axhline(htop_ghz, lw=1, color='red', linestyle='dotted')

    # determine arrow x-position (e.g., 5% from left)
    xmin, xmax = ax.get_xlim()
    x_arrow = xmin + 0.05 * (xmax - xmin)

    # draw double-headed arrow between the two lines
    ax.annotate(
        '',
        xy=(x_arrow, htop_ghz), xytext=(x_arrow, hbot_ghz),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )

    # annotate the spacing next to the arrow
    y_mid = (htop_ghz + hbot_ghz) / 2
    ax.text(
        x_arrow + 0.01*(xmax-xmin), y_mid,
        f"{spacing_mhz:.2f} MHz",
        va='center', ha='left', color='black', fontsize=10
    )

    plt.show()
    return spacing_hz


def flux_interval_qt_widget(
    fig: Figure,
    num_intervals: int = 3,
    initial_positions: Optional[list] = None,
    resolution_fraction: float = 0.01
) -> dict:
    """
    Interactive Qt dialog to adjust a variable number of pairs of vertical lines
    and a pair of horizontal lines on an existing Matplotlib Figure,
    with zoom/pan support and the ability to remove entire vertical line pairs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A Figure with at least one Axes on which to overlay the lines.
    resolution_fraction : float, default=0.01
        Slider step as a fraction of the axis span (both x and y).
    num_intervals : int
        Number of intervals, or rather pairs of vertical lines

    Returns
    -------
    positions : dict[str, float]
        Final positions of the remaining lines, keyed by:
        'V1 left', 'V1 right', 'V2 left', 'V2 right', 'V3 left', 'V3 right', ...
        'H bottom', 'H top'.
    """
 

    # Ensure fig has an Axes
    if not fig.axes:
        raise ValueError("Figure must contain at least one Axes")
    ax = fig.axes[0]

    # Determine axis limits and scales
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin
    span_y = ymax - ymin
    step_x = span_x * resolution_fraction
    step_y = span_y * resolution_fraction
    scale_x = 1.0 / step_x
    scale_y = 1.0 / step_y

    # Initial positions
    if initial_positions is not None:
        boundaries_x = initial_positions
        num_intervals = len(initial_positions)//2
    else:
        boundaries_x = np.linspace(xmin+0.1*span_x, xmax-0.1*span_x, 2*num_intervals)
        
    v_pairs = [(boundaries_x[2*i], boundaries_x[2*i+1]) for i in range(num_intervals)]
    h_positions = (ymin, ymax)

    # Labels
    v_labels = [f"V{i+1} left" for i in range(num_intervals)] + [f"V{i+1} right" for i in range(num_intervals)]
    h_labels = ["H bottom", "H top"]

    # Prepare dialog
    dlg = QDialog()
    dlg.setWindowTitle("Adjust Flux Intervals")
    dlg._sliders = {}
    dlg._labels = {}
    dlg._remove_pair_btns = {}

    main_layout = QVBoxLayout(dlg)

    # Resize the dialog to 1200×700 pixels
    dlg.resize(1200, 700)

    # Toolbar and canvas
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    # Plot initial lines
    v_lines = {}
    for idx, (left, right) in enumerate(v_pairs, start=1):
        v_lines[f"V{idx} left"] = ax.axvline(left, lw=1, label=f"V{idx} left", linestyle='dotted')
        v_lines[f"V{idx} right"] = ax.axvline(right, lw=1, label=f"V{idx} right", linestyle='dotted')
    h_lines = {
        "H bottom": ax.axhline(h_positions[0], lw=1, label="H bottom", color='red', linestyle='dotted'),
        "H top":    ax.axhline(h_positions[1], lw=1, label="H top", color='red', linestyle='dotted'),
    }
    # ax.legend(loc='upper right')

    # Removal helper for vertical pairs
    def remove_v_pair(idx):
        # remove both lines, sliders, labels, and button
        left_lbl = f"V{idx} left"
        right_lbl = f"V{idx} right"
        for lbl in (left_lbl, right_lbl):
            dlg._labels[lbl].hide()
            dlg._sliders[lbl].hide()
            v_lines[lbl].remove()
            canvas.draw_idle()
            dlg._labels.pop(lbl)
            dlg._sliders.pop(lbl)
            v_lines.pop(lbl)
        # hide button
        btn = dlg._remove_pair_btns[idx]
        btn.hide()
        dlg._remove_pair_btns.pop(idx)
        # relayout not needed for simplicity

    # Sliders layout
    slider_row = QHBoxLayout()

    # Vertical sliders and remove-pair buttons
    for idx, (left, right) in enumerate(v_pairs, start=1):
        # Left slider
        lbl_l = QLabel(f"V{idx} left", parent=dlg)
        sl_l = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        sl_l.setRange(int(np.floor(xmin * scale_x)), int(np.ceil(xmax * scale_x)))
        sl_l.setValue(int(left * scale_x))
        sl_l.valueChanged.connect(lambda val, name=f"V{idx} left": (
            v_lines[name].set_xdata([val/scale_x, val/scale_x]), canvas.draw_idle()
        ))
        dlg._labels[lbl_l.text()] = lbl_l
        dlg._sliders[lbl_l.text()] = sl_l

        # Right slider
        lbl_r = QLabel(f"V{idx} right", parent=dlg)
        sl_r = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        sl_r.setRange(int(np.floor(xmin * scale_x)), int(np.ceil(xmax * scale_x)))
        sl_r.setValue(int(right * scale_x))
        sl_r.valueChanged.connect(lambda val, name=f"V{idx} right": (
            v_lines[name].set_xdata([val/scale_x, val/scale_x]), canvas.draw_idle()
        ))
        dlg._labels[lbl_r.text()] = lbl_r
        dlg._sliders[lbl_r.text()] = sl_r

        # Remove pair button
        btn = QPushButton(f"Remove V{idx}", parent=dlg)
        btn.clicked.connect(lambda _, i=idx: remove_v_pair(i))
        dlg._remove_pair_btns[idx] = btn

        # Column
        col = QVBoxLayout()
        col.addWidget(lbl_l)
        col.addWidget(sl_l)
        col.addWidget(lbl_r)
        col.addWidget(sl_r)
        col.addWidget(btn)
        slider_row.addLayout(col)

    # Horizontal sliders
    for lbl, pos in zip(h_labels, h_positions):
        label_widget = QLabel(lbl, parent=dlg)
        slider = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        slider.setRange(int(np.floor(ymin * scale_y)), int(np.ceil(ymax * scale_y)))
        slider.setValue(int(pos * scale_y))
        slider.valueChanged.connect(lambda val, name=lbl: (
            h_lines[name].set_ydata([val/scale_y, val/scale_y]), canvas.draw_idle()
        ))
        dlg._labels[lbl] = label_widget
        dlg._sliders[lbl] = slider
        col = QVBoxLayout()
        col.addWidget(label_widget)
        col.addWidget(slider)
        slider_row.addLayout(col)

    main_layout.addLayout(slider_row)

    # Done button
    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    # Execute dialog
    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    # Collect positions
    results = {}
    for lbl, slider in dlg._sliders.items():
        if lbl.startswith('V'):
            results[lbl] = slider.value() / scale_x
        else:
            results[lbl] = slider.value() / scale_y
    return results


def parabola(x,a,b,c):
    x = np.asarray(x)
    return a*x**2 + b*x + c

def fit_parabola_extremum(
    voltages: list[float],
    lsfs: list[shunt.LinearShuntFitter],
    fit_with_errors: bool = True
) -> Tuple[float, float]:
    
    frs = [lsf.f_r/1e9 for lsf in lsfs]
        
    if fit_with_errors:
        fr_errors = [lsf.f_r_error/1e9 for lsf in lsfs]
        popt, pcov = curve_fit(parabola, voltages, frs, sigma=fr_errors)
    else:
        popt, pcov = curve_fit(parabola, voltages, frs)

    # extract extremum by checking sign of a for a* x**2
    if popt[0] < 0:
        ext_index = np.argmax(parabola(voltages, *popt))
        # print('maximum')
    else:
        ext_index = np.argmin(parabola(voltages, *popt))
        # print('minimum')


    return voltages[ext_index], frs[ext_index]


def fit_resonances_voltage_intervals(
    data: FluxMapData,
    num_intervals: int = 3,
    initial_positions: Optional[list] = None,
    resolution_fraction: float = 0.0001,
) -> Tuple[List[List[float]], List[List[shunt.LinearShuntFitter]]]:
    """
    1) Ask user to set vertical & horizontal intervals via Qt widget.
    2) Bucket sweeps by voltage into the vertical intervals.
    3) For each sweep in a given bucket, restrict to the frequency band
       between H bottom and H top, then fit resonances—showing progress & ETA.
    """

    # 1) Get the intervals
    fig, ax = Plotter.plot_flux_map(data, comment=False)
    ip = get_ipython()
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    flux_dict = flux_interval_qt_widget(
        fig=fig,
        num_intervals=num_intervals,
        initial_positions=initial_positions,
        resolution_fraction=resolution_fraction
    )
    
    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    # 2) Build vertical windows
    vert_intervals = []

    if 'mV' in ax.get_xlabel():
        voltage_units = 1e-3
    else:
        voltage_units = 1

    # 1) Find all indices i for which both `V{i} left` and `V{i} right` exist
    indices = sorted(
        int(key[1:key.find(' ')])       # grab the number after 'V'
        for key in flux_dict
        if key.startswith('V') and 'left' in key
    )

    for i in indices:
        l = flux_dict.get(f"V{i} left") * voltage_units; r = flux_dict.get(f"V{i} right") * voltage_units
        if l is None or r is None: 
            continue
        vert_intervals.append((min(l, r), max(l, r)))

    # 3) Read horizontal (frequency) band
    hbot = flux_dict["H bottom"] * 1e9
    htop = flux_dict["H top"] * 1e9
    f = np.asarray(data.f)
    hmask = (f >= min(hbot, htop)) & (f <= max(hbot, htop))
    if not hmask.any():
        raise RuntimeError(f"No points between {hbot} and {htop}")

    # 4) Prepare buckets
    voltages_per_interval = [[] for _ in vert_intervals]
    S_per_interval = [[] for _ in vert_intervals]
    fits_per_interval     = [[] for _ in vert_intervals]

    # Pre‐filter the sweeps we will actually fit
    fit_sweeps = [
        (voltage, s21)
        for voltage, s21 in zip(data.voltages, data.S)
        if any(lv <= voltage <= rv for lv, rv in vert_intervals)
    ]

    total = len(fit_sweeps)
    start = time.time()
    pbar = tqdm(total=total, desc="Fitting sweeps", unit="sweep")

    # 5) Fit the resonances

    for idx_sweep, (voltage, s21) in enumerate(fit_sweeps, start=1):
        # restrict to the frequency band
        freq_band = f[hmask]
        s21_band  = s21[hmask]

        # time the fit
        t0 = time.time()
        # find and fit into its interval
        for idx_int, (vl, vr) in enumerate(vert_intervals):
            if vl <= voltage <= vr:
                lsf = shunt.LinearShuntFitter(
                    frequency=freq_band,
                    data=s21_band,
                    background_model=background.MagnitudeSlopeOffsetPhaseDelay()
                )
                fits_per_interval[idx_int].append(lsf)
                S_per_interval[idx_int].append(s21)
                voltages_per_interval[idx_int].append(voltage)
                break
        dt = (time.time() - t0)

        # update progress and ETA
        elapsed = time.time() - start
        avg = elapsed / idx_sweep
        eta = avg * (total - idx_sweep)
        pbar.set_postfix_str(f"{dt*1000:.1f} ms/sweep, ETA {eta:.1f}s")
        pbar.update(1)

    pbar.close()

    return voltages_per_interval, S_per_interval, fits_per_interval


def find_spacing(
    data: FluxMapData,
    num_intervals: int = 3,
    initial_positions: Optional[list] = None,
    resolution_fraction: float = 0.0001,
    fit_with_errors: bool = False
) -> float:
    
    ### 1. fit upper branches to get pi flux point

    # initial position based on voltage to flux conversion
    if data.fluxes is not None:
        flux_min = np.min(data.fluxes)
        flux_max = np.max(data.fluxes)

        pi_fluxes = np.arange(np.ceil(flux_min * 2) / 2, np.floor(flux_max * 2) / 2 + 0.5, 0.5)
        pi_fluxes = pi_fluxes[pi_fluxes % 1 != 0]

        initial_flux_positions = [pi_flux + delta for pi_flux in pi_fluxes for delta in (-0.05, 0.05)]
        initial_flux_indices = [np.argmin(np.abs(np.array(data.fluxes) - v)) for v in initial_flux_positions]

        initial_voltage_positions = [data.voltages[idx] for idx in initial_flux_indices]
        initial_positions = np.array(initial_voltage_positions)/1e-3
        

    voltages_upper, S_upper, lsfs_upper = fit_resonances_voltage_intervals(data, num_intervals, initial_positions, resolution_fraction)

    pi_voltages = []
    pi_freqs = []

    # iterate over the extracted upper intervals
    for voltages, lsfs in zip(voltages_upper, lsfs_upper):
        
        pi_voltage, pi_freq = fit_parabola_extremum(voltages, lsfs, fit_with_errors)
        pi_voltages.append(pi_voltage)
        pi_freqs.append(pi_freq)


    
    ### 2. fit lower branches to get 0 flux point

    # initial position based on voltage to flux conversion
    if data.fluxes is not None:
        flux_min = np.min(data.fluxes)
        flux_max = np.max(data.fluxes)

        zero_fluxes = np.arange(np.ceil(flux_min), np.floor(flux_max)+1, 1, dtype=int)

        initial_flux_positions = [zero_flux + delta for zero_flux in zero_fluxes for delta in (-0.18, 0.18)]
        initial_flux_indices = [np.argmin(np.abs(np.array(data.fluxes) - v)) for v in initial_flux_positions]

        initial_voltage_positions = [data.voltages[idx] for idx in initial_flux_indices]
        initial_positions = np.array(initial_voltage_positions)/1e-3

    voltages_lower, S_lower, lsfs_lower = fit_resonances_voltage_intervals(data, num_intervals, initial_positions, resolution_fraction)

    zero_voltages = []
    zero_freqs = []


    # iterate over the extracted lower intervals
    for voltages, lsfs in zip(voltages_lower, lsfs_lower):
        
        zero_voltage, zero_freq = fit_parabola_extremum(voltages, lsfs, fit_with_errors)
        zero_voltages.append(zero_voltage)
        zero_freqs.append(zero_freq)


    ### 3. compute spacing
    
    spacing = np.mean(pi_freqs) - np.mean(zero_freqs)
    print(f'Spacing: {spacing*1e3} MHz')


    ### 4. plot results

    fig, ax = Plotter.plot_flux_map_voltage(data)

    for voltages, lsfs in zip(voltages_upper, lsfs_upper):
        plt.scatter([voltage/1e-3 for voltage in voltages], [lsf.f_r/1e9 for lsf in lsfs], s=0.5, alpha=0.75, color='green')
        
    for voltages, lsfs in zip(voltages_lower, lsfs_lower):
        plt.scatter([voltage/1e-3 for voltage in voltages], [lsf.f_r/1e9 for lsf in lsfs], s=0.5, alpha=0.75, color='orange')

    plt.scatter(np.asarray(pi_voltages)/1e-3, pi_freqs, alpha=0.5, color='red')
    plt.scatter(np.asarray(zero_voltages)/1e-3, zero_freqs, alpha=0.5, color='red')

    # determine arrow x-position (e.g., 5% from left)
    xmin, xmax = ax.get_xlim()
    x_arrow = xmin + 0.05 * (xmax - xmin)

    htop_ghz = pi_freq
    hbot_ghz = zero_freq
    spacing_mhz = spacing*1e3

    # draw double-headed arrow between the two lines
    ax.annotate(
        '',
        xy=(x_arrow, htop_ghz), xytext=(x_arrow, hbot_ghz),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2)
    )

    # annotate the spacing next to the arrow
    y_mid = (htop_ghz + hbot_ghz) / 2
    ax.text(
        x_arrow + 0.01*(xmax-xmin), y_mid,
        f"{spacing_mhz:.2f} MHz",
        va='center', ha='left', color='black', fontsize=10
    )

    plt.show()


    # check fitting of resonances at 0 and pi flux

    # — pick out the single fit nearest each extremum —
    pi_closest = []        # list of (voltage, lsf) for each pi‐interval
    for volts, S, lsfs, pv in zip(voltages_upper, S_upper, lsfs_upper, pi_voltages):
        arr   = np.array(volts)
        idx   = np.abs(arr - pv).argmin()
        pi_closest.append((arr[idx], S[idx], lsfs[idx]))

    zero_closest = []      # list of (voltage, lsf) for each zero‐interval
    for volts, S, lsfs, zv in zip(voltages_lower, S_lower, lsfs_lower, zero_voltages):
        arr   = np.array(volts)
        idx   = np.abs(arr - zv).argmin()
        zero_closest.append((arr[idx], S[idx], lsfs[idx]))


    v0, s0, lsf0 = zero_closest[0]

    fig, axes = see.triptych(lsf0, figure_settings={'figsize': (12, 5)}, frequency_scale=1e-9)
    axes[0].vlines(np.mean(zero_freqs), *axes[0].get_ylim(), color='red', lw=1)
    fig.suptitle(r"Fit at $\phi_\text{ext} = 0 \phi_0$")
    plt.show()


    # and similarly for the first pi‐flux point:
    vpi, spi, lsfpi = pi_closest[0]

    fig, axes = see.triptych(lsfpi, figure_settings={'figsize': (12, 5)}, frequency_scale=1e-9)
    axes[0].vlines(np.mean(pi_freqs), *axes[0].get_ylim(), color='red', lw=1)
    fig.suptitle(r"Fit at $\phi_\text{ext} = 0.5 \phi_0$")
    plt.show()
    

    return spacing, (v0, lsf0), (vpi, lsfpi)


def perturbative_resonator_shift_sym(f_r, E_C, E_J, lambda_c, flux=0, n_g=0, i=0, n_max=10):
    # i = initial state, default is ground state

    # Charge‐basis truncation: include n = -n_max … +n_max
    n_vals   = np.arange(-n_max, n_max+1)
    Nc_dim   = len(n_vals)

    # ─── Build CPB operators in charge basis ───────────────────────────────────────
    # Charge operator: diagonal in the charge basis
    n_op     = Qobj(np.diag(n_vals))

    # Charge‐shift operator: exp(+i φ) shifts n→n+1
    shift_up = Qobj(np.roll(np.eye(Nc_dim), -1, axis=1))
    # cosφ = (e^{iφ} + e^{-iφ})/2
    cos_phi  = 0.5 * (shift_up + shift_up.dag())

    # fluxtunable CPB Hamiltonian: E_C/2 (n-n_g)² - E_J * | cos(φx/(2*φ0)) | * cosφ
    H_cpb: Qobj  = E_C/2 * (n_op - n_g * qeye(Nc_dim))**2 - E_J *  np.abs(np.cos(np.pi * flux)) * cos_phi

    # ─── Diagonalize ──────────────────────────────────────────────────────────────
    eigs, states = H_cpb.eigenstates()   # eigs: list of eigenenergies; states: list of Qobj

    resonator_frequency_shift = 0

    for j,state_j in enumerate(states):
        
        if i != j:
            matrix_element = lambda_c * E_C * np.abs( state_j.dag()*n_op*states[i] ) # GHz

            frequency_shift_j = - matrix_element**2 * ( 1/( np.abs( eigs[j] - eigs[i] ) - f_r ) + 1/( np.abs( eigs[j] - eigs[i] ) + f_r )) # GHz

            resonator_frequency_shift += frequency_shift_j # GHz

    return resonator_frequency_shift

def perturbative_resonator_shift_asym(f_r, E_C, E_J1, E_J2, lambda_c, flux=0, n_g=0, i=0, n_max=10):
    # i = initial state, default is ground state

    # Charge‐basis truncation: include n = -n_max … +n_max
    n_vals   = np.arange(-n_max, n_max+1)
    Nc_dim   = len(n_vals)

    # ─── Build CPB operators in charge basis ───────────────────────────────────────
    # Charge operator: diagonal in the charge basis
    n_op     = Qobj(np.diag(n_vals))

    # Charge‐shift operator: exp(+i φ) shifts n→n+1
    shift_up = Qobj(np.roll(np.eye(Nc_dim), -1, axis=1))
    # cosφ = (e^{iφ} + e^{-iφ})/2, sinφ = (e^{iφ} - e^{-iφ})/(2i)
    cos_phi  = 0.5 * (shift_up + shift_up.dag())
    sin_phi  = -0.5j * (shift_up - shift_up.dag())

    # Asymmetric SQUID: total Josephson energy and asymmetry parameter,
    # same convention as hamiltonian_asym's (Ej, d).
    E_J = E_J1 + E_J2
    d   = (E_J1 - E_J2) / (E_J1 + E_J2)

    # fluxtunable asymmetric CPB Hamiltonian: E_C/2 (n-n_g)²
    # - E_J * |cos(φx/2)| * cosφ - d*E_J * sin(φx/2) * sinφ, with φx/2 = π*flux.
    # The sinφ term (absent for a symmetric SQUID) is what the junction
    # mismatch actually couples in; don't abs() it, its sign must flip across
    # zero flux for the asymmetry to be physical.
    H_cpb: Qobj  = E_C/2 * (n_op - n_g * qeye(Nc_dim))**2 \
                   - E_J * np.abs(np.cos(np.pi * flux)) * cos_phi \
                   - d * E_J * np.sin(np.pi * flux) * sin_phi

    # ─── Diagonalize ──────────────────────────────────────────────────────────────
    eigs, states = H_cpb.eigenstates()   # eigs: list of eigenenergies; states: list of Qobj

    resonator_frequency_shift = 0

    for j,state_j in enumerate(states):

        if i != j:
            matrix_element = lambda_c * E_C * np.abs( state_j.dag()*n_op*states[i] ) # GHz

            frequency_shift_j = - matrix_element**2 * ( 1/( np.abs( eigs[j] - eigs[i] ) - f_r ) + 1/( np.abs( eigs[j] - eigs[i] ) + f_r )) # GHz

            resonator_frequency_shift += frequency_shift_j # GHz

    return resonator_frequency_shift


def qubit_transition_frequency_sym(E_C, E_J, flux=0, n_g=0, k=0, l=1, n_max=10):
    """
    Bare k -> l qubit transition frequency |E_l - E_k|, from the same CPB
    Hamiltonian used by perturbative_resonator_shift_sym and
    perturbative_qubit_shift_sym, for a symmetric SQUID. Add
    perturbative_qubit_shift_sym(...) to this to get the n-photon-shifted
    transition frequency actually measured in two-tone spectroscopy.
    """

    # Charge‐basis truncation: include n = -n_max … +n_max
    n_vals   = np.arange(-n_max, n_max+1)
    Nc_dim   = len(n_vals)

    n_op     = Qobj(np.diag(n_vals))
    shift_up = Qobj(np.roll(np.eye(Nc_dim), -1, axis=1))
    cos_phi  = 0.5 * (shift_up + shift_up.dag())

    H_cpb: Qobj = E_C/2 * (n_op - n_g * qeye(Nc_dim))**2 - E_J * np.abs(np.cos(np.pi * flux)) * cos_phi

    eigs = H_cpb.eigenenergies()

    return np.abs(eigs[l] - eigs[k])


def perturbative_qubit_shift_sym(f_r, E_C, E_J, lambda_c, flux=0, n_g=0, k=0, l=1, n=0, n_max=10):
    r"""
    Dispersive shift \delta\omega_{kl}^{(n)} of the k -> l qubit transition
    frequency when the resonator is populated with n photons, for a
    symmetric SQUID. Companion to perturbative_resonator_shift_sym (which
    gives the resonator's photon-independent shift instead), built from
    the same CPB Hamiltonian and coupling matrix elements:

        delta_omega_kl^(n) = (1/2) * [T(l) - T(k)] + n * [D(k) - D(l)]

    where, for eigenstate m with energy E_m,
        T(m) = sum_{j!=m} g_jm^2 / (|E_j - E_m| - f_r)
        D(m) = sum_{j!=m} g_jm^2 * [1/(|E_j - E_m| - f_r) + 1/(|E_j - E_m| + f_r)]
    and g_jm = lambda_c * E_C * |<j| n_op |m>|.

    Parameters
    ----------
    k, l : int
        Indices of the two qubit eigenstates whose transition frequency is
        being shifted (k=0, l=1 is the qubit's own 0->1 transition).
    n : int
        Photon number in the resonator.

    See perturbative_resonator_shift_sym for f_r, E_C, E_J, lambda_c, flux,
    n_g, n_max.
    """

    # Charge‐basis truncation: include n = -n_max … +n_max
    n_vals   = np.arange(-n_max, n_max+1)
    Nc_dim   = len(n_vals)

    # ─── Build CPB operators in charge basis ───────────────────────────────────────
    n_op     = Qobj(np.diag(n_vals))
    shift_up = Qobj(np.roll(np.eye(Nc_dim), -1, axis=1))
    cos_phi  = 0.5 * (shift_up + shift_up.dag())

    # fluxtunable CPB Hamiltonian: E_C/2 (n-n_g)² - E_J * |cos(φx/(2*φ0))| * cosφ
    H_cpb: Qobj = E_C/2 * (n_op - n_g * qeye(Nc_dim))**2 - E_J * np.abs(np.cos(np.pi * flux)) * cos_phi

    # ─── Diagonalize ──────────────────────────────────────────────────────────────
    eigs, states = H_cpb.eigenstates()   # eigs: list of eigenenergies; states: list of Qobj

    def state_sums(state_index):
        lamb_sum = 0
        dispersive_sum = 0
        for j, state_j in enumerate(states):
            if j == state_index:
                continue
            g_sq = (lambda_c * E_C * np.abs(state_j.dag() * n_op * states[state_index]))**2 # GHz^2
            detuning = np.abs(eigs[j] - eigs[state_index]) # GHz
            lamb_sum += g_sq * (1/(detuning - f_r)) # GHz
            dispersive_sum += g_sq * (1/(detuning - f_r) + 1/(detuning + f_r)) # GHz
        return lamb_sum, dispersive_sum

    lamb_l, dispersive_l = state_sums(l)
    lamb_k, dispersive_k = state_sums(k)

    lamb_shift = 0.5 * (lamb_l - lamb_k) # GHz, n-independent (Lamb-shift-like) part
    photon_shift = n * (dispersive_k - dispersive_l) # GHz, per-photon dispersive part

    return lamb_shift + photon_shift


def perturbative_qubit_shift_asym(f_r, E_C, E_J1, E_J2, lambda_c, flux=0, n_g=0, k=0, l=1, n=0, n_max=10):
    r"""
    Dispersive shift \delta\omega_{kl}^{(n)} of the k -> l qubit transition
    frequency when the resonator is populated with n photons, for an
    asymmetric SQUID. Companion to perturbative_resonator_shift_asym; see
    perturbative_qubit_shift_sym for the formula and the meaning of k, l, n.
    """

    # Charge‐basis truncation: include n = -n_max … +n_max
    n_vals   = np.arange(-n_max, n_max+1)
    Nc_dim   = len(n_vals)

    # ─── Build CPB operators in charge basis ───────────────────────────────────────
    n_op     = Qobj(np.diag(n_vals))
    shift_up = Qobj(np.roll(np.eye(Nc_dim), -1, axis=1))
    cos_phi  = 0.5 * (shift_up + shift_up.dag())
    sin_phi  = -0.5j * (shift_up - shift_up.dag())

    # Asymmetric SQUID: total Josephson energy and asymmetry parameter,
    # same convention as hamiltonian_asym's (Ej, d).
    E_J = E_J1 + E_J2
    d   = (E_J1 - E_J2) / (E_J1 + E_J2)

    H_cpb: Qobj = E_C/2 * (n_op - n_g * qeye(Nc_dim))**2 \
                  - E_J * np.abs(np.cos(np.pi * flux)) * cos_phi \
                  - d * E_J * np.sin(np.pi * flux) * sin_phi

    eigs, states = H_cpb.eigenstates()

    def state_sums(state_index):
        lamb_sum = 0
        dispersive_sum = 0
        for j, state_j in enumerate(states):
            if j == state_index:
                continue
            g_sq = (lambda_c * E_C * np.abs(state_j.dag() * n_op * states[state_index]))**2
            detuning = np.abs(eigs[j] - eigs[state_index])
            lamb_sum += g_sq * (1/(detuning - f_r))
            dispersive_sum += g_sq * (1/(detuning - f_r) + 1/(detuning + f_r))
        return lamb_sum, dispersive_sum

    lamb_l, dispersive_l = state_sums(l)
    lamb_k, dispersive_k = state_sums(k)

    lamb_shift = 0.5 * (lamb_l - lamb_k)
    photon_shift = n * (dispersive_k - dispersive_l)

    return lamb_shift + photon_shift


def perturbative_resonator_shift_sym_qt_widget(
    fig: Figure,
    initial_params: dict,
    param_ranges: Optional[dict] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
    num_steps: int = 1000,
) -> dict:
    """
    Interactive Qt dialog with sliders for f_r, E_C, E_J, lambda_c and n_g,
    overlaying the perturbative_resonator_shift_sym spectrum on an existing Matplotlib
    Figure (e.g. a two-tone flux map, flux on x, frequency in GHz on y) and
    updating the curve live as the sliders move.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A Figure with at least one Axes to overlay the fitted spectrum on.
    initial_params : dict
        Starting values, keyed by 'f_r', 'E_C', 'E_J', 'lambda_c', 'n_g'
        (f_r, E_C, E_J in GHz, n_g dimensionless).
    param_ranges : dict, optional
        (min, max) slider bounds per key above. Any key left out defaults
        to +/-50% of its initial value ('n_g' defaults to [-0.5, 0.5]).
    flux_vals : np.ndarray, optional
        Flux points (in units of Phi_0) to evaluate the curve at. Defaults
        to 201 points spanning the Axes' current x-limits.
    n_max : int
        Charge-basis truncation passed to perturbative_resonator_shift_sym.
    i : int
        Initial state index passed to perturbative_resonator_shift_sym.
    num_steps : int
        Slider resolution (number of discrete steps) per parameter.

    Returns
    -------
    dict[str, float]
        Final tuned values, keyed like `initial_params`.
    """

    if not fig.axes:
        raise ValueError("Figure must contain at least one Axes")
    ax = fig.axes[0]

    keys = ['f_r', 'E_C', 'E_J', 'lambda_c', 'n_g']
    for key in keys:
        if key not in initial_params:
            raise KeyError(f"initial_params is missing '{key}'")

    if flux_vals is None:
        xmin, xmax = ax.get_xlim()
        flux_vals = np.linspace(xmin, xmax, 201)

    default_ranges = {key: (0.5 * initial_params[key], 1.5 * initial_params[key]) for key in keys}
    default_ranges['n_g'] = (-0.5, 0.5)

    ranges = {**default_ranges, **(param_ranges or {})}
    # normalise ranges in case a negative initial value flipped (lo, hi)
    ranges = {key: (min(lo, hi), max(lo, hi)) for key, (lo, hi) in ranges.items()}

    values = dict(initial_params)

    def compute_curve():
        dfrs = np.array([
            perturbative_resonator_shift_sym(
                values['f_r'], values['E_C'], values['E_J'], values['lambda_c'],
                flux=flux, n_g=values['n_g'], i=i, n_max=n_max
            )
            for flux in flux_vals
        ])
        return values['f_r'] + dfrs

    def slider_to_value(key, slider_val):
        lo, hi = ranges[key]
        return lo + (slider_val / num_steps) * (hi - lo)

    def value_to_slider(key, value):
        lo, hi = ranges[key]
        if hi == lo:
            return 0
        return int(round((value - lo) / (hi - lo) * num_steps))

    # build dialog
    dlg = QDialog()
    dlg.setWindowTitle("Tune Perturbative Shift Fit")
    dlg._sliders = {}
    dlg._value_labels = {}

    main_layout = QVBoxLayout(dlg)
    dlg.resize(1200, 800)

    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    (line,) = ax.plot(flux_vals, compute_curve(), color='red', lw=1.5, label='Fit')
    ax.legend(loc='upper right')

    def make_updater(key, value_label):
        def _upd(slider_val):
            values[key] = slider_to_value(key, slider_val)
            value_label.setText(f"{key} = {values[key]:.5g}")
            line.set_ydata(compute_curve())
            canvas.draw_idle()
        return _upd

    slider_row = QHBoxLayout()
    for key in keys:
        label = QLabel(f"{key} = {values[key]:.5g}", parent=dlg)
        slider = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        slider.setRange(0, num_steps)
        slider.setValue(value_to_slider(key, values[key]))
        slider.valueChanged.connect(make_updater(key, label))

        dlg._sliders[key] = slider
        dlg._value_labels[key] = label

        col = QVBoxLayout()
        col.addWidget(label)
        col.addWidget(slider)
        slider_row.addLayout(col)

    main_layout.addLayout(slider_row)

    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    return values


def fit_perturbative_resonator_shift_sym_manual(
    data: TwoToneData,
    initial_params: dict,
    param_ranges: Optional[dict] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
) -> dict:
    """
    Pops up a Qt window overlaying the perturbative_resonator_shift_sym spectrum on
    `data`'s two-tone flux map, with sliders for f_r, E_C, E_J, lambda_c and
    n_g. Move the sliders until the red curve tracks the measured branch,
    close the window, and the tuned values are printed and returned.

    Parameters
    ----------
    data : TwoToneData
        Two-tone data with `fluxes`, `f2_frequencies` and `signal` fields
        (as plotted elsewhere via Plotter.plot_flat_pcolormesh).
    initial_params, param_ranges, flux_vals, n_max, i :
        See perturbative_resonator_shift_sym_qt_widget.

    Returns
    -------
    dict[str, float]
        Final tuned values, keyed by 'f_r', 'E_C', 'E_J', 'lambda_c', 'n_g'.
    """

    fig, ax, cbar = Plotter.plot_flat_pcolormesh(data.fluxes, data.f2_frequencies / 1e9, data.signal)
    ax.set_xlabel(r'Flux ($2\pi$)')
    ax.set_ylabel(r'$f_2$ Frequency (GHz)')

    ip = get_ipython()

    # 1) Close existing figures and switch to external Qt windows
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    tuned_params = perturbative_resonator_shift_sym_qt_widget(
        fig, initial_params, param_ranges=param_ranges, flux_vals=flux_vals, n_max=n_max, i=i
    )

    # 2) Close Qt figures and switch back to inline plotting
    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    for key, value in tuned_params.items():
        print(f'{key} = {value:.6g}')

    return tuned_params


def resonator_circuit_params(E_j_E_c_ratio_unloaded, omega_p_unloaded, f_r_unloaded, Z_r_unloaded, C_c):
    """
    Converts the raw symmetric-SQUID circuit parameters into the
    (f_r, E_C, E_J, lambda_c) inputs perturbative_resonator_shift_sym expects, using
    the same conversion chain as the "Fit" section of
    analysis/QRCSJ_17_Bot_2.ipynb.

    Parameters
    ----------
    E_j_E_c_ratio_unloaded : float
        E_J/E_C ratio of the bare (unloaded) transmon, dimensionless.
    omega_p_unloaded : float
        Bare plasma frequency, in GHz.
    f_r_unloaded : float
        Bare (unloaded) resonator frequency, in GHz.
    Z_r_unloaded : float
        Bare resonator impedance, in Ohm.
    C_c : float
        Coupling capacitance, in Farad.

    Returns
    -------
    dict with keys 'E_J', 'E_C', 'f_r' (all in GHz) and 'lambda_c'
    (dimensionless) -- the inputs perturbative_resonator_shift_sym /
    perturbative_qubit_shift_sym / qubit_transition_frequency_sym expect
    -- plus the other loaded quantities from the same conversion chain:
    'E_C_loaded', 'omega_p_loaded' (GHz), 'Z_r_loaded' (Ohm) and
    'E_j_E_c_ratio_loaded' (dimensionless).
    """
    E_j_unloaded = np.sqrt(omega_p_unloaded**2 * E_j_E_c_ratio_unloaded)  # GHz
    E_c_unloaded = np.sqrt(omega_p_unloaded**2 / E_j_E_c_ratio_unloaded)  # GHz

    C_j = (2 * elementary_charge)**2 / (h * E_c_unloaded * 1e9)  # Farad
    C_r_unloaded = 1 / (Z_r_unloaded * 2 * np.pi * f_r_unloaded * 1e9)  # Farad
    L_r_unloaded = Z_r_unloaded / (2 * np.pi * f_r_unloaded * 1e9)  # Henry

    C_star = np.sqrt(C_j * C_c + C_j * C_r_unloaded + C_c * C_r_unloaded)
    Z_r_loaded = np.sqrt((L_r_unloaded * (C_j + C_c)) / (C_star**2))
    f_r_loaded = 1 / (2 * np.pi) * 1 / np.sqrt(L_r_unloaded * (C_star**2) / (C_j + C_c)) / 1e9  # GHz

    # E_J does not change with loading (that would be renormalization); only
    # E_C (and everything derived from it) shifts due to the coupling.
    E_c_loaded = ((2 * elementary_charge)**2 * (C_c + C_r_unloaded) / (C_star**2)) / h / 1e9  # GHz
    E_j_E_c_ratio_loaded = E_j_unloaded / E_c_loaded  # dimensionless
    omega_p_loaded = np.sqrt(E_j_unloaded * E_c_loaded)  # GHz

    rq = hbar / (2 * elementary_charge)**2
    Rq = rq * 2 * np.pi
    lambda_c = C_c / (C_r_unloaded + C_c) * np.sqrt(Rq / (4 * np.pi * Z_r_loaded))  # dimensionless

    return {
        'E_J': E_j_unloaded, 'E_C': E_c_unloaded, 'f_r': f_r_loaded, 'lambda_c': lambda_c,
        'E_C_loaded': E_c_loaded, 'omega_p_loaded': omega_p_loaded,
        'Z_r_loaded': Z_r_loaded, 'E_j_E_c_ratio_loaded': E_j_E_c_ratio_loaded,
    }


def fold_ng_to_zone(ng):
    """
    n_g is periodic with period 1 and mirror-symmetric about every
    half-integer; fold any value into the physically distinct [0, 0.5]
    zone (e.g. n_g=0.8 -> 0.2, n_g=1.3 -> 0.3).
    """
    folded = ng % 1
    if folded > 0.5:
        folded = 1 - folded
    return folded


def symmetric_squid_curve_for(kind, flux_arr, derived, ng, i=0, n_max=10):
    """
    Evaluates the resonator-shift curve (kind='resonator') or the
    n=0-photon qubit-transition curve (kind='qubit') over `flux_arr`,
    using the (f_r, E_C, E_J, lambda_c) already converted by
    resonator_circuit_params (`derived`). Shared building block for
    symmetric_squid_circuit_qt_widget and
    compute_symmetric_squid_fit_curves.
    """
    if kind == 'resonator':
        dfrs = np.array([
            perturbative_resonator_shift_sym(
                derived['f_r'], derived['E_C_loaded'], derived['E_J'], derived['lambda_c'],
                flux=flux, n_g=ng, i=i, n_max=n_max
            )
            for flux in flux_arr
        ])
        return derived['f_r'] + dfrs
    else:
        # qubit 0->1 transition branch, at n=0 photons
        return np.array([
            qubit_transition_frequency_sym(
                derived['E_C_loaded'], derived['E_J'], flux=flux, n_g=ng, n_max=n_max
            )
            + perturbative_qubit_shift_sym(
                derived['f_r'], derived['E_C_loaded'], derived['E_J'], derived['lambda_c'],
                flux=flux, n_g=ng, n=0, n_max=n_max
            )
            for flux in flux_arr
        ])


def compute_symmetric_squid_fit_curves(params, axis_kinds, per_axis_flux_vals, ng_offsets=None, i=0, n_max=10):
    """
    Computes, for each entry of `axis_kinds`/`per_axis_flux_vals`, the two
    fit curves at n_g and n_g+0.5 (each folded into [0, 0.5] via
    fold_ng_to_zone and shifted by that axis' ng_offset), from the raw
    symmetric-SQUID circuit parameters in `params` (as produced by
    fit_symmetric_squid_circuit_manual). Shared by
    symmetric_squid_circuit_qt_widget (with the live slider/field values)
    and plot_symmetric_squid_fit_results (with the final tuned values).

    Returns
    -------
    (curves_primary, curves_secondary, ngs_primary, ngs_secondary, derived)
        Each of the first four is a list, one entry per axis; `derived`
        is the resonator_circuit_params() dict for `params`.
    """
    if ng_offsets is None:
        ng_offsets = [0.0] * len(axis_kinds)

    derived = resonator_circuit_params(
        params['E_j_E_c_ratio_unloaded'], params['omega_p_unloaded'],
        params['f_r_unloaded'], params['Z_r_unloaded'], params['C_c']
    )

    curves_primary = []
    curves_secondary = []
    ngs_primary = []
    ngs_secondary = []
    for flux_arr, kind, offset in zip(per_axis_flux_vals, axis_kinds, ng_offsets):
        ng_primary = fold_ng_to_zone(params['n_g'] + offset)
        ng_secondary = fold_ng_to_zone(params['n_g'] + offset + 0.5)
        curves_primary.append(symmetric_squid_curve_for(kind, flux_arr, derived, ng_primary, i=i, n_max=n_max))
        curves_secondary.append(symmetric_squid_curve_for(kind, flux_arr, derived, ng_secondary, i=i, n_max=n_max))
        ngs_primary.append(ng_primary)
        ngs_secondary.append(ng_secondary)
    return curves_primary, curves_secondary, ngs_primary, ngs_secondary, derived


def symmetric_squid_circuit_qt_widget(
    fig: Figure,
    axes: List[Axes],
    axis_kinds: List[str],
    initial_params: dict,
    ng_offsets: Optional[List[float]] = None,
    param_ranges: Optional[dict] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
    num_steps: int = 1000,
    auto_fit_targets: Optional[dict] = None,
) -> dict:
    """
    Interactive Qt dialog with sliders for the raw symmetric-SQUID circuit
    parameters (E_j_E_c_ratio_unloaded, omega_p_unloaded, f_r_unloaded,
    Z_r_unloaded, C_c) and the offset charge n_g. A "Enter values
    manually" checkbox swaps every slider for a text field so exact
    numbers can be typed in instead (pressing Enter or clicking away
    applies it); toggling back to sliders picks up whatever was typed,
    widening that parameter's slider range first if the typed value falls
    outside it. On every change (slider or field), converts the current
    values to (f_r, E_C, E_J, lambda_c) via resonator_circuit_params and
    redraws two fit curves on every Axes in `axes`, matching the "overlay
    fit" cells in the notebook: on each Axes, one curve at
    n_g + ng_offsets[axis] and one at n_g + ng_offsets[axis] + 0.5 (each
    folded into the physically distinct [0, 0.5] zone), using the
    resonator-shift formula (perturbative_resonator_shift_sym) where
    axis_kinds[axis] == 'resonator' (e.g. an unshunted flux map) or the
    qubit-transition formula (qubit_transition_frequency_sym +
    perturbative_qubit_shift_sym, at n=0 photons) where axis_kinds[axis]
    == 'qubit' (e.g. the two-tone map). Each Axes' view stays locked to
    its original data extent, so the fit curves are never what determines
    the window's zoom.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Figure to embed in the Qt canvas (may contain more Axes than
        `axes`, e.g. colorbars -- those are left untouched).
    axes : list[matplotlib.axes.Axes]
        The data Axes to overlay the fit curves on, each with flux on x
        and frequency in GHz on y (e.g. [ax_unshunted, ax_twotone,
        ax_unshunted_025]). Must NOT include colorbar Axes -- pass the
        real data Axes explicitly rather than `fig.axes`, which also
        contains any colorbars added via `fig.colorbar(...)` and would
        otherwise get spurious fit curves and a legend of their own.
    axis_kinds : list[str]
        One 'resonator' or 'qubit' per entry of `axes`, selecting which
        formula to overlay on that Axes.
    initial_params : dict
        Starting values, keyed by 'E_j_E_c_ratio_unloaded', 'omega_p_unloaded',
        'f_r_unloaded', 'Z_r_unloaded', 'C_c' and 'n_g'.
    ng_offsets : list[float], optional
        One offset per entry of `axes`, added to n_g before evaluating
        that Axes' curves (e.g. 0.25 for a flux map measured at the
        ng=0.25 charge sector). Defaults to 0 for every Axes.
    param_ranges : dict, optional
        (min, max) slider bounds per key above. Any key left out defaults
        to +/-50% of its initial value ('n_g' defaults to [-0.5, 0.5]).
    flux_vals : np.ndarray, optional
        Flux points (in units of Phi_0) to evaluate the curve at, shared
        across every Axes. Defaults to 201 points spanning each Axes' own
        original x-limits independently.
    n_max : int
        Charge-basis truncation passed to perturbative_resonator_shift_sym /
        perturbative_qubit_shift_sym / qubit_transition_frequency_sym.
    i : int
        Initial state index passed to perturbative_resonator_shift_sym.
    num_steps : int
        Slider resolution (number of discrete steps) per parameter.
    auto_fit_targets : dict, optional
        Turns one or more parameters from manually-tuned sliders into
        automatically-solved outputs, each re-solved (holding every other
        current parameter value fixed) every time any slider/field
        changes -- including other auto-fit parameters solved earlier in
        the same pass, so order matters: a dict's keys are processed in
        insertion order, so put e.g. 'omega_p_unloaded' before
        'f_r_unloaded' if the latter should be solved using the former's
        freshly-updated value. Keyed by parameter name (must be one of
        `keys` above); each value is a list of
        (flux, frequency_GHz, kind, ng_offset) tuples ('kind' is
        'resonator' or 'qubit', same as `axis_kinds`; 'ng_offset' is added
        to the *current* n_g and folded via fold_ng_to_zone before
        evaluating the curve, so it tracks n_g even if n_g is itself being
        tuned live). The parameter is solved via 1D bounded minimization
        (scipy.optimize.minimize_scalar) of the summed squared residual
        between the relevant curve and every point in its list, searched
        over that parameter's slider range from `ranges`. A read-only
        label shows its current auto-solved value instead of a slider.

    Returns
    -------
    dict[str, float]
        Final tuned raw circuit values, keyed like `initial_params`.
    """

    if not axes:
        raise ValueError("axes must contain at least one Axes")
    if len(axis_kinds) != len(axes):
        raise ValueError("axis_kinds must have one entry per Axes in `axes`")
    if any(kind not in ('resonator', 'qubit') for kind in axis_kinds):
        raise ValueError("axis_kinds entries must be 'resonator' or 'qubit'")

    if ng_offsets is None:
        ng_offsets = [0.0] * len(axes)
    elif len(ng_offsets) != len(axes):
        raise ValueError("ng_offsets must have one entry per Axes in `axes`")

    keys = ['E_j_E_c_ratio_unloaded', 'omega_p_unloaded', 'f_r_unloaded', 'Z_r_unloaded', 'C_c', 'n_g']
    for key in keys:
        if key not in initial_params:
            raise KeyError(f"initial_params is missing '{key}'")

    # Lock each Axes' view to the extent of the data already plotted on it,
    # before the fit curve (which may start out far off) can stretch it.
    original_xlims = [ax.get_xlim() for ax in axes]
    original_ylims = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_autoscale_on(False)

    if flux_vals is not None:
        per_axis_flux_vals = [flux_vals for _ in axes]
    else:
        per_axis_flux_vals = [np.linspace(xmin, xmax, 201) for xmin, xmax in original_xlims]

    default_ranges = {key: (0.5 * initial_params[key], 1.5 * initial_params[key]) for key in keys}
    default_ranges['n_g'] = (-0.5, 0.5)

    ranges = {**default_ranges, **(param_ranges or {})}
    # normalise ranges in case a negative initial value flipped (lo, hi)
    ranges = {key: (min(lo, hi), max(lo, hi)) for key, (lo, hi) in ranges.items()}

    values = dict(initial_params)

    auto_fit_targets = auto_fit_targets or {}
    for key in auto_fit_targets:
        if key not in keys:
            raise KeyError(f"auto_fit_targets key '{key}' is not a known parameter (one of {keys})")

    def _auto_fit_cost(key, candidate, points):
        trial = dict(values)
        trial[key] = candidate
        derived = resonator_circuit_params(
            trial['E_j_E_c_ratio_unloaded'], trial['omega_p_unloaded'],
            trial['f_r_unloaded'], trial['Z_r_unloaded'], trial['C_c']
        )
        cost = 0.0
        for flux, freq, kind, ng_offset in points:
            ng = fold_ng_to_zone(trial['n_g'] + ng_offset)
            predicted = symmetric_squid_curve_for(kind, [flux], derived, ng, i=i, n_max=n_max)[0]
            cost += (predicted - freq) ** 2
        return cost

    def _apply_auto_fits():
        # Solves each auto-fit key in insertion order, each holding every
        # other current value fixed -- including auto-fit keys already
        # solved earlier in this same pass -- so e.g. f_r_unloaded's solve
        # sees omega_p_unloaded's freshly-updated value, not last frame's.
        for key, points in auto_fit_targets.items():
            if not points:
                continue
            lo, hi = ranges[key]
            result = minimize_scalar(
                lambda candidate: _auto_fit_cost(key, candidate, points),
                bounds=(lo, hi), method='bounded'
            )
            values[key] = result.x

    _apply_auto_fits()

    def compute_curves():
        return compute_symmetric_squid_fit_curves(
            values, axis_kinds, per_axis_flux_vals, ng_offsets=ng_offsets, i=i, n_max=n_max
        )

    def slider_to_value(key, slider_val):
        lo, hi = ranges[key]
        return lo + (slider_val / num_steps) * (hi - lo)

    def value_to_slider(key, value):
        lo, hi = ranges[key]
        if hi == lo:
            return 0
        return int(round((value - lo) / (hi - lo) * num_steps))

    # build dialog
    dlg = QDialog()
    dlg.setWindowTitle("Tune Symmetric SQUID Circuit Fit")
    dlg._sliders = {}
    dlg._fields = {}
    dlg._value_labels = {}

    main_layout = QVBoxLayout(dlg)
    dlg.resize(1600, 900)

    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    def _fit_labels(ng_primary, ng_secondary):
        return (f"Fit, n_g = {ng_primary:.3g}", f"Fit, n_g = {ng_secondary:.3g} (n_g+0.5)")

    initial_curves_primary, initial_curves_secondary, initial_ngs_primary, initial_ngs_secondary, initial_derived = compute_curves()
    lines_primary = [
        ax.plot(flux_arr, curve, color='red', lw=1.5, label=_fit_labels(ng_p, ng_s)[0])[0]
        for ax, flux_arr, curve, ng_p, ng_s in zip(axes, per_axis_flux_vals, initial_curves_primary, initial_ngs_primary, initial_ngs_secondary)
    ]
    lines_secondary = [
        ax.plot(flux_arr, curve, color='orange', lw=1.5, linestyle='dashed', label=_fit_labels(ng_p, ng_s)[1])[0]
        for ax, flux_arr, curve, ng_p, ng_s in zip(axes, per_axis_flux_vals, initial_curves_secondary, initial_ngs_primary, initial_ngs_secondary)
    ]
    # re-pin the view in case plot() nudged it despite autoscale being off.
    # ax.legend() replaces any legend already on that Axes rather than
    # stacking a second one, so calling it again on every update is safe.
    for ax, xlim, ylim in zip(axes, original_xlims, original_ylims):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='upper right')

    def _format_derived(derived):
        return (f"f_r = {derived['f_r']:.5g} GHz    "
                f"E_C = {derived['E_C']:.5g} GHz    "
                f"E_J = {derived['E_J']:.5g} GHz    "
                f"lambda_c = {derived['lambda_c']:.5g}")

    derived_label = QLabel(_format_derived(initial_derived), parent=dlg)
    main_layout.addWidget(derived_label)

    def refresh_plot():
        _apply_auto_fits()
        for key in auto_fit_targets:
            dlg._auto_fit_labels[key].setText(f"{key} = {values[key]:.5g} (auto)")
        curves_primary, curves_secondary, ngs_primary, ngs_secondary, derived = compute_curves()
        for ax, line_p, line_s, curve_p, curve_s, ng_p, ng_s in zip(
            axes, lines_primary, lines_secondary, curves_primary, curves_secondary, ngs_primary, ngs_secondary
        ):
            label_primary, label_secondary = _fit_labels(ng_p, ng_s)
            line_p.set_ydata(curve_p)
            line_p.set_label(label_primary)
            line_s.set_ydata(curve_s)
            line_s.set_label(label_secondary)
            ax.legend(loc='upper right')
        derived_label.setText(_format_derived(derived))
        canvas.draw_idle()

    def make_slider_updater(key, value_label):
        def _upd(slider_val):
            values[key] = slider_to_value(key, slider_val)
            value_label.setText(f"{key} = {values[key]:.5g}")
            refresh_plot()
        return _upd

    def make_field_updater(key, value_label, slider, field):
        def _upd():
            try:
                new_value = float(field.text())
            except ValueError:
                field.setText(f"{values[key]:.6g}")
                return
            values[key] = new_value
            # widen the slider's range if the typed value falls outside it,
            # so switching back to slider mode doesn't clip it
            lo, hi = ranges[key]
            if new_value < lo or new_value > hi:
                ranges[key] = (min(lo, new_value), max(hi, new_value))
            slider.blockSignals(True)
            slider.setValue(value_to_slider(key, new_value))
            slider.blockSignals(False)
            value_label.setText(f"{key} = {values[key]:.5g}")
            refresh_plot()
        return _upd

    def _on_mode_toggled(use_fields):
        for key in keys:
            if key in auto_fit_targets:
                continue
            dlg._sliders[key].setVisible(not use_fields)
            dlg._fields[key].setVisible(use_fields)
            if use_fields:
                dlg._fields[key].setText(f"{values[key]:.6g}")

    mode_checkbox = QCheckBox("Enter values manually", parent=dlg)
    mode_checkbox.toggled.connect(_on_mode_toggled)
    main_layout.addWidget(mode_checkbox)

    dlg._auto_fit_labels = {}
    slider_row = QHBoxLayout()
    for key in keys:
        if key in auto_fit_targets:
            # Auto-solved (see auto_fit_targets): shown read-only, not a slider.
            auto_label = QLabel(f"{key} = {values[key]:.5g} (auto)", parent=dlg)
            dlg._auto_fit_labels[key] = auto_label
            col = QVBoxLayout()
            col.addWidget(auto_label)
            slider_row.addLayout(col)
            continue

        label = QLabel(f"{key} = {values[key]:.5g}", parent=dlg)

        slider = QSlider(Qt.Orientation.Horizontal, parent=dlg)
        slider.setRange(0, num_steps)
        slider.setValue(value_to_slider(key, values[key]))
        slider.valueChanged.connect(make_slider_updater(key, label))

        field = QLineEdit(f"{values[key]:.6g}", parent=dlg)
        field.setValidator(QDoubleValidator())
        field.setVisible(False)
        field.editingFinished.connect(make_field_updater(key, label, slider, field))

        dlg._sliders[key] = slider
        dlg._fields[key] = field
        dlg._value_labels[key] = label

        col = QVBoxLayout()
        col.addWidget(label)
        col.addWidget(slider)
        col.addWidget(field)
        slider_row.addLayout(col)

    main_layout.addLayout(slider_row)

    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    return values


def fit_symmetric_squid_circuit_manual(
    unshunted_data: FluxMapData,
    two_tone_data: TwoToneData,
    initial_params: dict,
    unshunted_025_data: Optional[FluxMapData] = None,
    param_ranges: Optional[dict] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
    auto_fit_targets: Optional[dict] = None,
    two_tone_ng_offset: float = 0.0,
) -> dict:
    """
    Pops up a Qt window with the unshunted flux map, the two-tone map, and
    (if given) a second unshunted flux map measured at the n_g+0.25 charge
    sector, side by side -- each window sized to its own data, not to the
    fit curves -- overlaying two fit curves per panel: one at n_g and one
    at n_g+0.5 (folded into [0, 0.5]), each shifted by that panel's own
    ng_offset (0 for the unshunted n_g=0 panel's primary curve and
    `two_tone_ng_offset` for the two-tone panel's, but the ng=0.25 panel's
    curves are effectively n_g+0.25 and n_g+0.75). Uses
    the resonator-shift curve (perturbative_resonator_shift_sym) on the
    unshunted panels and the qubit-transition curve
    (qubit_transition_frequency_sym + perturbative_qubit_shift_sym) on
    the two-tone panel, matching the "overlay fit" cells in this
    notebook. Sliders (or typed values, via the "Enter values manually"
    checkbox) control the raw symmetric-SQUID circuit parameters
    (E_j_E_c_ratio_unloaded, omega_p_unloaded, f_r_unloaded, Z_r_unloaded,
    C_c) and the offset charge n_g -- the same parameters and conversion
    chain as the "Fit" section of this notebook (see
    resonator_circuit_params). Move the sliders until the curves track
    the measured branches on every panel, close the window, and the
    tuned raw values are printed and returned.

    Parameters
    ----------
    unshunted_data : FluxMapData
        Unshunted flux map data at the n_g=0 charge sector (plotted via
        Plotter.plot_flux_map_fluxes).
    two_tone_data : TwoToneData
        Two-tone data with `fluxes`, `f2_frequencies` and `signal` fields
        (plotted via Plotter.plot_flat_pcolormesh).
    unshunted_025_data : FluxMapData, optional
        A second unshunted flux map measured at the n_g=0.25 charge
        sector. If given, adds a third panel overlaying the same
        resonator-shift curve, shifted by n_g+0.25.
    initial_params, param_ranges, flux_vals, n_max, i, auto_fit_targets :
        See symmetric_squid_circuit_qt_widget. `auto_fit_targets`'
        (flux, frequency, kind, ng_offset) points are just numbers here --
        see fit_symmetric_squid_circuit_guided for a version that gets them by
        letting you click them directly on this same data.
    two_tone_ng_offset : float
        The n_g offset-charge sector `two_tone_data` was actually measured
        at, relative to `unshunted_data`'s n_g=0 (added to the shared n_g
        fit parameter, folded via fold_ng_to_zone, same as
        `unshunted_025_data`'s fixed 0.25). Defaults to 0.0, i.e. assumes
        the two-tone trace was taken at the same charge sector as the
        n_g=0 flux map.

    Returns
    -------
    dict[str, float]
        Final tuned raw circuit values, keyed by 'E_j_E_c_ratio_unloaded',
        'omega_p_unloaded', 'f_r_unloaded', 'Z_r_unloaded', 'C_c', 'n_g'.
    """

    num_panels = 3 if unshunted_025_data is not None else 2
    fig, panel_axes = plt.subplots(1, num_panels, figsize=(7 * num_panels, 6))
    ax_unshunted, ax_twotone = panel_axes[0], panel_axes[1]

    Plotter.plot_flux_map_fluxes(unshunted_data, title='Unshunted, n_g=0', comment=False, fig=fig, ax=ax_unshunted)

    Plotter.plot_flat_pcolormesh(two_tone_data.fluxes, two_tone_data.f2_frequencies / 1e9, two_tone_data.signal, fig=fig, ax=ax_twotone)
    ax_twotone.set_xlabel(r'Flux ($2\pi$)')
    ax_twotone.set_ylabel(r'$f_2$ Frequency (GHz)')
    ax_twotone.set_title('Two tone')

    axes = [ax_unshunted, ax_twotone]
    axis_kinds = ['resonator', 'qubit']
    ng_offsets = [0.0, two_tone_ng_offset]

    if unshunted_025_data is not None:
        ax_unshunted_025 = panel_axes[2]
        Plotter.plot_flux_map_fluxes(unshunted_025_data, title='Unshunted, n_g=0.25', comment=False, fig=fig, ax=ax_unshunted_025)
        axes.append(ax_unshunted_025)
        axis_kinds.append('resonator')
        ng_offsets.append(0.25)

    ip = get_ipython()

    # 1) Close existing figures and switch to external Qt windows
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    tuned_params = symmetric_squid_circuit_qt_widget(
        fig, axes, axis_kinds, initial_params, ng_offsets=ng_offsets,
        param_ranges=param_ranges, flux_vals=flux_vals, n_max=n_max, i=i,
        auto_fit_targets=auto_fit_targets,
    )

    # 2) Close Qt figures and switch back to inline plotting
    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    for key, value in tuned_params.items():
        print(f'{key} = {value:.6g}')

    return tuned_params


def plot_symmetric_squid_fit_results(
    unshunted_data: FluxMapData,
    two_tone_data: TwoToneData,
    tuned_params: dict,
    unshunted_025_data: Optional[FluxMapData] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
    two_tone_ng_offset: float = 0.0,
) -> Figure:
    """
    Builds a static (inline/savable) results figure summarizing a
    symmetric-SQUID circuit fit made with fit_symmetric_squid_circuit_manual:
    the unshunted flux map, the two-tone map, and (if given) the second
    unshunted flux map at the n_g=0.25 charge sector, each overlaid with
    the fit curves at n_g and n_g+0.5 (folded into [0, 0.5], shifted by
    that panel's own ng_offset -- 0.25 for the third panel), using
    `tuned_params` -- the same raw circuit parameters and formulas as the
    interactive tool (see compute_symmetric_squid_fit_curves). Also adds
    a text box with the final derived quantities: the Ej/Ec ratio, plasma
    frequency, resonator frequency and impedance, each loaded and
    unloaded, plus the coupling capacitance C_c.

    Parameters
    ----------
    unshunted_data : FluxMapData
        Unshunted flux map data at the n_g=0 charge sector.
    two_tone_data : TwoToneData
        Two-tone data with `fluxes`, `f2_frequencies` and `signal` fields.
    tuned_params : dict
        Raw circuit parameters, keyed by 'E_j_E_c_ratio_unloaded',
        'omega_p_unloaded', 'f_r_unloaded', 'Z_r_unloaded', 'C_c' and
        'n_g' -- typically the dict returned by
        fit_symmetric_squid_circuit_manual.
    unshunted_025_data : FluxMapData, optional
        A second unshunted flux map measured at the n_g=0.25 charge
        sector. If given, adds a third panel.
    flux_vals, n_max, i :
        See symmetric_squid_circuit_qt_widget.
    two_tone_ng_offset : float
        The n_g offset-charge sector `two_tone_data` was actually measured
        at; pass whatever value was used for `fit_symmetric_squid_circuit_manual`'s
        (or fit_symmetric_squid_circuit_guided's) same argument so this
        summary figure's two-tone overlay matches the one you tuned
        against. Defaults to 0.0.

    Returns
    -------
    matplotlib.figure.Figure
        The finished results figure (not shown automatically -- call
        plt.show() or save it, e.g. fig.savefig(...)).
    """

    num_panels = 3 if unshunted_025_data is not None else 2
    fig, panel_axes = plt.subplots(1, num_panels, figsize=(7 * num_panels, 6))
    ax_unshunted, ax_twotone = panel_axes[0], panel_axes[1]

    Plotter.plot_flux_map_fluxes(unshunted_data, title='Unshunted, n_g=0', comment=False, fig=fig, ax=ax_unshunted)

    Plotter.plot_flat_pcolormesh(two_tone_data.fluxes, two_tone_data.f2_frequencies / 1e9, two_tone_data.signal, fig=fig, ax=ax_twotone)
    ax_twotone.set_xlabel(r'Flux ($2\pi$)')
    ax_twotone.set_ylabel(r'$f_2$ Frequency (GHz)')
    ax_twotone.set_title('Two tone')

    axes = [ax_unshunted, ax_twotone]
    axis_kinds = ['resonator', 'qubit']
    ng_offsets = [0.0, two_tone_ng_offset]

    if unshunted_025_data is not None:
        ax_unshunted_025 = panel_axes[2]
        Plotter.plot_flux_map_fluxes(unshunted_025_data, title='Unshunted, n_g=0.25', comment=False, fig=fig, ax=ax_unshunted_025)
        axes.append(ax_unshunted_025)
        axis_kinds.append('resonator')
        ng_offsets.append(0.25)

    # capture each panel's own data extent before adding fit curves, so
    # they never stretch the view away from the measured data
    original_xlims = [ax.get_xlim() for ax in axes]
    original_ylims = [ax.get_ylim() for ax in axes]

    if flux_vals is not None:
        per_axis_flux_vals = [flux_vals for _ in axes]
    else:
        per_axis_flux_vals = [np.linspace(xmin, xmax, 201) for xmin, xmax in original_xlims]

    curves_primary, curves_secondary, ngs_primary, ngs_secondary, derived = compute_symmetric_squid_fit_curves(
        tuned_params, axis_kinds, per_axis_flux_vals, ng_offsets=ng_offsets, i=i, n_max=n_max
    )

    for ax, flux_arr, curve_p, curve_s, ng_p, ng_s, xlim, ylim in zip(
        axes, per_axis_flux_vals, curves_primary, curves_secondary, ngs_primary, ngs_secondary, original_xlims, original_ylims
    ):
        ax.plot(flux_arr, curve_p, color='red', lw=1.5, label=f'Fit, n_g = {ng_p:.3g}')
        ax.plot(flux_arr, curve_s, color='orange', lw=1.5, linestyle='dashed', label=f'Fit, n_g = {ng_s:.3g} (n_g+0.5)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='upper right')

    summary = (
        f"$E_J/E_C$: unloaded = {tuned_params['E_j_E_c_ratio_unloaded']:.3g}, "
        f"loaded = {derived['E_j_E_c_ratio_loaded']:.3g}\n"
        f"$\\omega_p$: unloaded = {tuned_params['omega_p_unloaded']:.4g} GHz, "
        f"loaded = {derived['omega_p_loaded']:.4g} GHz\n"
        f"$f_r$: unloaded = {tuned_params['f_r_unloaded']:.4g} GHz, "
        f"loaded = {derived['f_r']:.4g} GHz\n"
        f"$Z_r$: unloaded = {tuned_params['Z_r_unloaded']:.4g} $\\Omega$, "
        f"loaded = {derived['Z_r_loaded']:.4g} $\\Omega$\n"
        f"$C_c$ = {tuned_params['C_c']:.3g} F"
    )
    fig.subplots_adjust(bottom=0.32)
    fig.text(0.5, 0.02, summary, ha='center', va='bottom', fontsize=10)

    return fig


def pick_points_qt_widget(
    fig: Figure,
    title: str = "Pick Points",
    max_points: Optional[int] = None,
    marker_color: str = 'red',
    ask_n_g: bool = False,
    n_g_default: float = 0.0,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Interactive Qt dialog for picking (x, y) points by clicking on an
    existing Matplotlib Figure, with zoom/pan support.

    Left-clicking on the Axes adds a point, drawn as an 'x' marker, and
    appends it to the returned list. While a toolbar tool (zoom/pan) is
    active, clicks are consumed by that tool instead -- click its button
    again to release it before picking points. "Remove last point" undoes
    the most recent pick; "Done" closes the dialog.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A Figure with at least one Axes to pick points from.
    title : str
        Dialog window title.
    max_points : int, optional
        Stop accepting new clicks once this many points have been picked
        (further clicks are ignored). None (default): no limit.
    marker_color : str
        Color of the markers drawn for picked points.
    ask_n_g : bool
        If True, shows an editable n_g field next to the plot (same
        QLineEdit + QDoubleValidator style as the "Enter values manually"
        fields in symmetric_squid_circuit_qt_widget), so you can record
        which offset-charge sector the picked point(s) belong to
        alongside them, without a separate prompt.
    n_g_default : float
        Starting value shown in the n_g field (only shown if ask_n_g=True).

    Returns
    -------
    tuple[list[tuple[float, float]], float]
        The picked (x, y) points, in click order, and the n_g field's
        final value (unchanged `n_g_default` if ask_n_g=False).
    """
    if not fig.axes:
        raise ValueError("Figure must contain at least one Axes")
    ax = fig.axes[0]

    points: List[Tuple[float, float]] = []
    markers = []
    n_g_value = n_g_default

    dlg = QDialog()
    dlg.setWindowTitle(title)
    main_layout = QVBoxLayout(dlg)
    dlg.resize(1200, 800)

    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, dlg)
    main_layout.addWidget(toolbar)
    main_layout.addWidget(canvas)

    if ask_n_g:
        def _on_ng_changed():
            nonlocal n_g_value
            try:
                n_g_value = float(ng_field.text())
            except ValueError:
                ng_field.setText(f"{n_g_value:g}")

        ng_row = QHBoxLayout()
        ng_row.addWidget(QLabel("n_g =", parent=dlg))
        ng_field = QLineEdit(f"{n_g_default:g}", parent=dlg)
        ng_field.setValidator(QDoubleValidator())
        ng_field.editingFinished.connect(_on_ng_changed)
        ng_row.addWidget(ng_field)
        main_layout.addLayout(ng_row)

    status_label = QLabel("Click on the plot to pick a point.", parent=dlg)
    main_layout.addWidget(status_label)

    def _update_status():
        count_str = f"{len(points)}" + (f"/{max_points}" if max_points is not None else "")
        status_label.setText(f"Picked {count_str} point(s).")

    def _on_click(event):
        # Ignore clicks outside the Axes, and clicks consumed by an active
        # toolbar tool (zoom/pan set `toolbar.mode` to a non-empty string).
        if event.inaxes != ax or toolbar.mode != '':
            return
        if max_points is not None and len(points) >= max_points:
            return
        points.append((event.xdata, event.ydata))
        (marker,) = ax.plot(event.xdata, event.ydata, marker='x', color=marker_color,
                             markersize=10, markeredgewidth=2, linestyle='none')
        markers.append(marker)
        canvas.draw_idle()
        _update_status()

    canvas.mpl_connect('button_press_event', _on_click)

    def _remove_last():
        if not points:
            return
        points.pop()
        markers.pop().remove()
        canvas.draw_idle()
        _update_status()

    remove_btn = QPushButton("Remove last point", parent=dlg)
    remove_btn.clicked.connect(_remove_last)
    main_layout.addWidget(remove_btn)

    done_btn = QPushButton("Done", parent=dlg)
    done_btn.clicked.connect(dlg.accept)
    main_layout.addWidget(done_btn)

    app = QApplication.instance() or QApplication(sys.argv)
    dlg.exec()

    return points, n_g_value


def pick_two_tone_transition_point_manual(
    data: TwoToneData,
    title: str = "Pick the qubit 0->1 transition frequency at zero flux",
    n_g_default: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Pops up a Qt window with just `data`'s two-tone map (via
    Plotter.plot_flat_pcolormesh) and lets you click the single point you
    identify as the qubit's 0->1 transition frequency at zero flux. Once
    picked, an editable n_g field next to the plot (defaulting to
    `n_g_default`, 0.0) lets you record which offset-charge sector this
    two-tone trace was actually taken at, relative to the same reference
    as the flux maps -- override it if this two-tone trace wasn't taken
    at the same charge sector as the n_g=0 flux map.

    Returns
    -------
    tuple[float, float, float]
        The picked (flux, frequency_GHz, n_g).

    Raises
    ------
    RuntimeError
        If the window was closed without picking exactly one point.
    """
    fig, ax, cbar = Plotter.plot_flat_pcolormesh(data.fluxes, data.f2_frequencies / 1e9, data.signal)
    ax.set_xlabel(r'Flux ($2\pi$)')
    ax.set_ylabel(r'$f_2$ Frequency (GHz)')
    ax.set_title(title)

    ip = get_ipython()
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    points, n_g = pick_points_qt_widget(fig, title=title, max_points=1, ask_n_g=True, n_g_default=n_g_default)

    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    if len(points) != 1:
        raise RuntimeError(f"Expected exactly one point, got {len(points)}. Try again.")

    flux, freq = points[0]
    print(f"Zero-flux two-tone point: flux = {flux:.4g}, f = {freq:.6g} GHz, n_g = {n_g:g}")
    return flux, freq, n_g


def pick_half_flux_points_manual(
    data: FluxMapData,
    title: str = "Pick 0.5-flux (and equivalent) points",
) -> List[Tuple[float, float]]:
    """
    Pops up a Qt window with just `data`'s unshunted flux map (via
    Plotter.plot_flux_map_fluxes) and lets you click every point you
    identify as a 0.5-flux-quantum branch minimum -- 0.5, 1.5, -0.5 Phi_0,
    however many equivalent copies fall within the swept flux range.

    Returns
    -------
    list[tuple[float, float]]
        The picked (flux, frequency_GHz) points, in click order. Empty if
        the window was closed without picking any.
    """
    fig, ax = Plotter.plot_flux_map_fluxes(data, comment=False)
    ax.set_title(title)

    ip = get_ipython()
    plt.close('all')
    ip.run_line_magic('matplotlib', 'qt')

    points, _ = pick_points_qt_widget(fig, title=title)

    plt.close('all')
    ip.run_line_magic('matplotlib', 'inline')

    if not points:
        print("No points picked.")
    else:
        freqs = [f for _, f in points]
        picks_str = ", ".join(f"{f:.6g}" for f in freqs)
        print(f"Picked {len(points)} point(s), mean f = {np.mean(freqs):.6g} GHz (individual: {picks_str})")

    return points


def fit_symmetric_squid_circuit_guided(
    unshunted_data: FluxMapData,
    two_tone_data: TwoToneData,
    initial_params: dict,
    unshunted_025_data: Optional[FluxMapData] = None,
    param_ranges: Optional[dict] = None,
    flux_vals: Optional[np.ndarray] = None,
    n_max: int = 10,
    i: int = 0,
) -> dict:
    """
    Guided version of fit_symmetric_squid_circuit_manual: walks you through
    picking calibration points on the raw data first, then uses them to
    keep omega_p_unloaded and f_r_unloaded automatically anchored while
    you tune the remaining parameters by hand in the same interactive
    tuner.

    1) Pops up the two-tone map alone and asks you to click the qubit's
       0->1 transition frequency at zero flux; an editable n_g field next
       to the plot (defaulting to 0.0, i.e. the same sector as
       `unshunted_data`) lets you record which offset-charge sector that
       trace was actually taken at, if not 0
       (pick_two_tone_transition_point_manual).
    2) Pops up `unshunted_data` (n_g=0) alone and asks you to click every
       0.5-flux-equivalent branch minimum you can identify
       (pick_half_flux_points_manual).
    3) If `unshunted_025_data` is given, does the same for it (n_g=0.25).
    4) Opens the same interactive tuner as fit_symmetric_squid_circuit_manual
       (same panels; same sliders for E_j_E_c_ratio_unloaded,
       Z_r_unloaded, C_c and n_g) except omega_p_unloaded and f_r_unloaded
       are no longer sliders: every time any other parameter changes,
       omega_p_unloaded is first re-solved (all else held fixed) to match
       the zero-flux two-tone point from step 1 -- evaluated at the n_g
       chosen in step 1, not assumed to be 0 -- then f_r_unloaded is
       re-solved (all else, including the just-updated omega_p_unloaded,
       held fixed) to match the combined 0.5-flux points from steps 2-3.
       The two-tone panel's own overlaid fit curves also use the n_g
       chosen in step 1 (via `two_tone_ng_offset`), so what you see
       overlaid on the two-tone data matches what omega_p_unloaded is
       actually being solved against. Their current auto-solved values
       are shown as read-only labels.

    Because the 0.5-flux target combines points from both charge sectors
    (n_g=0 and n_g=0.25) into a single f_r_unloaded, and because the
    dispersive shift used for the two-tone point technically depends on
    f_r_unloaded too, this auto-tuning is a good starting point rather
    than an exact joint fit -- use the remaining sliders (E_j_E_c_ratio_unloaded
    especially) to refine it by eye against all panels at once, same as
    fit_symmetric_squid_circuit_manual.

    Parameters
    ----------
    unshunted_data, two_tone_data, unshunted_025_data, param_ranges,
    flux_vals, n_max, i :
        See fit_symmetric_squid_circuit_manual.
    initial_params : dict
        Same keys as fit_symmetric_squid_circuit_manual. Its
        'omega_p_unloaded' and 'f_r_unloaded' entries only matter as the
        center of the +/-50% search bracket used for their first
        auto-solve (see `param_ranges` to override that bracket) -- their
        displayed values will immediately change to the auto-solved ones.

    Returns
    -------
    dict[str, float]
        Final tuned raw circuit values, keyed like
        fit_symmetric_squid_circuit_manual's.
    """
    two_tone_flux, two_tone_freq, two_tone_ng = pick_two_tone_transition_point_manual(two_tone_data)

    ng0_points = pick_half_flux_points_manual(unshunted_data, title="Pick 0.5-flux points, n_g = 0")

    ng025_points: List[Tuple[float, float]] = []
    if unshunted_025_data is not None:
        ng025_points = pick_half_flux_points_manual(unshunted_025_data, title="Pick 0.5-flux points, n_g = 0.25")

    if not ng0_points and not ng025_points:
        raise RuntimeError("No 0.5-flux points were picked; can't auto-tune f_r_unloaded.")

    # (flux, frequency_GHz, kind, ng_offset) -- ng_offset tracks n_g even as
    # it's tuned live, via fold_ng_to_zone(current n_g + ng_offset). The
    # two-tone point uses the n_g picked alongside it in step 1, not 0.
    auto_fit_targets = {
        'omega_p_unloaded': [(two_tone_flux, two_tone_freq, 'qubit', two_tone_ng)],
        'f_r_unloaded': (
            [(flux, freq, 'resonator', 0.0) for flux, freq in ng0_points]
            + [(flux, freq, 'resonator', 0.25) for flux, freq in ng025_points]
        ),
    }

    return fit_symmetric_squid_circuit_manual(
        unshunted_data, two_tone_data, initial_params,
        unshunted_025_data=unshunted_025_data, param_ranges=param_ranges,
        flux_vals=flux_vals, n_max=n_max, i=i, auto_fit_targets=auto_fit_targets,
        two_tone_ng_offset=two_tone_ng,
    )