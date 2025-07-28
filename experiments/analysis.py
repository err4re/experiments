from typing import Type, Tuple, Union, List, Optional

import numpy as np
from qutip import Qobj

from scipy.optimize import curve_fit

import re

import sys
from PyQt6.QtWidgets import (
    QApplication, QDialog, QSlider, QPushButton,
    QHBoxLayout, QVBoxLayout, QLabel
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

from IPython import get_ipython
import matplotlib.pyplot as plt

import time
from tqdm.notebook import tqdm


from experiments.experiment_data import FluxMapData
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
    """
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
    """
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

def find_voltage_to_flux(data: FluxMapData, resolution_fraction: float = 0.0001):

    minima = [data.f[np.argmin(S_to_dBm(s))] for s in data.S]

    # frequency in the middle of the gap between upper and lower branches
    separating_freq = find_separating_line(minima)

    upper_branches_voltages = data.voltages[minima > separating_freq]
    lower_branches_voltages = data.voltages[minima < separating_freq]

    zero_flux_lines = find_separating_lines(upper_branches_voltages)
    pi_flux_lines = find_separating_lines(lower_branches_voltages)

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

