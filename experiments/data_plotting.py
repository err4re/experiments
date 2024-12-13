import matplotlib.pyplot as plt
import numpy as np

def plot_trace_mag_phase(f, z, S_parameter='S12', title='', freq_unit='GHz', mag_scale='dB'):
    """
    Plots the magnitude and phase of a complex-valued function (e.g., an S-parameter) against frequency.

    Parameters:
    - f (array-like): Array of frequencies at which the measurements are taken.
    - z (array-like): Complex-valued measurements corresponding to frequencies in `f`.
    - S_parameter (str, optional): Label of the S-parameter (default 'S12').
    - title (str, optional): Title of the plot.
    - freq_unit (str, optional): Unit for the frequency axis ('GHz', 'MHz', etc.).
    - mag_scale (str, optional): Scale for magnitude ('dB' for decibels, 'linear' for linear scale).

    Returns:
    - fig, ax1, ax2: Figure and axis objects of the plot.
    """
    # Validate inputs
    if not (isinstance(f, (list, np.ndarray)) and isinstance(z, (list, np.ndarray))):
        raise ValueError("Frequency and measurement inputs must be array-like.")

    if len(f) != len(z):
        raise ValueError("Frequency and measurement arrays must be of the same length.")

    # Create subplots
    fig, ax1 = plt.subplots()

    # Plot magnitude
    color = 'tab:blue'
    if mag_scale.lower() == 'db':
        ax1.plot(f / 1e9, 20 * np.log10(np.abs(z)), color=color)
        ax1.set_ylabel(f'|{S_parameter}| in dB', color=color)
    else:
        ax1.plot(f / 1e9, np.abs(z), color=color)
        ax1.set_ylabel(f'|{S_parameter}|', color=color)
    ax1.set_xlabel(f'Frequency ({freq_unit})')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot phase
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(f / 1e9, np.angle(z, deg=True), color=color)
    ax2.set_ylabel(f'arg({S_parameter}) in degrees', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Set title if provided
    if title:
        plt.title(title)

    plt.show()

    return fig, ax1, ax2


def plot_trace_mag_phase_stacked(f, z, S_parameter='S12', title='', freq_unit='GHz', mag_scale='dB'):
    """
    Plots the magnitude and phase of a complex-valued function (e.g., an S-parameter) against frequency.
    The magnitude and phase plots are stacked vertically.

    Parameters:
    - f (array-like): Array of frequencies at which the measurements are taken.
    - z (array-like): Complex-valued measurements corresponding to frequencies in `f`.
    - S_parameter (str, optional): Label of the S-parameter (default 'S12').
    - title (str, optional): Title of the plot.
    - freq_unit (str, optional): Unit for the frequency axis ('GHz', 'MHz', etc.).
    - mag_scale (str, optional): Scale for magnitude ('dB' for decibels, 'linear' for linear scale).

    Returns:
    - fig, ax_magnitude, ax_phase: Figure and axis objects for magnitude and phase plots.
    """
    # Validate inputs
    if not (isinstance(f, (list, np.ndarray)) and isinstance(z, (list, np.ndarray))):
        raise ValueError("Frequency and measurement inputs must be array-like.")

    if len(f) != len(z):
        raise ValueError("Frequency and measurement arrays must be of the same length.")

    # Create subplots for vertical stacking
    fig, (ax_magnitude, ax_phase) = plt.subplots(nrows=2, ncols=1, sharex=True)

    # Plot magnitude
    color = 'tab:blue'
    if mag_scale.lower() == 'db':
        ax_magnitude.plot(f / 1e9, 20 * np.log10(np.abs(z)), color=color)
        ax_magnitude.set_ylabel(f'|{S_parameter}| in dB', color=color)
    else:
        ax_magnitude.plot(f / 1e9, np.abs(z), color=color)
        ax_magnitude.set_ylabel(f'|{S_parameter}|', color=color)
    ax_magnitude.set_title(title)

    # Plot phase
    color = 'tab:red'
    ax_phase.plot(f / 1e9, np.angle(z, deg=True), color=color)
    ax_phase.set_xlabel(f'Frequency ({freq_unit})')
    ax_phase.set_ylabel(f'arg({S_parameter}) in degrees', color=color)

    plt.tight_layout()
    plt.show()

    return fig, ax_magnitude, ax_phase

