

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

from typing import Type, Tuple, Union, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes


### Hysteresis correction

def hystersis_system_of_equations(points, symmetry_points, degree : int) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Constructs a system of linear equations based on a set of points and symmetry points to model hysteresis behavior.

    This function generates a matrix `A` representing the coefficients of the system and a vector `b` representing the
    constants, based on the provided points, symmetry points, and the degree of the polynomial intended to model the
    hysteresis curve.

    Parameters:
    - points (List[Tuple[float, float]]): A list of tuples where each tuple represents a point (x, y) = (current, flux) on the hysteresis curve.
    - symmetry_points (List[Tuple[float, float]]): A list of tuples where each tuple represents two currents that should correspond to the same flux, imposing a symmetry
      condition on the hysteresis curve.
    - degree (int): The degree of the polynomial used to model the hysteresis curve.

    Returns:
    - A (numpy.ndarray): A matrix of shape (n_rows, n_cols) where `n_rows` is the total number of points and symmetry
      points, and `n_cols` is the degree of the polynomial plus one. This matrix contains the coefficients of the
      system of equations.
    - b (numpy.ndarray): A vector of length `n_rows` containing the constants of the system of equations.

    Raises:
    - ValueError: If the list of points is empty, as at least one point is required to construct the system.

    Notes:
    - use `solution, residuals, rank, s = lstsq(A, b)` to solve the system of equations and correct hysteresis
    - The function initializes matrix `A` and vector `b` with NaN values and fills them based on the input points and
      symmetry conditions. Each point adds an equation to the system, while each symmetry point adds a condition that
      both the x and y coordinates must satisfy.
    - The first column of `A` is filled with 1 for points and 2 for symmetry points, representing the constant term of
      the polynomial equation. The subsequent columns are filled based on the polynomial terms up to the specified degree.
    """

    #matrix describing system of linear equations
    n_rows = len(points) + len(symmetry_points)
    n_cols = degree + 1

    A = np.full((n_rows, n_cols), np.nan)
    b = np.full(n_rows, np.nan)

    x_origin = points[0][0]

    for i,point in enumerate(points):
        
        A[i][0] = 1
        b[i] = point[1]

        for j in range(degree):
            A[i][j+1] = (point[0])**(j+1)


    for i,symmetry_point in enumerate(symmetry_points):

        k = i + len(points)
        
        A[k][0] = 2
        b[k] = 0

        for j in range(degree):
            A[k][j+1] = (symmetry_point[0])**(j+1) + (symmetry_point[1])**(j+1)

    return A,b


def current_to_flux(I: Union[float, List[float], np.ndarray], solution: np.ndarray) -> Union[float, np.ndarray]:
    """
    Converts current(s) to flux, while correcting for hysteresis.
    Uses a polynomial equation defined by the coefficients in `solution`. 
    The coefficients are passed in in ascending order of power.

    Parameters:
    - I (Union[float, List[float], np.ndarray]): The current value(s) to be converted. 
      Can be a single float value, a list of floats, or a NumPy array of any shape.
    - solution (np.ndarray): An array of coefficients for the polynomial equation, 
      in ascending order of power.

    Returns:
    - Union[float, np.ndarray]: The calculated flux for the given current(s). Returns 
      a single float if the input is a single current value, or a NumPy array of the 
      same shape as the input if the input is a list or an array.

    Example:
    >>> current_to_flux(2.0, np.array([1, 0, 0.5]))
    3.0
    >>> current_to_flux(np.array([1, 2, 3]), np.array([1, 0, 0.5]))
    array([1.5, 3. , 5.5])
    """

    I_array = np.asarray(I)  # Convert input to a NumPy array for uniform processing
    reversed_solution = solution[::-1]  # Reverse solution for np.polyval
    flux = np.polyval(reversed_solution, I_array)  # Evaluate the polynomial
    return flux


def plot_current_to_flux(Is: Union[np.ndarray, List[float]], solution: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the flux as a function of current based on the correction for hysteresis according to `solution`.
    Returns the figure and axes objects for further manipulation or display.

    Parameters:
    - Is (Union[np.ndarray, List[float]]): An array or list of current values for which to calculate and plot the flux.
    - solution (np.ndarray): An array of polynomial coefficients that define the relationship between current and flux. The coefficients should be in ascending order of power.

    Returns:
    - Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and axes objects, allowing for further customization of the plot after the function call.

    Note:
    - The function does not call plt.show() itself, giving the caller control over when and how the plot is displayed (e.g., displaying immediately with plt.show() or saving to a file).
    """
    
    # Calculate min and max from Is to use as range for x values
    min_I = min(Is)
    max_I = max(Is)
    xs = np.linspace(min_I, max_I, 1001)

    # Create the figure and axes object
    fig, ax = plt.subplots()

    # Generate y values using the current_to_flux function
    ys = np.array([current_to_flux(current, solution) for current in xs])

    # Plotting
    ax.plot(xs, ys)
    ax.set_ylabel('Flux')
    ax.set_xlabel('Current')

    # You can still call plt.show() outside the function if you want to display the plot immediately
    # plt.show()

    return fig, ax
