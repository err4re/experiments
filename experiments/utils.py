import numpy as np

from typing import Sequence, Union, List, Tuple

import math

def S_to_dBm(S: Union[np.ndarray, complex]) -> Union[np.ndarray, float]:
    return 20 * np.log10(np.abs(S))

def round_to_significant_digits(value: float, digits: int = 3) -> float:
    if value == 0:
        return 0  # Special case for zero to avoid math domain error
    
    # Calculate the order of magnitude of the value (i.e., how many digits before or after the decimal point)
    magnitude = int(math.floor(math.log10(abs(value))))
    
    # Calculate how much to round based on the number of significant digits
    factor = 10 ** (digits - 1 - magnitude)
    
    # Round to the nearest integer after shifting the decimal point
    return round(value * factor) / factor

def is_linearly_spaced(array: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Check if the array has linear spacing.

    Parameters:
    - array: np.ndarray : The array to check.
    - tol: float : Tolerance for numerical comparisons. Default is 1e-9.

    Returns:
    - bool : True if the array is linearly spaced, False otherwise.
    """
    if array.size < 2:
        return True  # An array with less than 2 elements is trivially linearly spaced

    # Calculate the differences between consecutive elements
    diffs = np.diff(array)

    # Check if all differences are approximately equal
    return np.all(np.abs(diffs - diffs[0]) < tol)


def highest_average_triplet(arr):
    """
    This function finds the center index of the highest average triplet in an array.


    Parameters:
    - arr (list of float/int): The array in which to find the highest average triplet. The array must contain 
      at least three elements.

    Returns:
    - int: The center index of the highest average triplet in the array.

    Raises:
    - ValueError: If the input array contains fewer than three elements.

    Example:
    >>> find_highest_average_triplet([1, 2, 3, 4, 5])
    3
    This indicates that the triplet [3, 4, 5] has the highest average, and the center index of this triplet is 3.

    Note:
    - The function assumes that the input array will only contain numeric values (integers or floats).
    - The index returned is 0-based, meaning that the first element of the array is at index 0.
    """

    if len(arr) < 3:
        raise ValueError("Array must have at least 3 elements")

    max_avg = float('-inf')
    max_index = 0

    # Loop through the array, stopping 2 elements before the end
    for i in range(len(arr) - 2):
        # Calculate the average of the current triplet
        current_avg = (arr[i] + arr[i+1] + arr[i+2]) / 3

        # Update the maximum average and index if necessary
        if current_avg > max_avg:
            max_avg = current_avg
            max_index = i

    #return center index of max average triplet
    return max_index+1

def highest_average_nlet(arr, n=5):
    """
    This function finds the center index of the highest average triplet in an array.


    Parameters:
    - arr (list of float/int): The array in which to find the highest average triplet. The array must contain 
      at least three elements.

    Returns:
    - int: The center index of the highest average triplet in the array.

    Raises:
    - ValueError: If the input array contains fewer than three elements.

    Example:
    >>> find_highest_average_triplet([1, 2, 3, 4, 5])
    3
    This indicates that the triplet [3, 4, 5] has the highest average, and the center index of this triplet is 3.

    Note:
    - The function assumes that the input array will only contain numeric values (integers or floats).
    - The index returned is 0-based, meaning that the first element of the array is at index 0.
    """

    if len(arr) < n:
        raise ValueError("Array must have at least 3 elements")

    max_avg = float('-inf')
    max_index = 0

    # Loop through the array, stopping 2 elements before the end
    for i in range(len(arr) - n-1):
        # Calculate the average of the current triplet
        current_avg = (np.sum([arr[i+j] for j in range(n)])) / n

        # Update the maximum average and index if necessary
        if current_avg > max_avg:
            max_avg = current_avg
            max_index = i

    #return center index of max average triplet
    return max_index+n//2


def background(xs: Sequence[Union[float, int]], 
               background_xs: Sequence[Union[float, int]], 
               background_ys: Sequence[Union[float, int, complex]]) -> np.ndarray:
    """
    Estimates the background values for a new set of x-coordinates based on a given set of background data points.

    This function uses one-dimensional linear interpolation to estimate the y-values (background values) at specific 
    x-coordinates. It is useful for signal correction, such as subtracting background noise or adjusting for baseline 
    drift in data analysis.

    Parameters:
    - xs (Sequence[Union[float, int]]): The x-coordinates at which to estimate the background values. Can be a list or an array.
    - background_xs (Sequence[Union[float, int]]): The x-coordinates of the known background data points.
    - background_ys (Sequence[Union[float, int, complex]]): The y-values of the known background data points.

    Returns:
    - np.ndarray: An array of interpolated y-values corresponding to each x in `xs`.

    Example:
    --------
    >>> xs = [0.5, 1.5, 2.5, 3.5]
    >>> background_xs = [0, 1, 2, 3, 4]
    >>> background_ys = [10, 12, 14, 16, 18]
    >>> estimated_background = background(xs, background_xs, background_ys)
    >>> print(estimated_background)
    [11. 13. 15. 17.]
    """
    return np.interp(xs, background_xs, background_ys)


def flatten_background_horizontally(x: np.ndarray, y: np.ndarray, z: np.ndarray, y_range: tuple = None, x_range: tuple = None) -> np.ndarray:
    """
    Subtracts the horizontal background from a 2D dataset within specified x and y ranges.
    
    The function calculates the average background signal within the specified x range for each row in the y range,
    and subtracts this average from the entire row. If no ranges are specified, it uses the entire range of x or y.
    
    Parameters:
    - x (np.ndarray): 1D array (or 2D array) of x coordinates.
    - y (np.ndarray): 1D array (or 2D array) of y coordinates.
    - z (np.ndarray): 2D dataset corresponding to the values at each (x, y) coordinate.
    - y_range (tuple, optional): Tuple specifying the min and max y values to consider for background subtraction.
                                If None, uses the full range of y.
    - x_range (tuple, optional): Tuple specifying the min and max x values to use for calculating the background.
                                If None, calculates background using the first 1% of the x range.
    
    Returns:
    - np.ndarray: The 2D dataset after horizontal background subtraction.
    """
    if y_range is None:
        y_min, y_max = y.min(), y.max()
    else:
        y_min, y_max = y_range
    
    if x_range is None:
        x_min, x_max = x.min(), x.min() + (x.max() - x.min())/100
    else:
        x_min, x_max = x_range

    z_flat_background = z.copy()
    xs = x if x.ndim == 1 else x[: , 0]
    ys = y if y.ndim == 1 else y[0 , :]


    for j in range(z.shape[0]):  # Assuming z is 2D: rows correspond to y, columns to x
        if ys[j] >= y_min and ys[j] <= y_max:
            background_indices = np.where((xs >= x_min) & (xs <= x_max))[0]
            if len(background_indices) > 0:
                background_avg = np.mean(z[j, background_indices])
                z_flat_background[j, :] -= background_avg
    

    return z_flat_background


def flatten_background_vertically(x: np.ndarray, y: np.ndarray, z: np.ndarray, y_range: tuple = None, x_range: tuple = None) -> np.ndarray:
    """
    Subtracts the vertical background from a 2D dataset within specified x and y ranges.
    
    This function leverages flatten_background_horizontally by transposing the input dataset and swapping the roles
    of x and y. It effectively performs background subtraction vertically within the specified ranges. After processing,
    it transposes the result back to the original orientation.
    
    Parameters:
    - x (np.ndarray): 1D array (or 2D array) of x coordinates.
    - y (np.ndarray): 1D array (or 2D array) of y coordinates.
    - z (np.ndarray): 2D dataset corresponding to the values at each (x, y) coordinate.
    - y_range (tuple, optional): If None, calculates background using the first 1% of the y range.
                                 Tuple specifying the min and max x values to consider for background subtraction,
                                 effectively acting on the vertical axis after transpose.
    - x_range (tuple, optional): Tuple specifying the min and max y values to use for calculating the background,
                                 effectively acting on the horizontal axis after transpose.
    
    Returns:
    - np.ndarray: The 2D dataset after vertical background subtraction.
    """
    z_flat_background = flatten_background_horizontally(x=y, y=x, z=z.T, y_range=x_range, x_range=y_range)

    return z_flat_background.T


def generate_mask(arr: np.ndarray, limits: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generates a boolean mask for an array based on specified range limits.

    This function creates a mask array of the same length as the input array `arr`.
    Each element in the mask is set to True if the corresponding element in `arr`
    falls within any of the specified ranges in `limits`; otherwise, it is set to False.

    Parameters:
    - arr (np.ndarray): A 1D numpy array of numerical values for which the mask will be generated.
    - limits (List[Tuple[float, float]]): A list of tuples, where each tuple contains two floats
      representing the lower and upper bounds of a range, inclusive.

    Returns:
    - np.ndarray: A boolean numpy array where each True value indicates that the corresponding
      element in `arr` falls within one of the specified ranges.

    Example:
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> limits = [(1.5, 3.5), (4.5, 5.5)]
    >>> mask = generate_mask(arr, limits)
    >>> print(mask)
    [False  True  True False  True]
    
    In this example, elements 2, 3, and 5 in `arr` fall within the specified ranges, so the mask
    for those positions is set to True.
    """
    mask = np.zeros(len(arr), dtype=bool)

    # Iterate over each range and update the mask
    for low, high in limits:
        # Update mask for each range
        mask = mask | ((arr >= low) & (arr <= high))

    return mask