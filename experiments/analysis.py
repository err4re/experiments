from typing import Type, Tuple, Union, List

import numpy as np
from qutip import Qobj



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



