from dataclasses import dataclass, asdict, replace
from typing import Union,Type, Tuple, Optional
from enum import Enum

import datetime
import numpy as np

class State(Enum):
    """
    Enum representing the various states of data acquisition.

    The possible states are:
    
    - `INITIALIZED`: The experiment has been initialized but has not yet started.
    - `RUNNING`: The experiment is currently in progress.
    - `COMPLETED`: The experiment has finished successfully.
    - `INTERRUPTED`: The experiment was interrupted before completion.
    """
    INITIALIZED = 'initialized'
    RUNNING = ' running'
    COMPLETED = 'completed'
    INTERRUPTED = 'interrupted'

@dataclass
class ExperimentData:
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    state: Optional[State] = None
    measurement_function: Optional[str] = None


#`kw_only=True` needed to allow for default arguments in parent class ExperimentData
@dataclass(kw_only=True)
class FluxMapData(ExperimentData):
    #1d array with the coil currents to sweep
    currents: np.ndarray[np.float64] = None
   
    #1d array with the coil voltages to sweep
    voltages: np.ndarray[np.float64] = None

    #1d array of the VNA frequencies inside one trace, (potentially 2d for adaptive segmented VNA sweep)
    f: np.ndarray[np.float64] = None

    #2d array of the S parameters, i.e. S[current][f]
    S: np.ndarray[np.complex128] = None
    
    #VNA meta data, e.g. bandwidth, power, averaging
    vna_meta: Optional[dict] = None

    #corresponding fluxes
    fluxes: Optional[np.ndarray[np.float64]] = None

    #powers (for power sweep)
    powers: np.ndarray[np.float64] = None


@dataclass(kw_only=True)
class TwoToneData(ExperimentData):
    

    f1_center_frequencies: np.ndarray

    f2_center_frequencies: np.ndarray
    f2_frequencies: np.ndarray
    f2_powers: np.ndarray

    f: np.ndarray[np.float64]
    S: np.ndarray[np.complex128]

    f_ref: np.ndarray[np.float64]
    S_ref: np.ndarray[np.complex128]

    currents: np.ndarray[np.float64] = None
    voltages: np.ndarray[np.float64] = None

    f1_powers: Optional[np.ndarray] = None
    
    signal: Optional[np.ndarray] = None

    f_flux_map_rough: Optional[np.ndarray] = None
    S_flux_map_rough: Optional[np.ndarray] = None
    f_flux_map_fine: Optional[np.ndarray] = None
    S_flux_map_fine: Optional[np.ndarray] = None

    vna_meta:  Optional[dict] = None

    fluxes: Optional[np.ndarray] = None