from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Any, List, TypedDict, Union, Optional, Callable
from enum import Enum

import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib
import functools

import h5py

from instruments import znb, anapico, yoko7651
from instruments.configs.yoko7651_config import YokoCurrSweepConfig
from instruments.configs.znb_config import ZnbLinConfig, ZnbSegm, ZnbSegmConfig, ZnbExtTrigOutConfig
from instruments.configs.anapico_config import AnaFreqSweepConfig, AnaExtTrigInConfig

from experiments.plotter import Plotter
from experiments.experiment_data import ExperimentData, State
from experiments.experiment_config import ExperimentConfig



class InstrumentDict(TypedDict):
    """
    A dictionary type that maps instrument names to their respective instances.

    This TypedDict is used to define the types of instruments that can be
    included in an experiment setup, providing a clear and consistent way
    to manage multiple instruments and keep autocompletion alive.

    Attributes:
        vna (znb.Znb): An instance of the Znb class, representing a Vector Network Analyzer.
        ana (anapico.AnaPico): An instance of the AnaPico class, representing a signal generator.
        yoko (yoko7651.Yoko7651): An instance of the Yoko7651 class, representing a Yokogawa precision source.
    """
    vna: znb.Znb
    ana: anapico.AnaPico
    yoko: yoko7651.Yoko7651


class ConfigurationDict(TypedDict):
    """
    A TypedDict that defines the configuration types for various instruments used in experiments.

    This dictionary type is designed to hold configuration objects for different instruments, 
    providing a standardized way to store and access these configurations.

    Attributes:
        vna (Union[ZnbSegmConfig, ZnbLinConfig]): 
            Configuration for the Vector Network Analyzer (VNA). It can be either a segmented sweep 
            configuration (ZnbSegmConfig) or a linear sweep configuration (ZnbLinConfig).
        
        vna_trigger (ZnbExtTrigOutConfig): 
            Configuration for the external trigger settings of the VNA. It defines parameters 
            such as trigger enable state, interval, position, and output polarity.
        
        yoko (YokoCurrSweepConfig): 
            Configuration for the Yokogawa current source. This includes settings for current 
            sweeps, compliance limits, and other relevant parameters.
        
        ana (AnaFreqSweepConfig): 
            Configuration for the Anapico frequency generator. It specifies the details for 
            frequency sweeps and other signal generation parameters.
        
        ana_trigger (AnaExtTrigInConfig): 
            Configuration for the external trigger settings of the Anapico frequency generator. 
            This includes settings for trigger modes, thresholds, and other trigger-related parameters.
    """
    vna: Union[ZnbSegmConfig, ZnbLinConfig]
    vna_trigger: ZnbExtTrigOutConfig
    yoko: YokoCurrSweepConfig
    ana: AnaFreqSweepConfig
    ana_trigger: AnaExtTrigInConfig






@dataclass
class Experiment:
    """
    Base class for all experiments.

    Attributes:
        instruments (InstrDict): A dictionary of instruments used in the experiment.
        configs_dict (Dict[str, Any]): A dictionary of configurations for the instruments.
        data (List[Any]): A dictionary to store the data collected during the experiment.
        start_time (datetime): The start time of the experiment.
        end_time (datetime): The end time of the experiment.
        state (str): The current state of the experiment (e.g., 'initialized', 'running', 'completed').
        sample_name (str): The name of the sample being tested or measured.
        file_directory (str): The directory for saving experiment data.
        file_name (str): The file name for saving experiment data.
    """
    sample_name: str
    sample_code: str
    measurement_code: str

    config: ExperimentConfig

    file_directory: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None

    vna: Optional[znb.Znb] = None
    ana: Optional[anapico.AnaPico] = None
    yoko: Optional[yoko7651.Yoko7651] = None

    data: Optional[ExperimentData] = None #dataclass to save data from specific experiments
    data_dict: Dict[str, Any] = field(default_factory=dict) #legacy stuff to handle .npz

    plotter: Plotter = None
    


    def start_experiment(self):
        """Start the experiment, including instrument initialization."""
        self.data.start_time = datetime.datetime.now()
        self.data.state = State.RUNNING
        # Implementation of experiment start...
        self.initialize_instruments()

    def pause_experiment(self):
        pass

    def end_experiment(self, interrupted: bool = False):
        """End the experiment, including releasing the instruments."""
        self.data.end_time = datetime.datetime.now()

        if interrupted:
            self.data.state = State.INTERRUPTED
        else:
            self.data.state = State.COMPLETED

        # Implementation of experiment end...
        self.release_instruments()

        plt.close('all')

        self.save_data_hdf5()
        

    def experiment_method(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self: Experiment, *args, **kwargs):
            # Call the common start_experiment method
            self.start_experiment()
            
            # Set the measurement type to the method's name
            self.data.measurement_function = method.__name__

            interrupted = False  # Track if the experiment was interrupted
        
            try:
                # Call the actual method and store the result or None
                result = method(self, *args, **kwargs)
            except KeyboardInterrupt:
                # Handle kernel interrupt (e.g., Ctrl+C or Jupyter "Interrupt" button)
                interrupted = True
                print("Experiment interrupted by user or kernel.")
                raise  # Re-raise to stop the execution properly after cleanup
            finally:
                # Ensure end_experiment is called after the method, passing the interrupted flag
                self.end_experiment(interrupted=interrupted)

            return result
        
        return wrapper
    


    # Instrument methods
    def initialize_instruments(self):
        """ Initialize the instruments based on the provided configurations. """
        raise NotImplementedError
    
    def initialize_vna(self, adress = "TCPIP0::192.168.0.43::hislip::INSTR"):
        self.vna = znb.Znb(adress) 
        
    def initialize_ana(self, adress = "TCPIP0::192.168.0.45"):
        self.ana = anapico.AnaPico(adress)
        
    def initialize_yoko(self, adress = "GPIB1::1::INSTR"):
        self.yoko = yoko7651.Yoko7651(adress)

    def release_instruments(self):
        if self.vna is not None:
            self.release_vna()
        if self.ana is not None:
            self.release_ana()
        if self.yoko is not None:
            self.release_yoko()
                
    def release_vna(self):
        self.vna.clean()
        del self.vna

    def release_ana(self):
        self.ana.clean()
        del self.ana

    def release_yoko(self):
        self.yoko.clean()
        del self.yoko
        

    def initialize_data(self):
        """ Initialize the data structure based on the requirements for the specific experiment. """
        raise NotImplementedError            


    # File management methods
    def _generate_file_name(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{self.sample_name}"
    
    
    def _generate_unique_filename(self, extension: str = '.h5') -> str:
        """
        Generate a unique file name by appending a counter if the file already exists.
        
        :param base_name: The base name of the file (without extension).
        :param directory: The directory where the file will be saved.
        :param extension: The file extension (default is '.h5').
        :return: A unique file name with the given extension.
        """
        counter = 1
        base_name = self._generate_base_name()
        file_name = f"{base_name}_{counter}"
        full_path = os.path.join(self.file_directory, file_name+extension)
        # Increment the counter until a unique file name is found
        while os.path.exists(full_path):
            file_name = f"{base_name}_{counter}"
            full_path = os.path.join(self.file_directory, file_name+extension)
            counter += 1
        
        return file_name
    
    def _generate_base_name(self) -> str:
        file_name = self.sample_code + self.measurement_code
        return file_name
    
    def save_data_hdf5(self, file_name: Optional[str] = None, compress: bool = True) -> None:
        start_time = time.time()

        #generate unique file name by numbering files in ascending order
        if file_name is None:
            self.file_name = self._generate_unique_filename(extension='.h5')
        else:
            self.file_name = file_name

        full_path = os.path.join(self.file_directory, self.file_name+'.h5')

        self.save_dataclass_to_hdf5(self.data, full_path, compress)
        self.file_path = full_path

        end_time = time.time()
        duration = end_time - start_time

        print(f"Data saved to {full_path}")
        if compress:
            print(f"Time taken with compression: {duration:.2f} seconds")
        else:
            print(f"Time taken without compression: {duration:.2f} seconds")

   
    def save_dataclass_to_hdf5(self, dataclass_instance, file_path: str, compress: bool = True) -> None:
        if not is_dataclass(dataclass_instance):
            raise ValueError("The provided instance is not a dataclass.")
        
        with h5py.File(file_path, 'w') as h5file:
            h5file.attrs['dataclass_type'] = type(dataclass_instance).__name__
            
            for field in fields(dataclass_instance):
                value = getattr(dataclass_instance, field.name)
                if value is None:
                    h5file.attrs[field.name] = "None"  # Store a marker for None values
                elif isinstance(value, np.ndarray):
                    if compress:
                        h5file.create_dataset(field.name, data=value, compression='lzf')
                    else:
                        h5file.create_dataset(field.name, data=value)
                elif isinstance(value, dict):
                    # Handle dictionary attributes
                    group = h5file.create_group(field.name)
                    self._save_dict_to_group(group, value)
                elif isinstance(value, datetime.datetime):
                    h5file.attrs[field.name] = value.isoformat()
                elif isinstance(value, State):
                    h5file.attrs[field.name] = value.value
                
                else:
                    h5file.attrs[field.name] = value

    @staticmethod
    def _save_dict_to_group(group: h5py.Group, dictionary: dict):
        for key, value in dictionary.items():
            if value is None:
                group.attrs[key] = "None"
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                Experiment._save_dict_to_group(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            else:
                group.attrs[key] = value

    @staticmethod
    def load_dataclass_from_hdf5(file_path: str, data_module_name: str = 'experiments.experiment_data') -> Any:
        with h5py.File(file_path, 'r') as h5file:
            dataclass_type_name = h5file.attrs['dataclass_type']
            
            # Dynamically import the module and get the dataclass type
            data_module = importlib.import_module(data_module_name)
            dataclass_type = getattr(data_module, dataclass_type_name)
            
            data_dict = {}
            for field in fields(dataclass_type):
                if field.name in h5file:
                    if isinstance(h5file[field.name], h5py.Dataset):
                        data_dict[field.name] = h5file[field.name][()]
                    elif isinstance(h5file[field.name], h5py.Group):
                        data_dict[field.name] = Experiment._load_dict_from_group(h5file[field.name])
                else:
                    attr_value = h5file.attrs.get(field.name)
                    if attr_value == "None":
                        data_dict[field.name] = None
                    elif isinstance(attr_value, str) and field.type == datetime.datetime:
                        data_dict[field.name] = datetime.datetime.fromisoformat(attr_value)
                    else:
                        data_dict[field.name] = attr_value

            return dataclass_type(**data_dict)

    @staticmethod
    def _load_dict_from_group(group: h5py.Group) -> dict:
        dictionary = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                dictionary[key] = item[()]
            elif isinstance(item, h5py.Group):
                dictionary[key] = Experiment._load_dict_from_group(item)
        for key, attr in group.attrs.items():
            if attr == "None":
                dictionary[key] = None
            else:
                dictionary[key] = attr
        return dictionary

    


    def save_data_npz(self, file_name=None, compress=True):
        if file_name is None:
            self.file_name = self._generate_file_name()
        else: 
            self.file_name = file_name

        full_path = os.path.join(self.file_directory, self.file_name)
        if compress:
            np.savez_compressed(full_path, **self.data_dict)
        else:
            np.savez(full_path, **self.data_dict)

        self.file_path = full_path
        print(f"Data saved to {full_path}.npz")

    
    



    
    # Plotting methods
    # live plotting?

    








