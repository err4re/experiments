from instruments import znb, anapico, yoko7651

from experiments.experiment import Experiment
from experiments.plotter import Plotter
from instruments import znb, anapico, yoko7651
from instruments.configs.znb_config import ZnbLinConfig, ZnbSegmConfig, ZnbExtTrigOutConfig
from instruments.configs.yoko7651_config import YokoCurrSweepConfig, YokoVoltSweepConfig
from instruments.configs.anapico_config import AnaFreqSweepConfig, AnaExtTrigInConfig

import numpy as np
from dataclasses import dataclass, asdict, replace
from typing import Union,Type, Tuple, Optional

from time import sleep

import matplotlib.pyplot as plt

from qutip import Qobj  

from experiments.utils import S_to_dBm

import time

from experiments.experiment_data import TwoToneData
from experiments.experiment_config import TwoToneConfig, ExperimentConfig



class TwoToneExperiment(Experiment):

    config: TwoToneConfig
    data: TwoToneData
    measurement_code: str = 'TT'


    def __init__(self, sample_name: str, sample_code: str, file_directory: str, config: TwoToneConfig, file_name: str = None):
        
        super().__init__(sample_name=sample_name, 
                         sample_code=sample_code,
                         measurement_code=self.measurement_code,
                         file_directory=file_directory, 
                         file_name=file_name, 
                         config=config)
        
        self.data = self.initialize_data()

        self.plotter = Plotter()
    

        if self.config.tracking_parameters is not None:
            self.initialize_f1_to_f2_tracking()


    def initialize_instruments(self):
        self.initialize_vna()
        self.initialize_ana()
        self.initialize_yoko()


    def initialize_data(self) -> TwoToneData:

        if self.config.ana_f2.power_sweep_length is not None:
            sweep_length = self.config.ana_f2.power_sweep_length
            if isinstance(self.config.yoko, YokoVoltSweepConfig):
                voltages = np.full(sweep_length, np.nan, dtype=np.float64)
                currents = None
            elif isinstance(self.config.yoko, YokoCurrSweepConfig):
                currents = np.full(sweep_length, np.nan, dtype=np.float64)
                voltages = None
        elif isinstance(self.config.yoko, YokoCurrSweepConfig):
            sweep_length = self.config.yoko.sweep_length
            currents = np.full(sweep_length, np.nan, dtype=np.float64)
            voltages = None
        elif isinstance(self.config.yoko, YokoVoltSweepConfig):
            sweep_length = self.config.yoko.sweep_length
            voltages = np.full(sweep_length, np.nan, dtype=np.float64)
            currents = None

        
        f = np.full((sweep_length, self.config.ana_f2.num_points, self.config.vna_f1.num_points), np.nan, dtype=np.float64)
        S = np.full((sweep_length, self.config.ana_f2.num_points, self.config.vna_f1.num_points), np.nan, dtype=complex)

        f_ref = np.full((sweep_length, self.config.vna_f1.num_points), np.nan, dtype=np.float64)
        S_ref = np.full((sweep_length, self.config.vna_f1.num_points), np.nan, dtype=complex)


        vna_meta = []

        f1_center_frequencies = np.full(sweep_length, np.nan)

        f2_center_frequencies = np.full(sweep_length, np.nan)
        
        
        f2_powers = np.full(sweep_length, np.nan)

        f2_frequencies = np.full((sweep_length, self.config.ana_f2.num_points), np.nan)

        signal = np.full((sweep_length, self.config.ana_f2.num_points), np.nan, dtype=np.float64)

        if self.config.tracking_parameters is not None:
            fluxes = np.full(sweep_length, np.nan)
            return TwoToneData(currents=currents,
                               voltages=voltages,
                               f=f,
                               S=S,
                               f_ref=f_ref,
                               S_ref=S_ref,
                               f1_center_frequencies=f1_center_frequencies,
                               f2_center_frequencies=f2_center_frequencies,
                               f2_powers=f2_powers,
                               f2_frequencies= f2_frequencies,
                               signal=signal,
                               vna_meta=vna_meta,
                               fluxes=fluxes
                               )
        
        else:
            return TwoToneData(currents=currents,
                               voltages=voltages,
                               f=f,
                               S=S,
                               f_ref=f_ref,
                               S_ref=S_ref,
                               f1_center_frequencies=f1_center_frequencies,
                               f2_center_frequencies=f2_center_frequencies,
                               f2_powers=f2_powers,
                               f2_frequencies= f2_frequencies,
                               signal=signal,
                               vna_meta=vna_meta
                               )

        


    def initialize_f1_to_f2_tracking(self):
        self.fc = self.config.tracking_parameters['fc']
        self.tolerance = self.config.tracking_parameters['tolerance']
        self.upper_branch_spline = self.config.tracking_parameters['upper_spline']
        self.lower_branch_spline = self.config.tracking_parameters['lower_spline']

        self.Ec = self.config.tracking_parameters['Ec']
        self.Ej = self.config.tracking_parameters['Ej']



    def save_data(self, filename=None):
        self.data_dict = asdict(self.data)
        super().save_data_npz(filename)
        
    


    def calibrate_f1(self) -> float:

        #setup sweep on VNA
        self.vna.set_sweep(self.config.vna_f1_calib)
        f_calib, z_calib = self.vna.sweep()

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))]

        vna_f1_calib_detailed = replace(self.config.vna_f1_calib)
        vna_f1_calib_detailed.center_frequency = cw_frequency
        vna_f1_calib_detailed.span = self.config.vna_f1_calib.span/10
        vna_f1_calib_detailed.num_points = int(self.config.vna_f1_calib.num_points/2)

        self.vna.set_sweep(vna_f1_calib_detailed)
        f_calib, z_calib = self.vna.sweep()

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))]

        return cw_frequency
    
    def calibrate_f1_z(self) -> float:

        #setup sweep on VNA
        self.vna.set_sweep(self.config.vna_f1_calib)
        f_calib, z_calib = self.vna.sweep()

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))]

        vna_f1_calib_detailed = replace(self.config.vna_f1_calib)
        vna_f1_calib_detailed.center_frequency = cw_frequency
        vna_f1_calib_detailed.span = self.config.vna_f1_calib.span/10
        vna_f1_calib_detailed.num_points = int(self.config.vna_f1_calib.num_points/2)

        self.vna.set_sweep(vna_f1_calib_detailed)
        f_calib, z_calib = self.vna.sweep()

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))]

        return cw_frequency, f_calib, z_calib
    
    def calibrate_f1_masked(self, masked_frequencies) -> float:

        #setup sweep on VNA
        self.vna.set_sweep(self.config.vna_f1_calib)
        f_calib, z_calib = self.vna.sweep()

        mask = self.mask_devices(f_calib, masked_frequencies)

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))]

        vna_f1_calib_detailed = replace(self.config.vna_f1_calib)
        vna_f1_calib_detailed.center_frequency = cw_frequency
        vna_f1_calib_detailed.span = self.config.vna_f1_calib.span/10
        vna_f1_calib_detailed.num_points = int(self.config.vna_f1_calib.num_points/2)

        self.vna.set_sweep(vna_f1_calib_detailed)
        f_calib, z_calib = self.vna.sweep()

        mask = self.mask_devices(f_calib, masked_frequencies)

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))]

        return cw_frequency
    
    def calibrate_f1_z_masked(self, masked_frequencies) -> float:

        #setup sweep on VNA
        self.vna.set_sweep(self.config.vna_f1_calib)
        f_calib, z_calib = self.vna.sweep()

        mask = self.mask_devices(f_calib, masked_frequencies)

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))]

        vna_f1_calib_detailed = replace(self.config.vna_f1_calib)
        vna_f1_calib_detailed.center_frequency = cw_frequency
        vna_f1_calib_detailed.span = self.config.vna_f1_calib.span/10
        vna_f1_calib_detailed.num_points = int(self.config.vna_f1_calib.num_points/2)

        self.vna.set_sweep(vna_f1_calib_detailed)
        f_calib, z_calib = self.vna.sweep()

        mask = self.mask_devices(f_calib, masked_frequencies)

        cw_frequency = f_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))]

        return cw_frequency, f_calib, z_calib
    
    @staticmethod
    #[(0,4e9), (6.44e9, 6.49e9), (7.175e9, 7.2e9), (8.05e9, 8.08e9), (8.4e9, 8.48e9), (8.1e9, 10e9)]
    def mask_devices(frequencies, devices = [(8.694e9, 8.6943e9)]):

        mask = np.zeros(len(frequencies), dtype=bool)

        # Iterate over each range and update the mask
        for low, high in devices:
            # Update mask for each range
            mask = mask | ((frequencies >= low) & (frequencies <= high))

        return mask
    
    @staticmethod
    def flatten_background_horizontally(fs, zs, background_fs, background_zs):

        background = np.interp(fs, background_fs, background_zs)

        return zs - background + np.average(np.abs(background))
    
    
    def calibrate_f1_and_flux_map(self, mask_devices=True) -> Tuple[float, np.ndarray, np.ndarray]:

        #setup sweep on VNA
        self.vna.set_sweep(self.config.vna_f1_calib)
        f_calib_rough, z_calib_rough = self.vna.sweep()

        limits = [0, 6e9]

        #subtract background
        if not np.isnan(self.data.f1_center_frequencies[0]):

            mask_rough = np.zeros(len(f_calib_rough), dtype=bool)
            mask_rough = mask_rough | ((f_calib_rough >= limits[0]) & (f_calib_rough <= limits[1]))

            fig, axes = self.plotter.plot_trace_mag_phase(f_calib_rough, z_calib_rough-1e-3)
            plt.show()
            
            z_calib_rough[mask_rough] = self.flatten_background_horizontally(f_calib_rough[mask_rough], z_calib_rough[mask_rough], self.data.f_flux_map_rough[0][mask_rough], self.data.S_flux_map_rough[0][mask_rough])
            
            fig, axes = self.plotter.plot_trace_mag_phase(f_calib_rough, z_calib_rough-1e-3)
            plt.show()

        
        if mask_devices:

            mask = self.mask_devices(f_calib_rough)

            cw_frequency = f_calib_rough[~mask][self.find_highest_average_triplet(-(np.abs(z_calib_rough[~mask])))]

        else:
            cw_frequency = f_calib_rough[self.find_highest_average_triplet(-(np.abs(z_calib_rough)))]

        vna_f1_calib_detailed = replace(self.config.vna_f1_calib)
        vna_f1_calib_detailed.center_frequency = cw_frequency
        vna_f1_calib_detailed.span = self.config.vna_f1_calib.span/10
        vna_f1_calib_detailed.num_points = int(self.config.vna_f1_calib.num_points/2)

        self.vna.set_sweep(vna_f1_calib_detailed)
        f_calib_fine, z_calib_fine = self.vna.sweep()

        #subtract background
        if not np.isnan(self.data.f1_center_frequencies[0]):

            mask_fine = np.zeros(len(f_calib_fine), dtype=bool)
            mask_fine = mask_fine | ((f_calib_fine >= limits[0]) & (f_calib_fine <= limits[1]))
            z_calib_fine[mask_fine] = self.flatten_background_horizontally(f_calib_fine[mask_fine], z_calib_fine[mask_fine], self.data.f_flux_map_rough[0][mask_rough], self.data.S_flux_map_rough[0][mask_rough])

            fig,axes = self.plotter.plot_trace_mag_phase(f_calib_fine, z_calib_fine-1e-3)
            plt.show()
            

        if mask_devices:

            mask = self.mask_devices(f_calib_fine)

            cw_frequency = f_calib_fine[~mask][self.find_highest_average_triplet(-(np.abs(z_calib_fine[~mask])))]

        else:
            cw_frequency = f_calib_fine[self.find_highest_average_triplet(-(np.abs(z_calib_fine)))]

        return cw_frequency, f_calib_rough, z_calib_rough, f_calib_fine, z_calib_fine
    

    
    def phase_shift_signal(self, S):

        root_mean_squares = np.zeros(len(S))

        for i,s in enumerate(S):

            root_mean_squares[i] = np.sqrt(np.mean((np.angle(s) - np.angle(S[0]))**2))

        return root_mean_squares
    
    def unwrapped_phase_shift_signal(self, S, S_ref):

        # signal = np.zeros(len(S))

        # for i,s in enumerate(S):

        #     signal[i] = np.mean(2*np.arcsin(1/2 * np.abs( np.exp(1j*np.angle(s)) - np.exp(1j*np.angle(S[0]))) ))

        # return signal

        signal = np.zeros(len(S))

        for i,s in enumerate(S):

            signal[i] = np.mean(np.angle( np.exp(1j*np.angle(s)) / np.exp(1j*np.angle(S_ref))) )

        return signal
    
    def full_S_parameter_signal(self, S, S_ref):

        signal = np.zeros(len(S))

        for i,s in enumerate(S):

            signal[i] = np.sum(np.abs(s - S_ref))

        return signal

    def f2_to_power(self, f2):

        powers = [-25, -32]
        frequencies = [16e9, 11e9]

        if f2 > 11e9:
            m = (powers[1] - powers[0])/(frequencies[1] - frequencies[0])
            t = powers[0] - m*frequencies[0]

            power = m*f2 + t
        elif f2 < 6e9:
            power = -30
            if f2 + self.config.ana_f2.span/2 < 3.6e9:
                power = -10
        else:
            power = -32

        return float(np.round(power,1))
    
    def f2_to_span(self, f2):

        spans = [0.5e6, 2e6]
        frequencies = [16e9, 11e9]

        if f2 > 11e9:
            m = (spans[1] - spans[0])/(frequencies[1] - frequencies[0])
            t = spans[0] - m*frequencies[0]

            span = m*f2 + t
        else:
            span = 2e6

        return float(np.round(span))
    
    def f2_power_compensate_filter(self, f2):
        filter_points = [(1e9, -28.45), (2.4e9, -30.21), (2.5e9, -31.98), (3e9, -30.01), (3.6e9, -27.73), (4e9, -5.76), (4.335e9, -3), (5e9, -1.7), (20e9, -1)]
        x_values, y_values = zip(*filter_points)

        return np.interp(f2, x_values, y_values)

    
    @staticmethod
    def find_highest_average_triplet(arr):
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

    def f2_tracking(self, i):

        f2_center_frequency = self.config.ana_f2.frequencies[self.find_highest_average_triplet(self.data.signal[i])]
        
        if i > 3 and np.sign(f2_center_frequency - self.data.f2_center_frequencies[i]) != np.sign(self.data.f1_center_frequencies[i] - self.data.f2_center_frequencies[i-1]) and abs(f2_center_frequency - self.data.f2_center_frequencies[i]) > 50e6:
            print('About to loose track of f2 center, following the slope of f2 center frequency based on previous points.')
            f2_center_frequency = self.data.f2_center_frequencies[i] + (self.data.f2_center_frequencies[i] - self.data.f2_center_frequencies[i-1])
        
        self.config.ana_f2.change_center_frequency(f2_center_frequency)



        # limit change in f2 as a function of the change in f1
        # if i > 5:
        #     if self.data.f1_center_frequencies[i] - self.data.f1_center_frequencies[i-1] != 0 and abs(f2_center_frequency - self.data.f2_center_frequencies[i-1]) > abs(self.data.f1_center_frequencies[i] - self.data.f1_center_frequencies[i-1]) *(4e9/2e6) *10:
        #         print('About to loose track of f2 center, keeping f2 center the same.')
        #     else:
        #         self.config.ana_f2.change_center_frequency(f2_center_frequency) 

        # else:
        #     self.config.ana_f2.change_center_frequency(f2_center_frequency)


    
    def f1_to_f2(self, f1):

        f1s = [6.458e9, 6.47e9]
        f2s = [10e9, 15e9]

        m = (f2s[1] - f2s[0])/(f1s[1] - f1s[0])
        t = f2s[0] - m*f1s[0]

        f2 = m*f1 + t

        return (np.round(f2/1e9,6))*1e9


    def find_f1_current(self, f1, precision, current_start, current_step) -> float:
        
        self.yoko.range_current(current_start)
        self.yoko.current(current_start)

        cw_freq = self.calibrate_f1()

        i = 0
        max_iterations = 1000
        while not (f1 - precision < cw_freq and cw_freq < f1 + precision):
            i += 1

            if np.abs(current_start - i * current_step) > self.yoko.range_i:
                self.yoko.range_current(np.abs(current_start - i * current_step))
            self.yoko.current(current_start - i * current_step)

            cw_freq = self.calibrate_f1()

            if i > max_iterations:
                print(f'Maximum iterations reached, correct current to obtain f1 = {f1} not found. Breaking.')
                break

        return current_start - i * current_step

    @staticmethod
    def f1_to_flux_spline(f1, tolerance, spline):

        # Specify the target y value
        target_y = f1

        x_range = np.linspace(spline.get_knots()[0], spline.get_knots()[-1], 10000)

        # Find the x values where the spline intersects the target y value
        x_intersections = [x_val for x_val in x_range if abs(spline(x_val) - target_y) < tolerance]  # Tolerance for closeness

        # Print the x values where the spline intersects the specified y value
        #print("X values for y =", target_y, ":", x_intersections)
        
        # Calculate the error for each intersection point
        errors = [abs(spline(x_val) - target_y) for x_val in x_intersections]

        if not errors:
            return None
        else:
            # Find the index of the intersection point with the lowest error
            index_of_lowest_error = errors.index(min(errors))

            # Get the x and y values of the intersection point with the lowest error
            x_lowest_error = x_intersections[index_of_lowest_error]
            y_lowest_error = spline(x_lowest_error)

            # Print the intersection point with the lowest error and its error value
            # print("Intersection with Lowest Error (x, y):", (x_lowest_error, y_lowest_error))
            # print("Lowest Error:", min(errors))

        return x_lowest_error *np.pi
    
    def f1_to_flux(self, f1, fc, tolerance, lower_branch_spline, upper_branch_spline):

        if f1 > fc:
            flux = self.f1_to_flux_spline(f1, tolerance, upper_branch_spline)
        elif f1 < fc:
            flux = self.f1_to_flux_spline(f1, tolerance, lower_branch_spline)

        return flux
    
    @staticmethod
    def hamiltonian_sym(Ec, Ej, phi_x, N, ng) -> Type[Qobj]:
        """
        Return the hamiltonian for a symmetric SQUID as a Qobj instance.
        """
        mc = np.diag(Ec * (np.arange(-N, N + 1) - ng) ** 2)
        mj =  0.5 * Ej * np.cos(phi_x/2) * (np.diag(-np.ones(2 * N), 1) + np.diag(-np.ones(2 * N), -1))

        m = mc + mj

        return Qobj(m)

    @staticmethod
    def hamiltonian_asym(Ec, Ej, d, phi_x, N, ng) -> Type[Qobj]:
        """
        Return the hamiltonian for an asymmetric SQUID as a Qobj instance.
        """
        mc = np.diag(Ec * (np.arange(-N, N + 1) - ng) ** 2)
        mj =  0.5 * Ej * np.cos(phi_x/2) * (np.diag(-np.ones(2 * N), 1) + np.diag(-np.ones(2 * N), -1)) + 0.5 * d * Ej * np.sin(phi_x/2) * (-1j)* (np.diag(-np.ones(2 * N), 1) - np.diag(-np.ones(2 * N), -1))

        m = mc + mj

        return Qobj(m)

    def flux_to_f2(self, Ec, Ej, flux, N=10):

        energies = self.hamiltonian_sym(Ec, Ej, flux, N, 0).eigenenergies()
        energy_diff = energies[1] - energies[0]

        return energy_diff
    
    def f1_to_f2_tracking(self, f1):

        flux = self.f1_to_flux(f1, self.fc, self.tolerance, self.lower_branch_spline, self.upper_branch_spline)
        if flux is not None:
            f2 = self.flux_to_f2(self.Ec, self.Ej, flux)
        else:
            f2 = None

        return flux, f2
    

    def two_tone_rough(self, f1, precision, current_start, current_step):

        self.start_experiment()

        #find starting current, flux pi
        # check yoko current, ramp slowly to starting value

        #turn current source on
        self.yoko.output(True)
        
        # ramp to initial current value
        self.yoko.range_current(current_start)
        self.yoko.ramp_current(current_start, blocking=True)

       
        initial_current = self.find_f1_current(f1, precision, current_start, current_step)
        print(f'initial current: {initial_current}')
        self.config.yoko.currents = self.config.yoko.currents - self.config.yoko.currents[0] + initial_current
        self.data.currents = self.config.yoko.currents

        self.yoko.set_source_current_sweep(self.config.yoko)

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)

            cw_frequency = self.calibrate_f1()
            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency


            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            self.ana.set_freq_sweep(self.config.ana_f2)
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()

            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)

            for j in range(self.config.ana_f2.num_points):

                f, z = self.vna.sweep()

                self.data.f[i][j] = f
                self.data.S[i][j] = z

            signal = self.phase_shift_signal(self.data.S[i])
            self.data.signal[i] = signal
            
            #f2_center_frequency = self.configs['ana_f2'].frequencies[np.argmax(signal)]
            # f2_center_frequency = self.f1_to_f2(cw_frequency)
            # self.config.ana_f2.change_center_frequency(f2_center_frequency)
            # #adapt two tone power?
            # self.config.ana_f2.power = self.f2_to_power(f2_center_frequency)


            meta = self.vna.get_meta()
            self.data.meta.append(meta)

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.currents/1e-3, self.data.signal)

            #save backup?


        #save data?
        
            

        self.end_experiment()


    def two_tone_power(self, powers):
        
        self.start_experiment()

        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            cw_frequency = self.calibrate_f1()
            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency


            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.config.ana_f2.power = powers[i]

            self.ana.set_ext_trig(self.config.ana_trigger)
            self.ana.set_freq_sweep(self.config.ana_f2)
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()

            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)

            for j in range(self.config.ana_f2.num_points):

                f, z = self.vna.sweep()

                self.data.f[i][j] = f
                self.data.S[i][j] = z
                

            signal = self.phase_shift_signal(self.data.S[i])
            self.data.signal[i] = signal
            
            #f2_center_frequency = self.configs['ana_f2'].frequencies[np.argmax(signal)]
            #f2_center_frequency = self.f1_to_f2(cw_frequency)
            #self.config.ana_f2.change_center_frequency(f2_center_frequency)
            #adapt two tone power?
            #self.config.ana_f2.power = self.f2_to_power(f2_center_frequency)


            meta = self.vna.get_meta()
            self.data.meta.append(meta)

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.data.currents/1e-3, self.data.signal)

            #save backup?


        #save data?
        
            

        self.end_experiment()


    



    def two_tone(self):

        self.start_experiment()

        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)

            cw_frequency = self.calibrate_f1()
            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency


            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            self.ana.set_freq_sweep(self.config.ana_f2)
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()

            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)

            for j in range(self.config.ana_f2.num_points):

                f, z = self.vna.sweep()

                self.data.f[i][j] = f
                self.data.S[i][j] = z

            signal = self.phase_shift_signal(self.data.S[i])
            self.data.signal[i] = signal
            
            #f2_center_frequency = self.configs['ana_f2'].frequencies[np.argmax(signal)]
            #f2_center_frequency = self.f1_to_f2(cw_frequency)
            #self.config.ana_f2.change_center_frequency(f2_center_frequency)
            #adapt two tone power?
            #self.config.ana_f2.power = self.f2_to_power(f2_center_frequency)


            meta = self.vna.get_meta()
            self.data.vna_meta.append(meta)

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.data.currents/1e-3, self.data.signal)

            #save backup?


        #save data?
        
            

        self.end_experiment()



    @Experiment.experiment_method
    def two_tone_optimized(self, vna_window: bool = True, masked_frequencies: Optional[list] = None):

              
        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = 1
            f_ref, z_ref = self.vna.sweep()
            self.data.f_ref[i] = f_ref
            self.data.S_ref[i] = z_ref

            # set scale
            if i == 0:
                scale = self.unwrapped_phase_shift_signal([z_ref[2:]], [z_ref[:-2]])
                print(f'scale: {scale}')


            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()
            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()


            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.ana_f2.num_points

            print(time.time() - start)
            print("configured")

            start = time.time()

            f, z = self.vna.sweeps()
            print(f'f shape: {np.shape(f)}')
            print(f'z shape: {np.shape(z)}')

            

            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

            for j in range(self.config.ana_f2.num_points):

                self.data.f[i][j] = f

                try:
                    self.data.S[i][j] = z[j]
                except:
                    self.data.S[i][j] = np.nan


            signal = self.full_S_parameter_signal(self.data.S[i], self.data.S_ref[i])
            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            vna_meta = self.vna.get_meta()
            self.data.vna_meta = vna_meta

            self.ana.output_off()

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.currents/1e-3, self.data.signal, scale)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    @Experiment.experiment_method
    def two_tone_optimized_average(self, average: int = 1, vna_window: bool = True, masked_frequencies: Optional[list] = None):

              
        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = 1
            f_ref, z_ref = self.vna.sweep()
            self.data.f_ref[i] = f_ref
            self.data.S_ref[i] = z_ref

            # set scale
            if i == 0:
                scale = self.unwrapped_phase_shift_signal([z_ref[2:]], [z_ref[:-2]])
                print(f'scale: {scale}')

            for b in range(average):


                self.vna.set_ext_trigger_out(self.config.vna_trigger)

                self.ana.set_ext_trig(self.config.ana_trigger)
                start_ana = time.time()
                self.ana.set_freq_sweep(self.config.ana_f2)
                print(time.time() - start_ana)
                print("Anapico freq sweep configured")
                self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                self.data.f2_powers[i] = self.config.ana_f2.power
                self.ana.output_on()


                self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

                self.vna.set_sweep(self.config.vna_f1)
                self.vna.sweep_count = self.config.ana_f2.num_points

                print(time.time() - start)
                print("configured")

                start = time.time()

                f, z = self.vna.sweeps()
                print(f'f shape: {np.shape(f)}')
                print(f'z shape: {np.shape(z)}')

                

                print(time.time() - start)

                if not vna_window:
                    self.vna.write('SYSTem:DISPlay:UPDate ONCE')

                self.vna.sweep_count = 1

                self.acquired_sweeps.append(len(z))

                for j in range(self.config.ana_f2.num_points):

                    self.data.f[i][j] = f

                    try:
                        self.data.S[i][j] = z[j]
                    except:
                        self.data.S[i][j] = np.nan


                signal = self.full_S_parameter_signal(self.data.S[i], self.data.S_ref[i])
                if b== 0:
                    self.data.signal[i] = signal
                else:
                    self.data.signal[i] += signal
                
                #self.f2_tracking(i)


                vna_meta = self.vna.get_meta()
                self.data.vna_meta = vna_meta

                self.ana.output_off()

                self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                                self.data.f2_center_frequencies/1e9,
                                                (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                                self.config.yoko.currents/1e-3, self.data.signal, scale)
                
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

    @Experiment.experiment_method
    def two_tone_optimized_average_current(self, average: int = 1, vna_window: bool = True, masked_frequencies: Optional[list] = None):

              
        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = 1
            f_ref, z_ref = self.vna.sweep()
            self.data.f_ref[i] = f_ref
            self.data.S_ref[i] = z_ref

            # set scale
            if i == 0:
                scale = self.unwrapped_phase_shift_signal([z_ref[2:]], [z_ref[:-2]])
                print(f'scale: {scale}')

            for b in range(average):


                self.vna.set_ext_trigger_out(self.config.vna_trigger)

                self.ana.set_ext_trig(self.config.ana_trigger)
                start_ana = time.time()
                self.ana.set_freq_sweep(self.config.ana_f2)
                print(time.time() - start_ana)
                print("Anapico freq sweep configured")
                self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                self.data.f2_powers[i] = self.config.ana_f2.power
                self.ana.output_on()


                self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

                self.vna.set_sweep(self.config.vna_f1)
                self.vna.sweep_count = self.config.ana_f2.num_points

                print(time.time() - start)
                print("configured")

                start = time.time()

                f, z = self.vna.sweeps()
                print(f'f shape: {np.shape(f)}')
                print(f'z shape: {np.shape(z)}')

                

                print(time.time() - start)

                if not vna_window:
                    self.vna.write('SYSTem:DISPlay:UPDate ONCE')

                self.vna.sweep_count = 1

                self.acquired_sweeps.append(len(z))

                for j in range(self.config.ana_f2.num_points):

                    self.data.f[i][j] = f

                    try:
                        self.data.S[i][j] = z[j]
                    except:
                        self.data.S[i][j] = np.nan


                signal = self.full_S_parameter_signal(self.data.S[i], self.data.S_ref[i])
                if b== 0:
                    self.data.signal[i] = signal
                else:
                    self.data.signal[i] += signal
                
                #self.f2_tracking(i)


                vna_meta = self.vna.get_meta()
                self.data.vna_meta = vna_meta

                self.ana.output_off()

                self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                                self.data.f2_center_frequencies/1e9,
                                                (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                                self.config.yoko.currents/1e-3, self.data.signal, scale)
                
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    @Experiment.experiment_method
    def two_tone_optimized_average_voltage(self, average: int = 1, vna_window: bool = True, masked_frequencies: Optional[list] = None):

              
        # check yoko voltage, ramp slowly to starting value
        self.yoko.set_source_voltage_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial voltage value
        self.yoko.ramp_voltage(self.config.yoko.voltages[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, voltage in enumerate(self.config.yoko.voltages):
            #set current
            self.yoko.voltage(voltage)
            self.data.voltages[i] = voltage

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = 1
            f_ref, z_ref = self.vna.sweep()
            self.data.f_ref[i] = f_ref
            self.data.S_ref[i] = z_ref

            # set scale
            if i == 0:
                scale = self.unwrapped_phase_shift_signal([z_ref[2:]], [z_ref[:-2]])
                print(f'scale: {scale}')

            for b in range(average):


                self.vna.set_ext_trigger_out(self.config.vna_trigger)

                self.ana.set_ext_trig(self.config.ana_trigger)
                start_ana = time.time()
                self.ana.set_freq_sweep(self.config.ana_f2)
                print(time.time() - start_ana)
                print("Anapico freq sweep configured")
                self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                self.data.f2_powers[i] = self.config.ana_f2.power
                self.ana.output_on()


                self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

                self.vna.set_sweep(self.config.vna_f1)
                self.vna.sweep_count = self.config.ana_f2.num_points

                print(time.time() - start)
                print("configured")

                start = time.time()

                f, z = self.vna.sweeps()
                print(f'f shape: {np.shape(f)}')
                print(f'z shape: {np.shape(z)}')

                

                print(time.time() - start)

                if not vna_window:
                    self.vna.write('SYSTem:DISPlay:UPDate ONCE')

                self.vna.sweep_count = 1

                self.acquired_sweeps.append(len(z))

                for j in range(self.config.ana_f2.num_points):

                    self.data.f[i][j] = f

                    try:
                        self.data.S[i][j] = z[j]
                    except:
                        self.data.S[i][j] = np.nan


                signal = self.full_S_parameter_signal(self.data.S[i], self.data.S_ref[i])
                if b== 0:
                    self.data.signal[i] = signal
                else:
                    self.data.signal[i] += signal
                
                #self.f2_tracking(i)


                vna_meta = self.vna.get_meta()
                self.data.vna_meta = vna_meta

                self.ana.output_off()

                self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                                self.data.f2_center_frequencies/1e9,
                                                (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                                self.config.yoko.voltages/1e-3, self.data.signal, scale)
                
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

    @Experiment.experiment_method
    def two_tone_f2_power_sweep(self, vna_window: bool = True, average: int =1, masked_frequencies: Optional[list] = None):
              
        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current
            sleep(self.config.yoko.wait)

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = 1
            f_ref, z_ref = self.vna.sweep()
            self.data.f_ref[i] = f_ref
            self.data.S_ref[i] = z_ref

            # set scale
            if i == 0:
                scale = self.unwrapped_phase_shift_signal([z_ref[2:]], [z_ref[:-2]])
                print(f'scale: {scale}')

            for b in range(average):
                for a, f2_power in enumerate(self.config.ana_f2.powers):

                    self.vna.set_ext_trigger_out(self.config.vna_trigger)

                    self.ana.set_ext_trig(self.config.ana_trigger)
                    start_ana = time.time()
                    self.config.ana_f2.power = f2_power
                    self.ana.set_freq_sweep(self.config.ana_f2)
                    print(time.time() - start_ana)
                    print("Anapico freq sweep configured")
                    self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                    self.data.f2_powers[a] = self.config.ana_f2.power
                    self.ana.output_on()


                    self.data.f2_frequencies[a] = self.config.ana_f2.frequencies

                    self.vna.set_sweep(self.config.vna_f1)
                    self.vna.sweep_count = self.config.ana_f2.num_points

                    print(time.time() - start)
                    print("configured")

                    start = time.time()

                    f, z = self.vna.sweeps()
                    print(f'f shape: {np.shape(f)}')
                    print(f'z shape: {np.shape(z)}')

                    

                    print(time.time() - start)

                    if not vna_window:
                        self.vna.write('SYSTem:DISPlay:UPDate ONCE')

                    self.vna.sweep_count = 1

                    self.acquired_sweeps.append(len(z))

                    for j in range(self.config.ana_f2.num_points):

                        self.data.f[a][j] = f

                        try:
                            self.data.S[a][j] = z[j]
                        except:
                            self.data.S[a][j] = np.nan


                    signal = self.full_S_parameter_signal(self.data.S[a], self.data.S_ref[i])
                    
                    if b==0:
                        self.data.signal[a] = signal
                    else:
                        self.data.signal[a] += signal

                    #self.f2_tracking(i)

                    vna_meta = self.vna.get_meta()
                    self.data.vna_meta = vna_meta

                    self.ana.output_off()

                    self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                                    self.data.f2_center_frequencies/1e9,
                                                    (self.data.f2_frequencies[a][-1] - self.data.f2_frequencies[a][0])/1e9,
                                                    self.config.ana_f2.powers, self.data.signal, scale)
                    
                

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    @Experiment.experiment_method
    def two_tone_cw_f2_power_sweep(self, vna_window: bool = True, average: int =1, masked_frequencies: Optional[list] = None):
              
        # check yoko current, ramp slowly to starting value
        if self.config.yoko.current_range >= self.yoko.current():
            self.yoko.set_source_current_sweep(self.config.yoko)
        else:
            self.yoko.source_current()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)
        self.yoko.set_source_current_sweep(self.config.yoko)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current
            sleep(self.config.yoko.wait)

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z_masked(masked_frequencies)
            else:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

        
            
            for a, f2_power in enumerate(self.config.ana_f2.powers):

                self.vna.set_ext_trigger_out(self.config.vna_trigger)

                self.ana.set_ext_trig(self.config.ana_trigger)
                start_ana = time.time()
                self.config.ana_f2.power = f2_power
                self.ana.set_freq_sweep(self.config.ana_f2)
                print(time.time() - start_ana)
                print("Anapico freq sweep configured")
                self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                self.data.f2_powers[a] = self.config.ana_f2.power
                self.ana.output_on()


                self.data.f2_frequencies[a] = self.config.ana_f2.frequencies

                self.vna.set_sweep(self.config.vna_f1)
                self.vna.sweep_count = self.config.vna_f1.num_averages 

                print(time.time() - start)
                print("configured")

                start = time.time()

                f, z = self.vna.sweep()
                print(f'f shape: {np.shape(f)}')
                print(f'z shape: {np.shape(z)}')

                

                print(time.time() - start)

                if not vna_window:
                    self.vna.write('SYSTem:DISPlay:UPDate ONCE')


                self.acquired_sweeps.append(len(z))

                try:
                    self.data.S[i] = z
                except:
                    self.data.S[i] = np.nan



                if masked_frequencies is not None:
                    mask = self.mask_devices(f_calib, masked_frequencies)
                    signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))] )
                else:
                    signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))])
                
                
                #signal = signal - np.mean(signal)
                self.data.signal[a] = signal
                

                #self.f2_tracking(i)

                vna_meta = self.vna.get_meta()
                self.data.vna_meta = vna_meta

                self.ana.output_off()

                self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                                self.data.f2_center_frequencies/1e9,
                                                (self.data.f2_frequencies[a][-1] - self.data.f2_frequencies[a][0])/1e9,
                                                self.config.ana_f2.powers, self.data.signal)
                
                

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    @Experiment.experiment_method
    def two_tone_cw_f2_power_sweep_voltage(self, vna_window: bool = True, average: int =1, masked_frequencies: Optional[list] = None):
              
         # check yoko current, ramp slowly to starting value
        if self.config.yoko.voltage_range >= self.yoko.voltage():
            self.yoko.set_source_voltage_sweep(self.config.yoko)
        else:
            self.yoko.source_voltage()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_voltage(self.config.yoko.voltages[0], blocking=True)
        self.yoko.set_source_voltage_sweep(self.config.yoko)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, voltage in enumerate(self.config.yoko.voltages):
            #set current
            self.yoko.voltage(voltage)
            self.data.voltages[i] = voltage
            sleep(self.config.yoko.wait)

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z_masked(masked_frequencies)
            else:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z()

            
            self.config.vna_f1.center_frequency = cw_frequency
            self.data.f1_center_frequencies[i] = self.config.vna_f1.center_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

        
            
            for a, f2_power in enumerate(self.config.ana_f2.powers):

                self.vna.set_ext_trigger_out(self.config.vna_trigger)

                self.ana.set_ext_trig(self.config.ana_trigger)
                start_ana = time.time()
                self.config.ana_f2.power = f2_power
                self.ana.set_freq_sweep(self.config.ana_f2)
                print(time.time() - start_ana)
                print("Anapico freq sweep configured")
                self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
                self.data.f2_powers[a] = self.config.ana_f2.power
                self.ana.output_on()


                self.data.f2_frequencies[a] = self.config.ana_f2.frequencies

                self.vna.set_sweep(self.config.vna_f1)
                self.vna.sweep_count = self.config.vna_f1.num_averages 

                print(time.time() - start)
                print("configured")

                start = time.time()

                f, z = self.vna.sweep()
                print(f'f shape: {np.shape(f)}')
                print(f'z shape: {np.shape(z)}')

                

                print(time.time() - start)

                if not vna_window:
                    self.vna.write('SYSTem:DISPlay:UPDate ONCE')


                self.acquired_sweeps.append(len(z))

                try:
                    self.data.S[i] = z
                except:
                    self.data.S[i] = np.nan



                if masked_frequencies is not None:
                    mask = self.mask_devices(f_calib, masked_frequencies)
                    signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))])
                    #signal = np.angle(z) 
                else:
                    signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))])
                    #signal = np.angle(z)
                
                #signal = signal - np.mean(signal)
                self.data.signal[a] = signal
                

                #self.f2_tracking(i)

                vna_meta = self.vna.get_meta()
                self.data.vna_meta = vna_meta

                self.ana.output_off()

                self.plotter.update_twotone_imshow_voltage(self.data.f1_center_frequencies/1e9,
                                                self.data.f2_center_frequencies/1e9,
                                                (self.data.f2_frequencies[a][-1] - self.data.f2_frequencies[a][0])/1e9,
                                                self.config.ana_f2.powers, self.data.signal)
                
                

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

    @Experiment.experiment_method
    def two_tone_cw(self, vna_window: bool = True, masked_frequencies: Optional[list] = None):

              
        # check yoko current, ramp slowly to starting value
        if self.config.yoko.current_range >= self.yoko.current():
            self.yoko.set_source_current_sweep(self.config.yoko)
        else:
            self.yoko.source_current()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)
        self.yoko.set_source_current_sweep(self.config.yoko)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')
        else:
            self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency = self.calibrate_f1_masked(masked_frequencies)
            else:
                cw_frequency = self.calibrate_f1()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            # self.vna.set_sweep(self.config.vna_f1)
            # self.vna.sweep_count = 1
            # f_ref, z_ref = self.vna.sweep()
            # self.data.f_ref[i] = f_ref
            # self.data.S_ref[i] = z_ref

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()
            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()


            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.vna_f1.num_averages  

            print(time.time() - start)
            print("configured")

            start = time.time()

            
            f, z = self.vna.sweep()
            print(f'f shape: {np.shape(f)}')
            print(f'z shape: {np.shape(z)}')


            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            #self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

    
            try:
                self.data.S[i] = z
            except:
                self.data.S[i] = np.nan


            signal = S_to_dBm(z)
            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            vna_meta = self.vna.get_meta()
            self.data.vna_meta = vna_meta

            self.ana.output_off()

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.currents/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')



    def add_two_tone_trace(self, i: int, vna_window: bool = True):

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')
        else:
            self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.vna.visa_instr.timeout = 500e3

        self.data.f1_center_frequencies = self.config.vna_f1.center_frequency

        self.vna.set_ext_trigger_out(self.config.vna_trigger)

        self.ana.set_ext_trig(self.config.ana_trigger)
     
        self.ana.set_freq_sweep(self.config.ana_f2)
     
        self.data.f2_center_frequencies = self.config.ana_f2.center_frequency
        self.data.f2_powers = self.config.ana_f2.power
        self.ana.output_on()


        self.data.f2_frequencies = self.config.ana_f2.frequencies

        self.vna.set_sweep(self.config.vna_f1)
        self.vna.sweep_count = self.config.vna_f1.num_averages  

       

        start = time.time()

        
        f, z = self.vna.sweep()
  

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate ONCE')

        try:
            self.data.S[i] = z
        except:
            self.data.S[i] = np.nan


        vna_meta = self.vna.get_meta()
        self.data.vna_meta = vna_meta

        self.ana.output_off()






    @Experiment.experiment_method
    def two_tone_cw_voltage(self, vna_window: bool = True, masked_frequencies: Optional[list] = None):

        # check yoko current, ramp slowly to starting value
        if self.config.yoko.voltage_range >= self.yoko.voltage():
            self.yoko.set_source_voltage_sweep(self.config.yoko)
        else:
            self.yoko.source_voltage()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_voltage(self.config.yoko.voltages[0], blocking=True)
        self.yoko.set_source_voltage_sweep(self.config.yoko)
        

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')
        else:
            self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.vna.visa_instr.timeout = 500e3


        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, voltage in enumerate(self.config.yoko.voltages):
            #set current
            self.yoko.voltage(voltage)
            self.data.voltages[i] = voltage

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z_masked(masked_frequencies)
            else:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z()

            #self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            self.data.f1_center_frequencies[i] = self.config.vna_f1.center_frequency
            

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            # self.vna.set_sweep(self.config.vna_f1)
            # self.vna.sweep_count = 1
            # f_ref, z_ref = self.vna.sweep()
            # self.data.f_ref[i] = f_ref
            # self.data.S_ref[i] = z_ref

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()
            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()


            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.vna_f1.num_averages  

            print(time.time() - start)
            print("configured")

            start = time.time()

            
            f, z = self.vna.sweep()
            print(f'f shape: {np.shape(f)}')
            print(f'z shape: {np.shape(z)}')


            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            #self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

    
            try:
                self.data.S[i] = z
            except:
                self.data.S[i] = np.nan


            if masked_frequencies is not None:
                mask = self.mask_devices(f_calib, masked_frequencies)
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))] )
            else:
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))])

            #signal = S_to_dBm(z)
            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            vna_meta = self.vna.get_meta()
            self.data.vna_meta = vna_meta

            self.ana.output_off()

            self.plotter.update_twotone_imshow_voltage(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.voltages/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    def find_parabola_vertex_form(self, p1, p2, p3):
        # Extract x and y values from the points
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Set up the system of equations for the points
        # y1 = a(x1 - x0)^2 + c
        # y2 = a(x2 - x0)^2 + c
        # y3 = a(x3 - x0)^2 + c

        # We will solve for x0 first by eliminating a and c.
        # Define the function to solve for x0

        def func(x0):
            A = np.array([[(x1 - x0)**2, 1],
                        [(x2 - x0)**2, 1],
                        [(x3 - x0)**2, 1]])
            Y = np.array([y1, y2, y3])
            return np.linalg.lstsq(A, Y, rcond=None)[1]  # Returns the residuals for x0

        # Search for the best x0 that minimizes the residual
        x0_values = np.linspace(min(x1, x2, x3), max(x1, x2, x3), 1000)
        residuals = [func(x0) for x0 in x0_values]
        x0 = x0_values[np.argmin(residuals)]

        # Once we find x0, we can now solve for a and c
        A = np.array([[(x1 - x0)**2, 1],
                    [(x2 - x0)**2, 1],
                    [(x3 - x0)**2, 1]])
        Y = np.array([y1, y2, y3])

        a, c = np.linalg.lstsq(A, Y, rcond=None)[0]

        return a, x0, c

    @Experiment.experiment_method
    def two_tone_cw_tracking_parabola(self, vna_window: bool = True, tracking_parameters: list = None, masked_frequencies: Optional[list] = None):

        a, x0, c = self.find_parabola_vertex_form(*tracking_parameters)
        
        # check yoko current, ramp slowly to starting value
        if self.config.yoko.current_range >= self.yoko.current():
            self.yoko.set_source_current_sweep(self.config.yoko)
        else:
            self.yoko.source_current()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)
        self.yoko.set_source_current_sweep(self.config.yoko)


        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')
        else:
            self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)
            self.data.currents[i] = current

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z_masked(masked_frequencies)
            else:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            # self.vna.set_sweep(self.config.vna_f1)
            # self.vna.sweep_count = 1
            # f_ref, z_ref = self.vna.sweep()
            # self.data.f_ref[i] = f_ref
            # self.data.S_ref[i] = z_ref

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()

            f2_center = a* (current - x0)**2 + c
            self.config.ana_f2.change_center_frequency(f2_center)

            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()


            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.vna_f1.num_averages  

            print(time.time() - start)
            print("configured")

            start = time.time()

            
            f, z = self.vna.sweep()
            print(f'f shape: {np.shape(f)}')
            print(f'z shape: {np.shape(z)}')


            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            #self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

            for j in range(self.config.ana_f2.num_points):

                try:
                    self.data.S[i] = z
                except:
                    self.data.S[i] = np.nan

            if masked_frequencies is not None:
                mask = self.mask_devices(f_calib, masked_frequencies)
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))] )
            else:
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))])
            

            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            vna_meta = self.vna.get_meta()
            self.data.vna_meta = vna_meta

            self.ana.output_off()

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.currents/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

    @Experiment.experiment_method
    def two_tone_cw_voltage_tracking_parabola(self, vna_window: bool = True, tracking_parameters: list = None, masked_frequencies: Optional[list] = None):

        a, x0, c = self.find_parabola_vertex_form(*tracking_parameters)
        
         # check yoko current, ramp slowly to starting value
        if self.config.yoko.voltage_range >= self.yoko.voltage():
            self.yoko.set_source_voltage_sweep(self.config.yoko)
        else:
            self.yoko.source_voltage()

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_voltage(self.config.yoko.voltages[0], blocking=True)
        self.yoko.set_source_voltage_sweep(self.config.yoko)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')
        else:
            self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over voltages corresponding to different fluxes generated by the coil
        for i, voltage in enumerate(self.config.yoko.voltages):
            #set voltage
            self.yoko.voltage(voltage)
            self.data.voltages[i] = voltage

            start = time.time()

            if masked_frequencies is not None:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z_masked(masked_frequencies)
            else:
                cw_frequency, f_calib, z_calib = self.calibrate_f1_z()

            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            # take reference trace
            # self.vna.set_sweep(self.config.vna_f1)
            # self.vna.sweep_count = 1
            # f_ref, z_ref = self.vna.sweep()
            # self.data.f_ref[i] = f_ref
            # self.data.S_ref[i] = z_ref

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()

            f2_center = a* (voltage - x0)**2 + c
            self.config.ana_f2.change_center_frequency(f2_center)

            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()


            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.vna_f1.num_averages  

            print(time.time() - start)
            print("configured")

            start = time.time()

            
            f, z = self.vna.sweep()
            print(f'f shape: {np.shape(f)}')
            print(f'z shape: {np.shape(z)}')


            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            #self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

            for j in range(self.config.ana_f2.num_points):

                try:
                    self.data.S[i] = z
                except:
                    self.data.S[i] = np.nan

            if masked_frequencies is not None:
                mask = self.mask_devices(f_calib, masked_frequencies)
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib[~mask])))] )
            else:
                signal = np.abs(z - z_calib[self.find_highest_average_triplet(-20*np.log(np.abs(z_calib)))])
            

            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            vna_meta = self.vna.get_meta()
            self.data.vna_meta = vna_meta

            self.ana.output_off()

            self.plotter.update_twotone_imshow_voltage(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.config.yoko.voltages/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')


    def two_tone_tracking(self, vna_window=True, adaptive_power=True, adaptive_span=True):

        self.start_experiment()

        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)

            start = time.time()

            cw_frequency = self.calibrate_f1()
            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            flux, f2 = self.f1_to_f2_tracking(cw_frequency)
            if flux is not None:
                self.data.fluxes[i] = flux
                self.config.ana_f2.change_center_frequency(f2)

            if adaptive_power:
                f2_power = self.config.f2_to_power(self.config.ana_f2.center_frequency)
                self.config.ana_f2.power = f2_power

            if adaptive_span:
                f1_span = self.config.f2_to_span(self.config.ana_f2.center_frequency)
                self.config.vna_f1.span = f1_span
            


            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()
            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()

            

            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.ana_f2.num_points

            print(time.time() - start)
            print("configured")

            start = time.time()

            f, z = self.vna.sweeps()

            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

            for j in range(self.config.ana_f2.num_points):

                self.data.f[i][j] = f

                try:
                    self.data.S[i][j] = z[j]
                except:
                    self.data.S[i][j] = np.nan


            signal = self.unwrapped_phase_shift_signal(self.data.S[i])
            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            meta = self.vna.get_meta()
            self.data.meta.append(meta)

            self.ana.output_off()

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.data.currents/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.end_experiment()

    
    
    def two_tone_and_flux_map(self, vna_window=True):

        self.start_experiment()

        self.data.f_flux_map_rough = np.full((len(self.config.yoko.currents), self.config.vna_f1_calib.num_points), np.nan)
        self.data.S_flux_map_rough = np.full((len(self.config.yoko.currents), self.config.vna_f1_calib.num_points), np.nan, dtype=complex)
        self.data.f_flux_map_fine = np.full((len(self.config.yoko.currents), self.config.vna_f1_calib.num_points//2), np.nan)
        self.data.S_flux_map_fine = np.full((len(self.config.yoko.currents), self.config.vna_f1_calib.num_points//2), np.nan, dtype=complex)

        # check yoko current, ramp slowly to starting value
        self.yoko.set_source_current_sweep(self.config.yoko)

        #turn current source on
        self.yoko.output(True)

        # ramp to initial current value
        self.yoko.ramp_current(self.config.yoko.currents[0], blocking=True)

        self.acquired_sweeps = []

        if not vna_window:
            self.vna.write('SYSTem:DISPlay:UPDate OFF')

        self.vna.visa_instr.timeout = 500e3

        

        #iterate over currents corresponding to different fluxes generated by the coil
        for i, current in enumerate(self.config.yoko.currents):
            #set current
            self.yoko.current(current)

            start = time.time()

            cw_frequency, self.data.f_flux_map_rough[i], self.data.S_flux_map_rough[i], self.data.f_flux_map_fine[i], self.data.S_flux_map_fine[i] = self.calibrate_f1_and_flux_map()
            self.data.f1_center_frequencies[i] = cw_frequency
            self.config.vna_f1.center_frequency = cw_frequency

            print(time.time() - start)
            print("calibrated")

            #self.data.f1_center_frequencies[i] = self.config.vna_f1_calib.center_frequency

            start = time.time()

            self.vna.set_ext_trigger_out(self.config.vna_trigger)

            

            self.ana.set_ext_trig(self.config.ana_trigger)
            start_ana = time.time()
            self.ana.set_freq_sweep(self.config.ana_f2)
            print(time.time() - start_ana)
            print("Anapico freq sweep configured")
            self.data.f2_center_frequencies[i] = self.config.ana_f2.center_frequency
            self.data.f2_powers[i] = self.config.ana_f2.power
            self.ana.output_on()

            

            self.data.f2_frequencies[i] = self.config.ana_f2.frequencies

            self.vna.set_sweep(self.config.vna_f1)
            self.vna.sweep_count = self.config.ana_f2.num_points

            print(time.time() - start)
            print("configured")

            start = time.time()

            f, z = self.vna.sweeps()

            print(time.time() - start)

            if not vna_window:
                self.vna.write('SYSTem:DISPlay:UPDate ONCE')

            self.vna.sweep_count = 1

            self.acquired_sweeps.append(len(z))

            for j in range(self.config.ana_f2.num_points):

                self.data.f[i][j] = f

                try:
                    self.data.S[i][j] = z[j]
                except:
                    self.data.S[i][j] = np.nan


            signal = self.unwrapped_phase_shift_signal(self.data.S[i])
            self.data.signal[i] = signal
            
            #self.f2_tracking(i)


            meta = self.vna.get_meta()
            self.data.meta.append(meta)

            self.ana.output_off()

            self.plotter.update_twotone_imshow(self.data.f1_center_frequencies/1e9,
                                               self.data.f2_center_frequencies/1e9,
                                               (self.data.f2_frequencies[i][-1] - self.data.f2_frequencies[i][0])/1e9,
                                               self.data.currents/1e-3, self.data.signal)
            
            

            #save backup?


        #save data?
        
        self.vna.write('SYSTem:DISPlay:UPDate ON')

        self.end_experiment()



            





    





