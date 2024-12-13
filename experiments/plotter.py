import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
from matplotlib.collections import QuadMesh
from time import sleep

from IPython.display import display, clear_output

from typing import Tuple, Optional, Union

from experiments.experiment_data import FluxMapData, TwoToneData
from experiments.utils import S_to_dBm, is_linearly_spaced, round_to_significant_digits

class Plotter:

    pcolor_x: np.ndarray = None
    pcolor_y: np.ndarray = None
    pcolor_z: np.ndarray = None
    pcolor_mesh: QuadMesh = None
    pcolor_cmap: str = 'viridis'

    pcolor_fig: Figure = None
    pcolor_ax: Axes = None
    pcolor_bar: Colorbar = None

    imshow_x: np.ndarray = None
    imshow_y: np.ndarray = None
    imshow_z: np.ndarray = None

    imshow_fig: Figure = None
    imshow_ax: Axes = None
    imshow_im: AxesImage = None

    twotone_fig: Figure = None
    twotone_axes: Tuple[Axes] = None
    twotone_im: AxesImage = None

    def __init__(self) -> None:
        pass


    
    
    def update_twotone_imshow_voltage(self, f1_center_frequencies, f2_center_frequencies, f2_span, voltages, signal, scale=None):
        if self.twotone_fig is None:

            self.twotone_fig, self.twotone_axes = plt.subplots(1,3)
            self.twotone_fig.set_figwidth(20)

            self.twotone_axes[0].plot(voltages, f2_center_frequencies)
            self.twotone_axes[2].plot(voltages, f1_center_frequencies)
           

            self.twotone_im = self.twotone_axes[1].imshow(np.flip(signal.T, axis=0), cmap='viridis', aspect='auto',
                                                          extent=[voltages[0], voltages[-1], f2_center_frequencies[0] - f2_span/2, f2_center_frequencies[0] + f2_span/2],
                                                          interpolation='none')
            self.twotone_axes[1].set_xlabel('voltage (mV)')
            self.twotone_axes[1].set_ylabel('f2 (GHz)')

            #self.twotone_im.set_clim(vmin=scale, vmax=-scale)

            clear_output(wait=True)
            #plt.tight_layout()
            display(self.twotone_fig)

        else:

            self.twotone_axes[0].clear()
            self.twotone_axes[2].clear()
            self.twotone_axes[0].plot(voltages, f2_center_frequencies)
            self.twotone_axes[0].set_xlabel('voltage (mV)')
            self.twotone_axes[0].set_ylabel('f2 center (GHz)')
            self.twotone_axes[2].plot(voltages, f1_center_frequencies)
            self.twotone_axes[2].set_xlabel('voltage (mV)')
            self.twotone_axes[2].set_ylabel('f1 center (GHz)')

            self.twotone_im.set_data(np.flip(signal.T, axis=0))
            self.twotone_im.set_clim(vmin=np.nanmin(signal), vmax=np.nanmax(signal))

            #self.twotone_axes[1].relim()
            #self.twotone_axes[1].autoscale_view()

            clear_output(wait=True)
            #plt.tight_layout()
            display(self.twotone_fig)




    def update_imshow(self, new_x: np.ndarray, new_y: np.ndarray, new_z: np.ndarray):

        if self.imshow_fig is None:
            self.imshow_x = new_x
            self.imshow_y = new_y
            self.imshow_z = new_z

            self.imshow_fig, self.imshow_ax = plt.subplots()
            self.imshow_im = self.imshow_ax.imshow([self.imshow_z], cmap='viridis', aspect='auto') #extent=[self.imshow_x.min(), self.imshow_x.max(), self.imshow_y.min(), self.imshow_y.max()]
            #plt.show(block=False)
            clear_output(wait=True)
            #plt.tight_layout()
            display(self.imshow_fig)

            
        else:
            # Append the new row to the existing data
            self.imshow_x = np.vstack([self.imshow_x, new_x])
            self.imshow_y = np.vstack([self.imshow_y, new_y])
            self.imshow_z = np.vstack([self.imshow_z, new_z])

            # Update the image data
            self.imshow_im.set_data(np.flip(self.imshow_z.T, axis=0))
            #self.imshow_fig.show()

            # Adjust the limits and redraw the plot
            self.imshow_ax.relim()
            self.imshow_ax.autoscale_view()
            clear_output(wait=True)
            #plt.tight_layout()
            display(self.imshow_fig)

    def update_full_imshow(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):

        if self.imshow_fig is None:
            self.imshow_x = x
            self.imshow_y = y
            self.imshow_z = np.full((len(self.imshow_x), len(self.imshow_y)), np.nan)

            for i,row in enumerate(z):
                self.imshow_z[i] = row

            self.imshow_fig, self.imshow_ax = plt.subplots()
            self.imshow_im = self.imshow_ax.imshow(np.flip(self.imshow_z.T, axis=0), 
                                                   extent=[self.imshow_x.flat[0], self.imshow_x.flat[-1], self.imshow_y.min(), self.imshow_y.max()],
                                                   cmap='viridis', aspect='auto', interpolation='none')
        else:

            for i,row in enumerate(z):
                self.imshow_z[i] = row

            self.imshow_im.set_data(np.flip(self.imshow_z.T, axis=0))
            #self.imshow_im.set_clim(vmin=self.imshow_z.min(), vmax=self.imshow_z.max())
            self.imshow_ax.relim()
            self.imshow_ax.autoscale_view()
            clear_output(wait=True)
            #plt.tight_layout()
            display(self.imshow_fig)


    def plot_flux_map(self, data: FluxMapData, title: str = 'Flux Map' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        ### Would need to be improved for adaptive fluxmaps!

        if data.currents is not None:
            self.plot_flux_map_current(data, title, cmap, comment)
        elif data.voltages is not None:
            self.plot_flux_map_voltage(data, title, cmap, comment)

    
    def plot_flux_map_current(self, data: FluxMapData, title: str = 'Flux Map' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        # Create the figure and axes object
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        currents = data.currents/1e-3 #in mA
        frequencies = data.f/1e9 #in GHz

        # Add labels
        ax.set_xlabel('Current (mA)')
        ax.set_ylabel('Frequency (GHz)')
        

        # Plot the data
        if currents[0] < currents[-1]:
            cax = ax.imshow(np.flip(S_to_dBm(data.S).T, axis=0),
                        extent=[currents.min(), currents.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')
        else:
            cax = ax.imshow(np.flip(np.flip(S_to_dBm(data.S).T, axis=0), axis=1),
                        extent=[currents.min(), currents.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')

        # Add title
        ax.set_title(title)

        # Add colorbar with label
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(r'$|\text{S}_{21}|$ (dB)')

        # Generate comments and add as a text box
        comments = self.generate_flux_map_comments_current(data)

        cbar_box = cbar.ax.get_position()
        ax_box = ax.get_position()
        position = (cbar_box.x1 + 0.04, ax_box.y1)
        
        if comment:
            Plotter.add_text_box(ax, comments, position)

        return fig, ax

    def plot_flux_map_fluxes(self, data: FluxMapData, title: str = 'Flux Map' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        # Create the figure and axes object
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        fluxes = data.fluxes #in mA
        frequencies = data.f/1e9 #in GHz

        # Add labels
        ax.set_xlabel(r'Flux (2$\pi$)')
        ax.set_ylabel('Frequency (GHz)')
        

        # Plot the data
        if fluxes[0] < fluxes[-1]:
            cax = ax.imshow(np.flip(S_to_dBm(data.S).T, axis=0),
                        extent=[fluxes.min(), fluxes.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')
        #flip x axis 
        else:
            cax = ax.imshow(np.flip(np.flip(S_to_dBm(data.S).T, axis=0), axis=1),
                        extent=[fluxes.min(), fluxes.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')

        # Add title
        ax.set_title(title)

        # Add colorbar with label
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(r'$|\text{S}_{21}|$ (dB)')

        # Generate comments and add as a text box
        # comments = self.generate_flux_map_comments_flux(data)

        # cbar_box = cbar.ax.get_position()
        # ax_box = ax.get_position()
        # position = (cbar_box.x1 + 0.04, ax_box.y1)
        
        # if comment:
        #     Plotter.add_text_box(ax, comments, position)

        return fig, ax

    def plot_flux_map_voltage(self, data: FluxMapData, title: str = 'Flux Map' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        # Create the figure and axes object
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        voltages = data.voltages/1e-3 #in mA
        frequencies = data.f/1e9 #in GHz

        # Add labels
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Frequency (GHz)')
        

        # Plot the data
        if voltages[0] < voltages[-1]:
            cax = ax.imshow(np.flip(S_to_dBm(data.S).T, axis=0),
                        extent=[voltages.min(), voltages.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')
        #flip x axis 
        else:
            cax = ax.imshow(np.flip(np.flip(S_to_dBm(data.S).T, axis=0), axis=1),
                        extent=[voltages.min(), voltages.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')

        # Add title
        ax.set_title(title)

        # Add colorbar with label
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(r'$|\text{S}_{21}|$ (dB)')

        # Generate comments and add as a text box
        comments = self.generate_flux_map_comments_voltage(data)

        cbar_box = cbar.ax.get_position()
        ax_box = ax.get_position()
        position = (cbar_box.x1 + 0.04, ax_box.y1)
        
        if comment:
            Plotter.add_text_box(ax, comments, position)

        return fig, ax

    
    def plot_two_tone_signal_trace(self, data: TwoToneData, index=-1, title: str = 'Two Tone signal trace' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        
        # Create the figure and axes object
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        ax.plot(data.f2_frequencies[index]/1e9, data.signal[index])

        # Add title
        ax.set_title(title)
        ax.set_xlabel('$f_2$ frequency (GHz)')
        ax.set_ylabel('Signal')

        return fig, ax




    def plot_power_sweep(self, data: FluxMapData, title: str = 'Power Sweep' , cmap: str = 'viridis', comment: bool=True) -> Tuple[Figure, Axes]:
        ### Would need to be improved for adaptive fluxmaps!

        # Create the figure and axes object
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()

        powers = data.powers #in dBm
        frequencies = data.f/1e9 #in GHz

        # Add labels
        ax.set_xlabel('Power (dBm)')
        ax.set_ylabel('Frequency (GHz)')
        

        # Plot the data
        cax = ax.imshow(np.flip(S_to_dBm(data.S).T, axis=0),
                        extent=[powers.min(), powers.max(), frequencies.min(), frequencies.max()], 
                        cmap=cmap, aspect='auto', interpolation='none')


        # Add title
        ax.set_title(title)

        # Add colorbar with label
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(r'$|\text{S}_{21}|$ (dB)')

        # Generate comments and add as a text box
        comments = self.generate_power_sweep_comments(data)

        cbar_box = cbar.ax.get_position()
        ax_box = ax.get_position()
        position = (cbar_box.x1 + 0.04, ax_box.y1)
        
        if comment:
            Plotter.add_text_box(ax, comments, position)

        return fig, ax

    @staticmethod
    def generate_flux_map_comments_current(data: FluxMapData) -> str:
           
            currents = data.currents/1e-3 #in mA
            frequencies = data.f/1e9 #in GHz

            # Ascending currents
            if data.currents[0] < data.currents[-1]:
                current_sweep_dir = 'right'
            # Descending currents
            else:
                current_sweep_dir = 'left'

            if is_linearly_spaced(frequencies):
                frequency_step = f'{np.round((frequencies[1] - frequencies[0])*1e6)} kHz'
            else:
                frequency_step = 'non-linear'

            comment = (
                f"VNA power: {data.vna_meta['power']} dBm\n"
                f"VNA vbw: {data.vna_meta['VBW']} Hz\n"
                f"VNA average: {data.vna_meta['average']}\n"
                f"Current sweep direction: {current_sweep_dir}\n"
                f"Current range: {currents[0]} to {currents[-1]} mA\n"
                f"Current step: {np.round(currents[1] - currents[0],4)} mA\n"
                f"Frequency range: {frequencies.min()} to {frequencies.max()} GHz\n"
                f"Frequency step: {frequency_step}\n"
                f"Max |S|: {S_to_dBm(data.S).max():.2f} dB\n"
                f"Min |S|: {S_to_dBm(data.S).min():.2f} dB"   
            )
            return comment
    
    @staticmethod
    def generate_flux_map_comments_voltage(data: FluxMapData) -> str:
           
            voltages = data.voltages/1e-3 #in mA
            voltages_initial = round_to_significant_digits(voltages[0], 5)
            voltages_final = round_to_significant_digits(voltages[-1], 5)
            frequencies = data.f/1e9 #in GHz

            # Ascending currents
            if data.voltages[0] < data.voltages[-1]:
                current_sweep_dir = 'right'
            # Descending currents
            else:
                current_sweep_dir = 'left'

            if is_linearly_spaced(frequencies):
                frequency_step = f'{np.round((frequencies[1] - frequencies[0])*1e6)} kHz'
            else:
                frequency_step = 'non-linear'

            comment = (
                f"VNA power: {data.vna_meta['power']} dBm\n"
                f"VNA vbw: {data.vna_meta['VBW']} Hz\n"
                f"VNA average: {data.vna_meta['average']}\n"
                f"Voltage sweep direction: {current_sweep_dir}\n"
                f"Voltage range: {voltages_initial} to {voltages_final} mV\n"
                f"Voltage step: {np.round(voltages[1] - voltages[0],4)} mV\n"
                f"Frequency range: {frequencies.min()} to {frequencies.max()} GHz\n"
                f"Frequency step: {frequency_step}\n"
                f"Max |S|: {S_to_dBm(data.S).max():.2f} dB\n"
                f"Min |S|: {S_to_dBm(data.S).min():.2f} dB"   
            )
            return comment
    
    @staticmethod
    def generate_power_sweep_comments(data: FluxMapData) -> str:
           
            powers = data.powers #in dBm
            frequencies = data.f/1e9 #in GHz

            # Ascending currents
            if data.powers[0] < data.powers[-1]:
                power_sweep_dir = 'right'
            # Descending currents
            else:
                power_sweep_dir = 'left'

            if is_linearly_spaced(frequencies):
                frequency_step = f'{np.round((frequencies[1] - frequencies[0])*1e6)} kHz'
            else:
                frequency_step = 'non-linear'

            comment = (
                f"Parked at current: {data.currents[0]/1e-3} mA\n"
                f"VNA vbw: {data.vna_meta['VBW']} Hz\n"
                f"VNA average: {data.vna_meta['average']}\n"
                f"Power sweep direction: {power_sweep_dir}\n"
                f"Power range: {powers[0]} to {powers[-1]} dBm\n"
                f"Power step: {np.round(powers[1] - powers[0],4)} dBm\n"
                f"Frequency range: {frequencies.min()} to {frequencies.max()} GHz\n"
                f"Frequency step: {frequency_step}\n"
                f"Max |S|: {S_to_dBm(data.S).max():.2f} dB\n"
                f"Min |S|: {S_to_dBm(data.S).min():.2f} dB"   
            )
            return comment
    

    @staticmethod
    def generate_two_tone_comments(data: FluxMapData) -> str:
           
            currents = data.currents/1e-3 #in mA
            frequencies = data.f/1e9 #in GHz

            # Ascending currents
            if data.currents[0] < data.currents[-1]:
                current_sweep_dir = 'right'
            # Descending currents
            else:
                current_sweep_dir = 'left'

            if is_linearly_spaced(frequencies):
                frequency_step = f'{np.round((frequencies[1] - frequencies[0])*1e6)} kHz'
            else:
                frequency_step = 'non-linear'

            comment = (
                f"VNA power: {data.vna_meta['power']} dBm\n"
                f"VNA vbw: {data.vna_meta['VBW']} Hz\n"
                f"VNA average: {data.vna_meta['average']}\n"
                f"Current sweep direction: {current_sweep_dir}\n"
                f"Current range: {currents[0]} to {currents[-1]} mA\n"
                f"Current step: {np.round(currents[1] - currents[0],4)} mA\n"
                f"Frequency range: {frequencies.min()} to {frequencies.max()} GHz\n"
                f"Frequency step: {frequency_step}\n"
                f"Max |S|: {S_to_dBm(data.S).max():.2f} dB\n"
                f"Min |S|: {S_to_dBm(data.S).min():.2f} dB"   
            )
            return comment
    
    @staticmethod
    def add_text_box(ax: Axes, text: str, position: Tuple[float, float]) -> None:
        
        # Set the position to the right of the colorbar
        text_x = position[0]
        text_y = position[1]
        
        # Add a text box with the given text to the specified position on the plot
        ax.text(text_x, text_y, text, transform=ax.figure.transFigure, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))






    def update_pcolormesh(self, new_x: np.ndarray, new_y: np.ndarray, new_z: np.ndarray):

        if self.pcolor_mesh is None:
            self.pcolor_x = new_x
            self.pcolor_y = new_y
            self.pcolor_z = new_z

            # Create pcolormesh
            # Create initial pcolormesh and display it
            self.pcolor_fig, self.pcolor_ax = plt.subplots()
            self.pcolor_mesh = self.pcolor_ax.pcolormesh([self.pcolor_x], [self.pcolor_y], [self.pcolor_z])
            plt.tight_layout()
            display(self.pcolor_fig)

            # self.pcolor_fig, self.pcolor_ax, self.pcolor_mesh, self.pcolor_bar = self._initial_pcolormesh(self.pcolor_x, self.pcolor_y, self.pcolor_z)
            # display(self.pcolor_fig)
            # clear_output(wait=True)
        else:
            # Update the data
            self.pcolor_x = np.vstack([self.pcolor_x, new_x])
            self.pcolor_y = np.vstack([self.pcolor_y, new_y])
            self.pcolor_z = np.vstack([self.pcolor_z, new_z])


            # Regenerate the mesh with updated data
            x_final, y_final, z_final = self.generate_flat_mesh(self.pcolor_x, self.pcolor_y, self.pcolor_z)

            #self.pcolor_ax.clear()
            self.pcolor_mesh = self.pcolor_ax.pcolormesh(x_final, y_final, z_final, cmap=self.pcolor_cmap)

            if self.pcolor_bar is not None:
                self.pcolor_bar.remove()    
                self.pcolor_bar = self.pcolor_fig.colorbar(self.pcolor_mesh, ax=self.pcolor_ax)
            else:
                self.pcolor_bar = self.pcolor_fig.colorbar(self.pcolor_mesh, ax=self.pcolor_ax)            
        
            
            clear_output(wait=True)
            plt.tight_layout()
            display(self.pcolor_fig)
            
    def plot_trace_mag(self, f: Union[np.ndarray, list], z: Union[np.ndarray, list], S_parameter: str ='S12', freq_unit: str ='GHz', mag_scale: str ='dB') -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Plots the magnitude and phase of one or many complex-valued functions (e.g., an S-parameter) against frequency.

        Parameters:
        - f (array-like): Array of frequencies at which the measurements are taken.
        - z (array-like): Complex-valued measurements corresponding to frequencies in `f`.
        - S_parameter (str, optional): Label of the S-parameter (default 'S12').
        - title (str, optional): Title of the plot.
        - freq_unit (str, optional): Unit for the frequency axis ('GHz', 'MHz', etc.).
        - mag_scale (str, optional): Scale for magnitude ('dB' for decibels, 'linear' for linear scale).

        Returns:
        - fig, ax: Figure and axis objects of the plot.
        """
        # Validate inputs
        if not (isinstance(f, (list, np.ndarray)) and isinstance(z, (list, np.ndarray))):
            raise ValueError("Frequency and measurement inputs must be array-like.")

        if len(f) != len(z):
            raise ValueError("Frequency and measurement arrays must be of the same length.")
        
        # Convert inputs to numpy arrays if they are lists and transpose to be able to plot multiple traces
        f = np.array(f).transpose()
        z = np.array(z).transpose()

        # Create subplots
        fig, ax = plt.subplots()

        # Plot magnitude
        color = 'tab:blue'
        if mag_scale.lower() == 'db':
            ax.plot(f / 1e9, 20 * np.log10(np.abs(z)), color=color)
            ax.set_ylabel(f'|{S_parameter}| in dB', color=color)
        else:
            ax.plot(f / 1e9, np.abs(z), color=color)
            ax.set_ylabel(f'|{S_parameter}|', color=color)
        ax.set_xlabel(f'Frequency ({freq_unit})')
        ax.tick_params(axis='y', labelcolor=color)
        
        return fig, ax

    def plot_trace_mag_phase(self, f: Union[np.ndarray, list], z: Union[np.ndarray, list], S_parameter: str ='S12', freq_unit: str ='GHz', mag_scale: str ='dB') -> Tuple[Figure, Tuple[Axes, Axes]]:
        """
        Plots the magnitude and phase of one or many complex-valued functions (e.g., an S-parameter) against frequency.

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
        
        # Convert inputs to numpy arrays if they are lists and transpose to be able to plot multiple traces
        f = np.array(f).transpose()
        z = np.array(z).transpose()

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

        
        return fig, (ax1, ax2)


    def plot_trace_mag_phase_stacked(self, f: Union[np.ndarray, list], z: Union[np.ndarray, list], S_parameter='S12', title='', freq_unit='GHz', mag_scale='dB') -> Tuple[Figure, Tuple[Axes, Axes]]:
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
        
        # Convert inputs to numpy arrays if they are lists and transpose to be able to plot multiple traces
        f = np.array(f).transpose()
        z = np.array(z).transpose()

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

        

        return fig, (ax_magnitude, ax_phase)
    

    # 3d
    def generate_flat_mesh(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not (np.shape(x) == np.shape(y) == np.shape(z)):
            raise ValueError("Arrays x, y and z must be of the same shape, i.e. datapoint z[i][j] at coordinates (x[i][j], y[i][j]).")

        x_final = np.full((x.shape[0]*2, x.shape[1]+1), np.nan)
        y_final = np.full_like(x_final, np.nan)

        z_final = np.full((z.shape[0]*2 -1, z.shape[1]), np.nan)

        for i,row in enumerate(x):
            if i == 0:
                x_final[2*i] = (x[i][0])
                x_final[2*i +1] = (x[i][0] + (x[i+1][0] - x[i][0])/2)
            elif i == len(x)-1:
                x_final[2*i] = (x[i][0] - (x[i][0] - x[i-1][0])/2)
                x_final[2*i +1] = (x[i][0])
            else:
                x_final[2*i] = (x[i][0] - (x[i][0] - x[i-1][0])/2)
                x_final[2*i +1] = (x[i][0] + (x[i+1][0] - x[i][0])/2)

        for i,row in enumerate(y):
            halfway_points = (y[i][:-1] + y[i][1:]) / 2
            y_final[2*i][1:-1] = y_final[2*i +1][1:-1] = halfway_points
            y_final[2*i][0] = y_final[2*i +1][0] = y[i][0]
            y_final[2*i][-1] = y_final[2*i +1][-1] = y[i][-1]

        for i,row in enumerate(z):
            z_final[2*i] = row

            if 2*i+1 < len(z_final):
                z_final[2*i +1] = row

        return x_final, y_final, z_final


    def plot_flat_pcolormesh(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, vmax=None, vmin=None, cmap='viridis') -> Tuple[Figure, Axes, Colorbar]:
        """
        Plots a flat pcolormesh for data points on a grid that is potentially irregular in y.
        """

        if not (np.shape(x) == np.shape(y) == np.shape(z)):
            raise ValueError("Arrays x, y and z must be of the same shape, i.e. datapoint z[i][j] at coordinates (x[i][j], y[i][j]).")
            
        # generate new mesh, to account for irregularity in y
        x_final, y_final, z_final = self.generate_flat_mesh(x,y,z)


        # Create figure and axes
        fig, ax = plt.subplots()

    
        # Create the pcolormesh plot
        mesh = ax.pcolormesh(x_final, y_final, z_final, cmap=cmap, vmax=vmax, vmin=vmin)

    
        # Create a colorbar for the plot
        cbar = fig.colorbar(mesh, ax=ax)
       
        # return results
        return fig, ax, cbar
    
   