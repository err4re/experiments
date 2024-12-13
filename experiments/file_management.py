import datetime
import numpy as np


def generate_filename(sample_name):
    """
    Generates a filename based on the current date, time, and a provided sample name.
    
    The filename is formatted as 'YYYYMMDD_HHMMSS_sample_name', ensuring a unique
    filename for different instances based on the timestamp.

    Parameters:
    - sample_name (str): The name of the sample to be included in the filename.

    Returns:
    - str: A string representing the generated filename.

    Raises:
    - ValueError: If the sample_name is not a valid string.
    """

    # Validate sample_name
    if not isinstance(sample_name, str):
        raise ValueError("Sample name must be a string.")

    # Optionally, sanitize sample_name here to remove/replace special characters

    # Generate filename with current date-time and sample name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{sample_name}"

    return filename



def save_data(filename, data_dict, file_path=''):
    """
    Saves experiment data to a compressed .npz file.

    Parameters:
    - filename (str): The name of the file to save the data to.
    - data_dict (dict): A dictionary where keys are the variable names and values are the data to be saved.

    Raises:
    - ValueError: If filename is not a string or if data_dict is not a dictionary.
    """

    # Validate inputs
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string.")

    if not isinstance(data_dict, dict):
        raise ValueError("Data must be provided in a dictionary.")

    # Save data to full path
    full_path = os.path.join(self.file_path, filename)
    np.savez(full_path, **self.data)
    print(f"Data saved to {full_path}.npz")


