#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:28:17 2023

@author: costantino_ai
"""
# Standard library imports
import os
import inspect
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
import random
import sys
import shutil
import pickle

# Third-party imports
import numpy as np
import pandas as pd
import torch

from modules import logging

def initialize_or_load_dataframe(file_path):
    """
    Initialize or load a DataFrame from a CSV file.

    :param file_path: The path to the CSV file.
    :type file_path: str
    :returns: The loaded or initialized DataFrame.
    :rtype: pd.DataFrame
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep=',')
    else:
        df = pd.DataFrame()
        df.to_csv(file_path, index=False)
        return df

def dataframe_to_dict(df):
    """
    Convert a DataFrame to a list of dictionaries.

    :param df: The DataFrame to convert.
    :type df: pd.DataFrame
    :returns: A list of dictionaries, each representing a row in the DataFrame.
    :rtype: list[dict]
    """
    return df.to_dict('records')

def dict_to_dataframe(data, columns):
    """
    Convert a list of dictionaries to a DataFrame.

    :param data: The data to convert.
    :type data: list[dict]
    :param columns: The column names for the DataFrame.
    :type columns: list[str]
    :returns: The constructed DataFrame.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(data, columns=columns)


def create_output_directory(directory_path):
    """
    Creates an output directory at the specified path.

    This function attempts to create a directory at the given path. 
    It logs the process, indicating whether the directory creation was successful or if any error occurred.
    If the directory already exists, it will not be created again, and this will also be logged.

    :param directory_path: The path where the output directory will be created.
    :type directory_path: str
    """
    try:
        # Log the attempt to create the output directory
        logging.debug(f"Attempting to create output directory at: {directory_path}")

        # Check if directory already exists to avoid overwriting
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            # Log the successful creation
            logging.info("Output directory created successfully.")
        else:
            # Log if the directory already exists
            logging.info("Output directory already exists.")
    except Exception as e:
        # Log any errors encountered during the directory creation
        logging.error(f"An error occurred while creating the output directory: {e}", exc_info=True)

def save_script_to_file(output_directory):
    """
    Saves the script file that is calling this function to the specified output directory.

    This function automatically detects the script file that is executing this function
    and creates a copy of it in the output directory.
    It logs the process, indicating whether the saving was successful or if any error occurred.

    :param output_directory: The directory where the script file will be saved.
    :type output_directory: str
    """
    try:
        # Get the frame of the caller to this function
        caller_frame = inspect.stack()[1]
        # Get the file name of the script that called this function
        script_file = caller_frame.filename

        # Construct the output file path
        script_file_out = os.path.join(output_directory, os.path.basename(script_file))

        # Log the attempt to save the script file
        logging.debug(f"Attempting to save the script file to: {script_file_out}")

        # Copy the script file to the output directory
        shutil.copy(script_file, script_file_out)

        # Log the successful save
        logging.info("Script file saved successfully.")
    except Exception as e:
        # Log any errors encountered during the saving process
        logging.error(f"An error occurred while saving the script file: {e}", exc_info=True)

def save_numpy_array(array, file_path):
    """
    Save a numpy array to a file.

    This function saves a given numpy array to a file at the specified path. The file format
    is determined by the file extension in the file_path.

    :param array: The numpy array to be saved.
    :type array: numpy.ndarray
    :param file_path: The file path where the array will be saved. The file format is inferred from the extension.
    :type file_path: str
    """
    with open(file_path, 'wb') as f:
        np.save(f, array)
    logging.info(f"Saved numpy array to {file_path}")

def save_ordered_dict(odict, filename):
    """
    Save an OrderedDict object to a pickle file.

    This function serializes an OrderedDict and saves it to a file in pickle format. 
    The file is created or overwritten at the specified path.

    :param odict: The OrderedDict to be saved.
    :type odict: collections.OrderedDict
    :param filename: Path to the file where the OrderedDict will be saved.
    :type filename: str
    """
    odict = sort_ordered_dict_by_key(odict)
    with open(filename, 'wb') as file:
        pickle.dump(odict, file)
        
def load_ordered_dict(filename):
    """
    Load an OrderedDict object from a pickle file.

    This function deserializes an OrderedDict from a specified pickle file. 
    It loads and returns the OrderedDict object contained in the file.

    :param filename: Path to the pickle file from which to load the OrderedDict.
    :type filename: str
    :returns: The loaded OrderedDict object.
    :rtype: collections.OrderedDict
    """
    with open(filename, 'rb') as file:
        return sort_ordered_dict_by_key(pickle.load(file))
    

def sort_ordered_dict_by_key(original_dict):
    """
    Sorts an OrderedDict by its keys.

    This function takes an OrderedDict and returns a new OrderedDict which is sorted based on the keys.
    The sorting is done in ascending order of the keys. This function does not modify the original
    OrderedDict but returns a sorted deep copy.

    :param original_dict: The OrderedDict to be sorted.
    :type original_dict: collections.OrderedDict
    :returns: A new OrderedDict sorted by keys.
    :rtype: collections.OrderedDict

    .. note::
        This function creates a deep copy of the original OrderedDict, so changes to the returned
        OrderedDict will not affect the original one.
    """
    # Create a deep copy of the original OrderedDict to maintain the integrity of the original data
    copied_dict = deepcopy(original_dict)

    # Sort the copied OrderedDict by its keys and return the sorted OrderedDict
    sorted_activations = OrderedDict(sorted(copied_dict.items(), key=lambda x: x[0]))
    return sorted_activations

def print_dict(d, indent=0):
    """
    Recursively prints a dictionary, formatting it with indentation for nested dictionaries.

    This function iterates through the dictionary, and for each key-value pair, it prints the key.
    If the value is also a dictionary, it recursively prints its key-value pairs with increased indentation.

    :param d: The dictionary to be printed.
    :type d: dict
    :param indent: The indentation level for nested dictionaries. Defaults to 0 for the top-level dictionary.
    :type indent: int
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f'"{str(key)}": ')
            print_dict(value, indent + 1)
        else:
            print("\t" * indent + f'"{str(key)}": "{str(value)}"')
    

class OutputLogger:
    """
    A context manager that logs output to a file and the console (stdout).

    This class is designed to be used with the 'with' statement to temporarily redirect
    stdout to both the console and a log file. It's useful for logging the output of a
    specific block of code.

    :ivar log: Whether or not to log output to a file. If True, messages will be written to file_path.
    :vartype log: bool
    :ivar file_path: The file path where messages will be written if log is True.
    :vartype file_path: str
    :ivar original_stdout: The original stdout stream, which will be restored when the context is exited.
    :vartype original_stdout: object
    :ivar log_file: The file object used to write output to the file_path.
    :vartype log_file: file object
    
    :Example:

    >>> with OutputLogger(True, out_text_path):
            # The rest of your code
    """
    def __init__(self, log: bool, file_path: str):
        """
        Initializes the OutputLogger object.
    
        :param log: Whether or not to log output to a file. If True, messages will be written to file_path.
        :type log: bool
        :param file_path: The file path where messages will be written if log is True.
        :type file_path: str
        """
        self.log = log
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        """
        Called when the context is entered. If log is True, opens the log file and redirects stdout to self.

        :returns: The OutputLogger object.
        :rtype: OutputLogger
        """
        if self.log:
            self.log_file = open(
                self.file_path, "w"
            )  # open the log file for writing
            sys.stdout = self  # redirect stdout to this OutputLogger object
            sys.stderr = self
            # Reconfigure logging to use the new stderr
            for handler in logging.root.handlers:
                handler.stream = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context is exited. If log is True, restores stdout to its original state and closes the log file.

        :param exc_type: The exception type, if any.
        :type exc_type: type
        :param exc_val: The exception value, if any.
        :type exc_val: Exception
        :param exc_tb: The traceback object, if any.
        :type exc_tb: traceback
        """
        if self.log:
            # Restore logging to use the original stderr
            for handler in logging.root.handlers:
                handler.stream = self.original_stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.log_file.close()
            
    def write(self, message: str):
        """
        Writes a message to stdout and the log file (if log is True).

        :param message: The message to write.
        :type message: str
        """
        self.original_stdout.write(message)  # write the message to the console
        if self.log and not self.log_file.closed:  # check if the file is not closed
            self.log_file.write(message)  # write the message to the log file

    def flush(self):
        """
        Flushes the stdout and the log file (if log is True).
        """
        self.original_stdout.flush()  # flush the console
        if self.log and not self.log_file.closed:  # check if the file is not closed
            self.log_file.flush()  # flush the log file
            
def set_random_seeds(seed=42):
    """
    Set the random seed for reproducibility in PyTorch, NumPy, and Python's random module.

    This function sets the seed for random number generation in PyTorch, NumPy, and Python's built-in random module.
    It also configures PyTorch to use deterministic algorithms and disables the benchmark mode for convolutional layers
    when CUDA is available, to ensure reproducibility.

    :param seed: The random seed. Defaults to 42.
    :type seed: int
    """
    # Set the seed for generating random numbers
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def create_run_id():
    """
    Generate a unique run identifier based on the current date and time.

    This function creates a string representing the current date and time in the format 'YYYYMMDD-HHMMSS'.
    It can be used to create unique identifiers for different runs or experiments.

    :returns: A string representing the current date and time.
    :rtype: str
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")




