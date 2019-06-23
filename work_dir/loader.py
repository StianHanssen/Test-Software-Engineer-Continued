# Regular Modules:
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

# Project Modules:
from .visualizer import display_drone_data

class Sweep():
    '''
    Container object to hold information related to one sweep.
    '''
    def __init__(self):
        self._drone_position = None
        self._lidar_polar = None
        self._lidar_cartesian = None
    
    def __repr__(self):
        string = "(drone_position: " + self._to_string(self._drone_position)
        string += "| lidar_polar: " + self._to_string(self._lidar_polar)
        string += "| lidar_cartesian: " + self._to_string(self._lidar_cartesian) + ")"
        return string
    
    @staticmethod
    def _to_string(attribute):
        if attribute is None:
            return "None"
        return str(attribute.shape)
    
    @property
    def drone_position(self):
        return self._drone_position

    @drone_position.setter
    def drone_position(self, value):
        if value is not None:
            assert isinstance(value, np.ndarray), "Expected drone_position of type np.ndarray"
            assert value.dtype == float, "Expected drone_position to have dtype float"
            assert value.shape == (2,), "Expected drone_position of shape (2,)"
            self._drone_position = value
        else:
            self._drone_position = None
    
    @property
    def lidar_polar(self):
        return self._lidar_polar

    @lidar_polar.setter
    def lidar_polar(self, value):
        if value is not None:
            assert isinstance(value, np.ndarray), "Expected lidar_polar of type np.ndarray"
            assert value.dtype == float, "Expected lidar_polar to have dtype float"
            assert len(value.shape) == 2 and value.shape[1] == 2, "Expected lidar_polar of shape (N, 2)"
            self._lidar_polar = value
        else:
            self._lidar_polar = None
    
    @property
    def lidar_cartesian(self):
        return self._lidar_cartesian

    @lidar_cartesian.setter
    def lidar_cartesian(self, value):
        if value is not None:
            assert isinstance(value, np.ndarray), "Expected lidar_cartesian of type np.ndarray"
            assert value.dtype == float, "Expected lidar_cartesian to have dtype float"
            assert len(value.shape) == 2 and value.shape[1] == 2, "Expected lidar_cartesian of shape (N, 2)"
            self._lidar_cartesian = value
        else: 
            self._lidar_cartesian = None

def to_np_array(data_lines):
    # Convert list into np.array of floats
    if len(data_lines) == 1:
        data_lines = data_lines[0]
    return np.array(data_lines, dtype=float)

def csv_reader(file_path, key, csv_dict=None):
    # Read data from a CSV file into a dictionary
    # Dictionary format: Dict[sweep_id][key] = np.array of n data lines from csv
    if csv_dict is None:
        csv_dict = defaultdict(Sweep)
    with open(file_path) as csv_data_file:
        csv_reader = list(csv.reader(csv_data_file))
        index = 0
        while index < len(csv_reader):
            # Get "header" row and turn values to int:
            try:
                sweep_id, data_size = [int(value) for value in csv_reader[index]]
            except ValueError:
                break
            index += 1
            # Load the data_size number of lines into dictionary
            if key == "drone_position":
                data = to_np_array(csv_reader[index: index + data_size])
                csv_dict[sweep_id].drone_position = data
            elif key == "lidar_polar":
                data = to_np_array(csv_reader[index: index + data_size])
                csv_dict[sweep_id].lidar_polar = data
            index += data_size
    return csv_dict

def read_mapping_csv(file_path):
    # Read data from CSV file into numpy array
    # Format: np.array of shape (N, 4) where N is number mappings
    with open(file_path) as csv_data_file:
        csv_reader = list(csv.reader(csv_data_file))
        array = to_np_array(csv_reader)
        return array

def get_ordered_list(sweep_dict):
    return list(sorted(sweep_dict.items(), key=lambda x: x[0]))

def lidar_to_cartesian(drone_position, lidar_polar):
    # Convert one sweep of LIDAR points to cartesian points
    # Output format np.array of shape (N, 2) for N LIDAR points
    assert drone_position.shape == (2,), "Expected drone_position shape of (2,)."
    assert len(lidar_polar.shape) == 2 and lidar_polar.shape[1] == 2, \
           "Expected lidar_polar of shape (N, 2) where N is a positive int."
    x, y = drone_position
    angles, distances = np.hsplit(lidar_polar, 2)
    distances = distances / 1000 # Converting to meters, using '/=' affects lidar_polar
    # Numpy can broadcast, thus executing the following equations for all LIDAR points:
    cartesian_xs = x + distances * np.cos(np.radians(angles))
    cartesian_ys = y - distances * np.sin(np.radians(angles))
    return np.concatenate((cartesian_xs, cartesian_ys), axis=1)

def add_cartesian_entry(sweep_dict):
    # Creates "lidar_cartesian" entry in sweep_dict if 
    # the keys "drone_position" and "lidar_polar" exist
    assert any([sweep.drone_position is not None for sweep in sweep_dict.values()]), \
           "Required entry 'drone_position' does not exist for any sweep in sweep_dict."
    assert any([sweep.lidar_polar is not None for sweep in sweep_dict.values()]), \
           "Required entry 'lidar_polar' does not exist for any sweep in sweep_dict."
    for sweep_id, sweep in sweep_dict.items():
        try:
            drone_position = sweep.drone_position
            lidar_polar = sweep.lidar_polar
            sweep_dict[sweep_id].lidar_cartesian = lidar_to_cartesian(drone_position, lidar_polar)
        except KeyError as key:
            print("Warning: Skipping id %d, due to missing key %s" % (sweep_id, key))

def SweepDict(file_path_lidar, file_path_flight_path):
    sweep_dict = csv_reader(file_path_lidar, "lidar_polar")
    sweep_dict = csv_reader(file_path_flight_path, "drone_position", sweep_dict)
    add_cartesian_entry(sweep_dict)
    return sweep_dict

if __name__ == '__main__':
    flight_path = os.path.join("data", "FlightPath.csv")
    lidar_path = os.path.join("data", "LIDARPoints.csv")

    sweep_dict = SweepDict(lidar_path, flight_path)
    #display_drone_data(sweep_dict)
    