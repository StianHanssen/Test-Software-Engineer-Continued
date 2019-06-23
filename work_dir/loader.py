# Regular Modules:
import csv
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

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

class SweepDict(OrderedDict):
    '''
    Creates an ordered dafault dict for containing Sweep objects.
    '''
    def __init__(self, file_path_lidar, file_path_flight_path, last_id=33):
        super().__init__()
        self._csv_reader(file_path_lidar, "lidar_polar", last_id)
        self._csv_reader(file_path_flight_path, "drone_position", last_id)
        self._set_lidar_cartesian()
    
    def __getitem__(self, key):
        # Adapts behavior of defaultdict.
        if key not in self:
            self.__setitem__(key, Sweep())
        return super().__getitem__(key)

    def _csv_reader(self, file_path, key, last_id):
        # Read data from a CSV file into a dictionary.
        # Dictionary format: Dict[sweep_id] = Sweep.
        with open(file_path) as csv_data_file:
            reader = list(csv.reader(csv_data_file))
            index = 0
            sweep_id = 0
            while index < len(reader):
                # Get "header" row and turn values to int:
                if sweep_id + 1 > last_id:
                    return
                sweep_id, data_size = [int(value) for value in reader[index]]
                index += 1
                # Load the data_size number of lines into dictionary.
                if key == "drone_position":
                    data = to_np_array(reader[index: index + data_size])
                    self[sweep_id].drone_position = data
                elif key == "lidar_polar":
                    data = to_np_array(reader[index: index + data_size])
                    self[sweep_id].lidar_polar = data
                index += data_size
    
    def _set_lidar_cartesian(self):
        # Convert one sweep of polar LIDAR points to cartesian LIDAR points.
        not_all_none = any([(sweep.lidar_polar is not None) and (sweep.drone_position is not None)
                        for sweep in self.values()])
        assert not_all_none, "All sweeps lack either drone_position or lidar_polar!"
        for key, sweep in self.items():
            drone_position = sweep.drone_position
            lidar_polar = sweep.lidar_polar
            if lidar_polar is None:
                print("Warning: Skipping sweep %d, got lidar_polar can not be None" % (key))
                continue
            if  drone_position is None:
                print("Warning: Skipping sweep %d, got drone_position can not be None" % (key))
                continue
            x, y = sweep.drone_position
            angles, distances = np.hsplit(sweep.lidar_polar, 2)
            distances = distances / 1000 # Converting to meters, using '/=' affects lidar_polar.
            # Numpy can broadcast, thus executing the following equations for all LIDAR points:
            cartesian_xs = x + distances * np.cos(np.radians(angles))
            cartesian_ys = y - distances * np.sin(np.radians(angles))
            sweep.lidar_cartesian = np.concatenate((cartesian_xs, cartesian_ys), axis=1)
        
    def get_all_drone_positions(self):
        # Returns all drone positions or None if there are none.
        collected = np.ndarray((len(self.keys()), 2))
        count = 0
        for sweep in self.values():
            if sweep.drone_position is not None:
                collected[count] = sweep.drone_position
                count += 1
        if count == 0:
            return None
        return collected[:count]
    
    def get_all_lidar_polar(self):
        # Returns all polar lidar points or None if there are none.
        collected = []
        for sweep in self.values():
            if sweep.lidar_polar is not None:
                collected.append(sweep.lidar_polar)
        if not collected:
            return None
        return np.concatenate(collected, axis=0)
    
    def get_all_lidar_cartesian(self):
        collected = []
        for sweep in self.values():
            if sweep.lidar_cartesian is not None:
                collected.append(sweep.lidar_cartesian)
        if not collected:
            return None
        return np.concatenate(collected, axis=0)

def to_np_array(data_lines):
    # Converts list of strings to np.array with dtype float.
    if len(data_lines) == 1:
        data_lines = data_lines[0]
    return np.array(data_lines, dtype=float)

def read_mapping_csv(file_path):
    '''Read data from CSV file into numpy array.
    Intended for Mappings.cvs.
    Output: np.array of shape (N, 4) where N is number mappings.'''
    with open(file_path) as csv_data_file:
        csv_reader = list(csv.reader(csv_data_file))
        array = to_np_array(csv_reader)
        return array
    