# Regular Modules:
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

# Project Modules:
from .visualizer import display_drone_data

def to_np_array(data_lines):
    # Convert list into np.array of floats
    if len(data_lines) == 1:
        data_lines = data_lines[0]
    return np.array(data_lines, dtype=float)

def csv_reader(file_path, key, csv_dict=None):
    # Read data from a CSV file into a dictionary
    # Dictionary format: Dict[sweep_id][key] = np.array of n data lines from csv
    if csv_dict is None:
        csv_dict = defaultdict(dict)
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
            csv_dict[sweep_id][key] = to_np_array(csv_reader[index: index + data_size])
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

def lidar_to_cartesian(drone_position, lidar_points):
    # Convert one sweep of LIDAR points to cartesian points
    # Output format np.array of shape (N, 2) for N LIDAR points
    assert drone_position.shape == (2,), "Expected drone_position shape of (2,)."
    assert len(lidar_points.shape) == 2 and lidar_points.shape[1] == 2, \
           "Expected lidar_points of shape (N, 2) where N is a positive int."
    x, y = drone_position
    angles, distances = np.hsplit(lidar_points, 2)
    distances = distances / 1000 # Converting to meters, using '/=' affects lidar_points
    # Numpy can broadcast, thus executing the following equations for all LIDAR points:
    cartesian_xs = x + distances * np.cos(np.radians(angles))
    cartesian_ys = y - distances * np.sin(np.radians(angles))
    return np.concatenate((cartesian_xs, cartesian_ys), axis=1)

def add_cartesian_entry(sweep_dict):
    # Creates "cartesian_points" entry in sweep_dict if 
    # the keys "drone_position" and "lidar_points" exist
    assert any(["drone_position" in sweep for sweep in sweep_dict.values()]), \
           "Required entry 'drone_position' does not exist for any sweep in sweep_dict."
    assert any(["lidar_points" in sweep for sweep in sweep_dict.values()]), \
           "Required entry 'lidar_points' does not exist for any sweep in sweep_dict."
    for sweep_id, sweep in sweep_dict.items():
        try:
            drone_position = sweep["drone_position"]
            lidar_points = sweep["lidar_points"]
            sweep_dict[sweep_id]["cartesian_points"] = lidar_to_cartesian(drone_position, lidar_points)
        except KeyError as key:
            print("Warning: Skipping id %d, due to missing key %s" % (sweep_id, key))

def SweepDict(file_path_lidar, file_path_flight_path):
    sweep_dict = csv_reader(file_path_lidar, "lidar_points")
    sweep_dict = csv_reader(file_path_flight_path, "drone_position", sweep_dict)
    add_cartesian_entry(sweep_dict)
    return sweep_dict

if __name__ == '__main__':
    flight_path = os.path.join("data", "FlightPath.csv")
    lidar_path = os.path.join("data", "LIDARPoints.csv")

    sweep_dict = SweepDict(lidar_path, flight_path)
    display_drone_data(sweep_dict)
    