# Regular Modules
import numpy as np
import unittest
import numpy as np
import os

# Test Subject Modules
from work_dir import loader as l

class TestLoaderMethods(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.addTypeEqualityFunc(np.ndarray, self.compare_np_array)
    
    def compare_np_array(self, first, second, msg=None):
        if first.shape != second.shape:
            if not msg:
                msg = "Numpy array shape mismatch."
            raise self.failureException(msg)
        if first.dtype == float or second.dtype == float:
            if not np.allclose(first, second):
                if not msg:
                    msg = "Numpy array (dtype=float) mismatch by allclose measure."
                raise self.failureException(msg)
        else:
            if not (first == second).all():
                if not msg:
                    msg = "Numpy array elementwise mismatch."
                raise self.failureException(msg)
    
    @staticmethod
    def cartesian_to_polar(cartesian_points, drone_position):
        # Converts cartesian points back to polar points assuming SweepDict implementation
        x, y = drone_position
        xs, ys = np.hsplit(cartesian_points, 2)
        xs -= x
        ys = y - ys
        cartesian_points = np.concatenate([xs, ys], axis=1)
        distances = np.expand_dims(np.sqrt(np.sum(np.square(cartesian_points), axis=1)), 1)
        distances *= 1000 # Convert back to millimeters
        angles = np.degrees(np.arctan2(ys, xs))
        angles[angles < 0] += 360 # Make all angles positive
        return np.concatenate([angles, distances], axis=1)
    
    @staticmethod
    def is_in_order(sweep_dict_keys):
        # Check that values in a list is in ascending order
        for i in range(len(sweep_dict_keys) - 1):
            if sweep_dict_keys[i] > sweep_dict_keys[i + 1]:
                return False
        return True
    
    def setUp(self):
        flight_path = os.path.join("tests", "integration", "test_data", "FlightPath.csv")
        lidar_path = os.path.join("tests", "integration", "test_data", "LIDARPoints.csv")
        self.last_id = 15
        self.sweep_dict = l.SweepDict(lidar_path, flight_path, self.last_id)
        self.keys = list(self.sweep_dict.keys())

    def test_SweepDict(self):
        '''Warning: This test assumes values in "test_data/FlightPath.csv" and
        "test_data/LIDARPoints.csv"! Furthermore, it assumes implementation decisions
        from _lidar_to_cartsian() in SweepDict: Distance in meters, inverted y value.'''

        self.assertTrue(TestLoaderMethods.is_in_order(self.keys),
            "Expected SweepDict to be in ascending order.")
        self.assertLessEqual(self.keys[-1], self.last_id,
            "Expected last id in dict to be less than or equal to the last_id argument.")
        self.assertEqual(self.sweep_dict[5].drone_position.tolist(), [17.18511, 8.036117],
            "Expected drone_position in SweepDict to match data in FlightPath.csv.")
        self.assertEqual(self.sweep_dict[2].lidar_polar[11].tolist(), [8.637695, 7002], 
            "Expected lidar_polar in SweepDict to match data in LIDARPoints.csv.")
        
        sweep = self.sweep_dict[5]
        reconverted = TestLoaderMethods.cartesian_to_polar(sweep.lidar_cartesian, sweep.drone_position)
        self.assertEqual(reconverted, sweep.lidar_polar,
            "Expected lidar_cartesian to correspond to lidar_polar in SweepDict.")
    
    def test_get_all_drone_positions(self):
        all_drone_positions = self.sweep_dict.get_all_drone_positions()
        total_length = sum([1 for sweep in self.sweep_dict.values()
                            if sweep.drone_position is not None])

        self.assertEqual(total_length, len(all_drone_positions),
            "Expected number of all drone positions to be the same as sum of numbers " +
            "of drone positions in all sweeps")
        self.assertEqual(self.sweep_dict[0].drone_position, all_drone_positions[0],
            "Expected values in all drone positions to match values in each sweep")
    
    def test_get_all_lidar_polar(self):
        all_lidar_polar = self.sweep_dict.get_all_lidar_polar()
        total_length = sum([len(sweep.lidar_polar)
                            for sweep in self.sweep_dict.values()
                            if sweep.lidar_polar is not None])

        self.assertEqual(total_length, len(all_lidar_polar),
            "Expected number of all lidar_polar to be the same as sum of numbers " +
            "of lidar_polar in all sweeps")
        self.assertEqual(self.sweep_dict[0].lidar_polar[5], all_lidar_polar[5],
            "Expected values in all lidar polar to match values in all sweeps")
    
    def test_get_all_lidar_cartesian(self):
        all_lidar_cartesian = self.sweep_dict.get_all_lidar_cartesian()
        total_length = sum([len(sweep.lidar_cartesian)
                            for sweep in self.sweep_dict.values()
                            if sweep.lidar_cartesian is not None])

        self.assertEqual(total_length, len(all_lidar_cartesian),
            "Expected number of all lidar_cartesian to be the same as sum of numbers " +
            "of lidar_cartesian in all sweeps")
        self.assertEqual(self.sweep_dict[0].lidar_cartesian[5], all_lidar_cartesian[5],
            "Expected values in all lidar cartesian to match values in all sweeps")