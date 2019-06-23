# Regular Modules:
import numpy as np
import unittest
from unittest.mock import patch
from collections import defaultdict
import numpy as np
import os

# Test Subject Modules:
from work_dir import loader as l

class TestLoaderMethods(unittest.TestCase):

    def test_Sweep(self):
        sweep = l.Sweep()

        self.assertIsNone(sweep.drone_position,
            "Expected drone_position to be None")
        self.assertIsNone(sweep.lidar_polar,
            "Expected lidar_polar to be None")
        self.assertIsNone(sweep.lidar_cartesian,
            "Expected lidar_cartesian to be None")

        with self.assertRaises(AssertionError):
            sweep.drone_position = 5
        with self.assertRaises(AssertionError):
            sweep.drone_position = np.random.rand(5)
        with self.assertRaises(AssertionError):
            sweep.drone_position = np.random.randint(0, 10, (2))
        
        with self.assertRaises(AssertionError):
            sweep.lidar_polar = 5
        with self.assertRaises(AssertionError):
            sweep.lidar_polar = np.random.rand(5, 3)
        with self.assertRaises(AssertionError):
            sweep.lidar_polar = np.random.randint(0, 10, (10, 2))
        
        with self.assertRaises(AssertionError):
            sweep.lidar_cartesian = 5
        with self.assertRaises(AssertionError):
            sweep.lidar_cartesian = np.random.rand(5, 3)
        with self.assertRaises(AssertionError):
            sweep.lidar_cartesian = np.random.randint(0, 10, (10, 2))
    
    def test_to_np_array(self):
        data = [["0", "1", "2", "3", "4", "5"]]
        data2 = [["1.2", "3.6"], ["5.7", "4.5"]]
        self.assertTrue((l.to_np_array(data) == np.array([0, 1, 2, 3, 4, 5], dtype=float)).all(),
            "Expected np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])")
        self.assertTrue((l.to_np_array(data2) == np.array([[1.2, 3.6], [5.7, 4.5]], dtype=float)).all(),
            "Expected np.array([[1.2, 3.6], [5.7, 4.5]])")