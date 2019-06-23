# Regular Modules:
import numpy as np
import unittest
from collections import defaultdict

# Test Subject Modules:
from work_dir import path_finder as pf

class TestPathFinderMethods(unittest.TestCase):
    
    @staticmethod
    def walls_to_list(wall_dict):
        return {key: [wall.tolist() for wall in walls] for key, walls in wall_dict.items()}

    def test_get_wall_dict(self):
        walls = np.random.randint(0, 30, (30, 2))
        test_point = walls[0, 0]
        test_point2 = walls[0, 1]
        wall_dict = pf.get_wall_dict(walls)
        point_linked_to_point2 = wall_dict[test_point2][0][1] # Second point in wall 0

        self.assertTrue(all([wall[0] == test_point
                        for wall in wall_dict[test_point]]),
            "For key point1 all walls should have point1 as the first value.")
        self.assertIn(test_point2, [wall[1] for wall in wall_dict[point_linked_to_point2]], 
            "A wall made of point1 and point2 should exist both for key point1 and point2.")
        self.assertNotIn(0, [len(walls) for walls in wall_dict.values()],
            "It is not possible for a key to have no walls linked to it.")
        self.assertCountEqual(wall_dict.keys(), np.unique(walls.reshape(-1, 2)), 
            "Expected all points to exist in wall_dict.")
    
    def test_is_same_wall(self):
        wall1 = np.array([1, 2])
        wall2 = np.array([2, 1])
        wall3 = np.array([1, 5])
        wall4 = np.array([5, 2])

        self.assertTrue(pf.is_same_wall(wall1, wall1), 
                        "np.array([1, 2]) should be same the wall as np.array([1, 2]).")
        self.assertTrue(pf.is_same_wall(wall1, wall2),
                        "np.array([1, 2]) should be same the wall as np.array([2, 1].")
        self.assertTrue(pf.is_same_wall(wall2, wall1),
                        "np.array([2, 1] should be same the wall as np.array([1, 2]).")
        self.assertFalse(pf.is_same_wall(wall1, wall3),
                         "np.array([1, 2]) should not be the same wall as np.array([1, 5]).")
        self.assertFalse(pf.is_same_wall(wall1, wall4),
                         "np.array([1, 2]) should not be the same wall as np.array([5, 2]).")
    
    def test_select_next_wall(self):
        current_wall = np.array([1, 2])
        next_walls = [np.array([2, 1]), np.array([2, 5]), np.array([2, 7])]
        faulty_walls = [np.array([2, 1])]
        faulty_walls2 = []

        self.assertIn(pf.select_next_wall(current_wall, next_walls), next_walls[1:],
            "Expected next wall should be in next_walls and not be the same wall as current_wall.")
        self.assertIsNone(pf.select_next_wall(current_wall, faulty_walls), 
            "Expected None as the only choice for next wall was the same as current_wall.")
        self.assertIsNone(pf.select_next_wall(current_wall, faulty_walls2), 
            "Expected None as next_walls was empty, thus making it impossible to pick a next walls.")
    
    def test_add_wall_entry(self):
        wall_dict = defaultdict(list)
        pf.add_wall_entry(wall_dict, 1, 2)
        expected_wall1 = [1, 2]
        expected_wall2 = [2, 1]
        
        self.assertSequenceEqual(wall_dict[1][0].tolist(), expected_wall1,
            "Expected to find wall np.array([1, 2]) for key 1.")
        self.assertSequenceEqual(wall_dict[2][0].tolist(), expected_wall2,
            "Expected to find wall np.array([2, 1]) for key 2.")
    
    def test_remove_entry(self):
        # Not really a unit test. Can be something wrong with is_same_wall too
        wall_dict = {1: [np.array([1, 2]), np.array([1, 5])],
                     2: [np.array([2, 1]), np.array([2, 3])]}
        response = pf.remove_entry(wall_dict, np.array([1, 2]))
        wall_dict_l = TestPathFinderMethods.walls_to_list(wall_dict)

        self.assertTrue(response,
            "Expected True response after remove as wall exist in dict.")
        self.assertNotIn([1, 2], wall_dict_l[1],
            "Expected np.array([1, 2]) to have been removed from wall_dict.")
        self.assertNotIn([2, 1], wall_dict_l[2],
            "Expected np.array([2, 1]) to have been removed from wall_dict.")
        
        wall_dict_before = dict(wall_dict_l)
        response = pf.remove_entry(wall_dict, np.array([2, 5]))
        wall_dict_l = TestPathFinderMethods.walls_to_list(wall_dict)

        self.assertFalse(response,
            "Expected False response after remove as wall is not in dict.")
        self.assertDictEqual(wall_dict_before, wall_dict_l,
            "Expected wall_dict to be unchanged as entry is not in wall_dict.")
    
    def test_to_real_point(self):
        point = 0
        point2 = 55
        real_points = np.random.rand(10, 2)
        ref_dict = {55: 0}
        self.assertEqual(pf.to_real_point(point, ref_dict, real_points).tolist(), real_points[point].tolist(), 
            "Expected point 0 to be the first coordinate in real_points.")
        self.assertEqual(pf.to_real_point(point2, ref_dict, real_points).tolist(), real_points[point].tolist(), 
            "Expected point 55 to be converted to 0 and be the first coordinate in real_points.")
        
    def test_shift(self):
        current_point = np.array([1.5, 2.6])
        next_point = np.array([2.0, 0.0])
        shifted = pf.shift(current_point, next_point)
        shifted = shifted.tolist()
        current_point = current_point.tolist()
        self.assertNotEqual(current_point, shifted,
            "Expected current_point to have been shifted.")
        self.assertGreater(shifted[0], current_point[0], 
            "Expected the shift to go in positive x direction.")
        self.assertLess(shifted[1], current_point[1], 
            "Expected the shift to go in negiative y direction.")
        
        next_point = np.array([0.0, 5.0])
        shifted = pf.shift(current_point, next_point)
        shifted = shifted.tolist()
        self.assertLess(shifted[0], current_point[0], 
            "Expected the shift to go in negative x direction.")
        self.assertGreater(shifted[1], current_point[1], 
            "Expected the shift to go in positive y direction.")
    
    def test_get_bounding_box(self):
        all_points = np.random.rand(10, 2)
        max_point = all_points.max(axis=0)
        min_point = all_points.min(axis=0)
        bounding_box = pf.get_bouding_box(all_points)
        bound_max = np.array(bounding_box).max(axis=0)
        bound_min = np.array(bounding_box).min(axis=0)
        self.assertTrue(len(bounding_box) == 4,
            "Expected bounding box to be defined by 4 points.")
        unique_values = np.unique(np.stack(bounding_box, axis=0).reshape(-1))
        self.assertTrue(len(unique_values == 4),
            "Expected bounding box to be made of permutations of 4 unique values.")
        self.assertTrue(bound_max[0] > max_point[0] or bound_max[1] > max_point[1],
            "Expected bounding not to overlap with any points in all_points")
        self.assertTrue(bound_min[0] < min_point[0] or bound_min[1] < min_point[1],
            "Expected bounding not to overlap with any points in all_points")