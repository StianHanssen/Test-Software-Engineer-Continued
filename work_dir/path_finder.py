# Regular Modules:
import pyvisgraph as vg
from collections import defaultdict
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# Project Modules:
from .loader import read_mapping_csv, SweepDict, get_ordered_list

def print_dict(wall_dict):
    # Prints dict nicely
    for key, value in wall_dict.items():
        print(key, value)

def get_wall_dict(wall_indices):
    '''Converts np.array of shape (N, 2) to dict of shape
    Dict[point_index] = [wall1, wall2, ...] where walls are
    np.arrays (shape (2,)) of two points indicies. Point_index
    is a part of each of these walls.
    The point_index is always the first point in its walls:
    wall_dict[point_index] = [np.array(point_index, other_index), ..]'''
    points = np.unique(wall_indices.reshape(-1))
    wall_dict = defaultdict(list)
    for point in points:
        for wall in wall_indices:
            if point in wall:
                if point != wall[0]:
                    wall = wall[np.array([1, 0], dtype=int)]
                wall_dict[point].append(wall)
    return wall_dict

def is_same_wall(wall1, wall2):
    '''For a corresponding pair of points having
    the same wall, the wall is reversed:
    wall_dict[5] = [np.array([5, 2]), np.array([5, 4])]
    wall_dict[2] = [np.array([2, 5])]'''
    point1, point2 = wall1
    return (point1 in wall2) and (point2 in wall2)

def select_next_wall(current_wall, next_walls):
    '''current_wall are is in next_walls. They are all linked by
    one point. This function picks one of the walls that are not
    current_wall so that traversal goes forward.'''
    for wall in next_walls:
        if not is_same_wall(current_wall, wall):
            return wall

def traverse(point, wall, wall_dict, goal):
    '''Start with point and a wall belonging to point
    traverse wall to wall using wall_dict untill goal is
    reached. The function assumes goal is reachable from
    point, returns None if it is not the case.'''
    polygon = []
    for _ in range(len(wall_dict.keys())):
        polygon.append(point)
        wall = select_next_wall(wall, wall_dict[point])
        point = wall[1]
        if (point == goal).all():
            return polygon
    return None

def get_polygons(wall_dict): # TODO: Add timed stop for while loop
    '''Attempts to make convex polygons by starting at a
    point in a wall and traversing until we have come around
    finding the point again. Do the same process for remaining
    walls untill there are no walls.'''
    polygons = []
    wall_dict_keys = set(wall_dict.keys())
    while wall_dict_keys:
        wall = wall_dict[list(wall_dict_keys)[0]][0]
        p1, p2 = wall
        polygon = traverse(p2, wall, wall_dict, p1)
        polygon.append(p1)
        wall_dict_keys -= set(polygon)
        polygons.append(polygon)
    return polygons

def add_wall_entry(wall_dict, point1, point2):
    '''Add wall np.array([point1, point2]) / np.array([point2, point1])
    to each of the point keys.'''
    wall_dict[point1].append(np.array([point1, point2]))
    wall_dict[point2].append(np.array([point2, point1]))

def remove_entry(wall_dict, wall):
    '''Remove wall from the two point entries
    making up the wall.'''
    p1, p2 = wall
    # Would use list.remove(), with np.arrays
    wall_dict[p2] = [w for w in wall_dict[p2] 
                     if not is_same_wall(wall, w)]
    wall_dict[p1] = [w for w in wall_dict[p1]
                     if not is_same_wall(wall, w)]

def back_track(point, wall_dict, ref_dict, fake_point):
    '''Starts at an end point and traverse in the only direction it can go until
    it either finds a point connected to 3 walls or point that is connected to one wall.
    While traversing it creates a parallel path with fake points that starts with the given
    point. If the traversal reaches a point with one wall, connect the final fake point to this
    single point. If the traversal reaches a point connected to 3 walls, disconnect one of the
    3 walls and connect that to the final fake point.'''
    current_wall = wall_dict[point][0]
    current_point = current_wall[1]
    prev_fake_point = point
    for _ in range(len(wall_dict.keys())):
        connected_walls = wall_dict[current_point]
        # Traversed to point with two walls, continue making paralell path
        if len(connected_walls) == 2:
            add_wall_entry(wall_dict, prev_fake_point, fake_point)
            
            #Update fake point with a new fake index
            prev_fake_point = fake_point
            fake_point += 1
            # Link parallell points
            ref_dict[fake_point] = current_point
            
            current_wall = select_next_wall(current_wall, connected_walls)
            # To select next point, take second point in wall (first is current_point)
            current_point = current_wall[1]
        elif len(connected_walls) == 3:
            for wall in connected_walls:
                # Pick the first wall that is not current wall
                if not is_same_wall(current_wall, wall):
                    add_wall_entry(wall_dict, prev_fake_point, fake_point)

                    # Replace the selected wall (current_point to next point wall[1])
                    # with a connection between fake_point and the next point.
                    remove_entry(wall_dict, wall)
                    add_wall_entry(wall_dict, fake_point, wall[1])
                    ref_dict[fake_point] = current_point
                    fake_point += 1
                    return fake_point
        elif len(connected_walls) == 1:
            # Just connect last fake point to current point
            add_wall_entry(wall_dict, prev_fake_point, current_point)
            return fake_point
        else:
            raise Exception()
    return fake_point

def balance_dict(wall_dict):
    '''Attempts to transform the graph wall_dict represents into
    a circular graph with no branching.'''
    points = list(wall_dict.keys())
    fake_point = len(points)
    ref_dict = dict()
    for point in points:
        connected_walls = wall_dict[point]
        if len(connected_walls) == 1:
            fake_point = back_track(point, wall_dict, ref_dict, fake_point)
    return ref_dict

def to_real_point(point, ref_dict, real_points):
    # Transforms index point to real coordinate point
    if point in ref_dict:
        point = ref_dict[point]
    return real_points[point]

def shift(current_point, next_point):
    '''Shift current_point by a small amount in a direction
    that does keep the polygon convex'''
    delta = 0.00001
    shift = np.sign(next_point - current_point) * delta
    return current_point + shift

def to_real_polygons(polygons, ref_dict, real_points):
    '''Converts index points in real polygon to real coordinate points
    and shifts overlapping points by a small amount so they do not overlap.'''
    converted_polygons = []
    for polygon in polygons:
        poly = []
        for i, point in enumerate(polygon):
            real_point = to_real_point(point, ref_dict, real_points)
            if real_point.tolist() in poly:
                i = (i + 1) % len(polygon)
                next_point = to_real_point(polygon[i], ref_dict, real_points)
                real_point = shift(real_point, next_point)
            poly.append(real_point.tolist())
        poly = np.array(poly)
        converted_polygons.append(poly)
    return inside_out_polygon(converted_polygons)

def go_right(polygon):
    '''Check which direction the boundind box
    should be iterated.'''
    index = polygon[:, 1].argmin()
    prev_point_x = polygon[index - 1][0]
    point_x = polygon[index][0]
    direction = prev_point_x - point_x
    if direction == 0:
        next_point_x = polygon[(index + 1) % len(polygon)][0]
        direction = point_x - next_point_x
    if direction == 0:
        raise Exception()
    return direction > 0

def get_bouding_box(all_points):
    # Creates a bounding box around points
    delta = 0.00001
    max_point = all_points.max(axis=0) + delta
    min_point = all_points.min(axis=0) - delta
    max_x, max_y = max_point
    min_x, min_y = min_point
    return [min_point, np.array([min_x, max_y]),
            max_point, np.array([max_x, min_y])]

def inside_out_polygon(polygons):
    '''Adds a bounding box to the polygon that goes
    outside all the points in the polygon. Effectly making
    the inside of the polygon become the outside.'''
    delta = 0.00001
    all_points = np.concatenate(polygons, axis=0)
    min_y_index = all_points[:, 1].argmin()
    min_y_point = all_points[min_y_index]
    new_polygons = []
    for polygon in polygons:
        if min_y_point in polygon:
            bouding_box = get_bouding_box(all_points)
            min_y_poly_index = polygon[:, 1].argmin()
            if go_right(polygon):
                delta *= -1
                bouding_box = list(reversed(bouding_box))
            last_point = min_y_point + delta
            bouding_box.append(last_point)
            polygon = np.concatenate([polygon[:min_y_poly_index + 1],
                                      np.array(bouding_box),
                                      polygon[min_y_poly_index + 1:]], axis=0)
        new_polygons.append(polygon)
    return new_polygons

def get_shortest_path(mapping_path, start_point, end_point):
    walls = read_mapping_csv(mapping_path)
    walls = walls.reshape(-1, 2, 2)

    points = np.unique(walls.reshape(-1, 2), axis=0)
    point_to_inidices = {tuple(point): i for i, point in enumerate(points)}

    wall_indices = np.array([point_to_inidices[tuple(point)] for point in walls.reshape(-1, 2)])
    wall_indices = wall_indices.reshape(len(walls), 2)

    wall_dict = get_wall_dict(wall_indices)
    ref_dict = balance_dict(wall_dict)

    polygons = get_polygons(wall_dict)
    real_polygons = to_real_polygons(polygons, ref_dict, points)
    for poly in real_polygons:
        for i in range(len(poly) - 1):
            line = np.array([poly[i], poly[i + 1]])
            plt.plot(*line.T, 'b')
        line = np.array([poly[-1], poly[0]])
        plt.plot(*line.T, 'b')

    real_polygons = [[vg.Point(*point) for point in poly] for poly in real_polygons]

    graph = vg.VisGraph()
    graph.build(real_polygons)
    shortest = graph.shortest_path(vg.Point(*start_point), vg.Point(*end_point))
    for i in range(len(shortest) - 1):
        p = shortest[i]
        next_p = shortest[i + 1]
        line = np.array([[p.x, p.y], [next_p.x, next_p.y]])
        plt.plot(*line.T, 'r')
    plt.plot(*start,'o', c='orange',label='Start Point')
    plt.plot(*end,'go',label='End Point')
    plt.legend()
    count = 35
    with open('data/FlightPath.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(' ')
        for point in shortest:
            filewriter.writerow([count, 1])
            filewriter.writerow([point.x, point.y])
            count += 1

    plt.show()


if __name__ == '__main__':
    flight_path = os.path.join("data", "FlightPath.csv")
    lidar_path = os.path.join("data", "LIDARPoints.csv")

    sweep_dict = SweepDict(lidar_path, flight_path)
    sweeps = get_ordered_list(sweep_dict)
    start = sweeps[0][1]["drone_position"]
    end = sweeps[-1][1]["drone_position"]
    get_shortest_path('FakeMapping.csv', start, end)