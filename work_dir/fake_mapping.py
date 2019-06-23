# Regular Modules:
import matplotlib.pyplot as plt
import numpy as np
import csv

# Project Modules:
from .loader import SweepDict
from .visualizer import collect_all

def onpick(event):
    # Print given index and position of point when clicked on
    ind = event.ind[0]
    print(ind)
    print(event.artist.get_offsets()[ind])

if __name__ == '__main__':
    flight_path = "FlightPath.csv"
    lidar_path = "LIDARPoints.csv"
    debug = False

    sweep_dict = SweepDict(lidar_path, flight_path)
    all_cartesian_points = collect_all(sweep_dict, "cartesian_points")
    
    # Viewing points and show their index and position when clicking on them
    # This so I can find all the walls and make a fake mapping
    if debug:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.canvas.mpl_connect('pick_event', onpick)
        ax.scatter(all_cartesian_points[:, 0], all_cartesian_points[:, 1],
                s=1, c='r', marker='.', picker=True)
        plt.show()

    # Manually filled in list of walls using indicies
    walls = [[4449, 206],
             [206, 3276],
             [3276, 3157],
             [3157, 4245],
             [4245, 4223],
             [4223, 4274],
             [4274, 7964],
             [7964, 8930],
             [8930, 13817],
             [13817, 4589],
             [4589, 11698],
             [11698, 12055],
             [12055, 13028],
             [13028, 14067],
             [14067, 14750],
             [14750, 17397],
             [17397, 15663],
             [15663, 12895],
             [12895, 16483],
             [12895, 206],
             
             [360, 2445],
             [2445, 4478],
             [4478, 5500],
             [5500, 8667],
             [8667, 9165],
             [9165, 9186],
             [9186, 12819],
             [12819, 12852],
             [12852, 16370], 
             [16370, 16424],
             [16424, 16982],
             [16424, 2445]]
    
    # Used to see if plugged in walls are correct
    if debug:
        for wall in walls:
            p1, p2 = wall
            wall = np.array([all_cartesian_points[p1], all_cartesian_points[p2]])
            plt.plot(*wall.T, c='b')
        plt.show()

    
    for i, wall in enumerate(walls):
        p1, p2 = wall
        walls[i] = all_cartesian_points[p1].tolist() + all_cartesian_points[p2].tolist()
    with open('FakeMapping.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for wall in walls:
            filewriter.writerow(wall)