# Credit: https://matplotlib.org/gallery/event_handling/image__frames_viewer.html
# Modified version

# Regular Modules
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom Modules
from .loader import SweepDict

class _IndexTracker():
    def __init__(self, ax, sweep_dict):
        self._ax = ax[0]
        self._ax.set_title('Use scroll wheel to scroll through sweeps.')

        self._sweep_dict = sweep_dict
        self._frames = len(sweep_dict.keys())
        self._ind = 0

        all_drone_positions = sweep_dict.get_all_drone_positions()
        all_cartesian_points = sweep_dict.get_all_lidar_cartesian()
        # Initializing with all points to get the right scale
        self._points = self._ax.scatter(all_cartesian_points[:, 0],
                                      all_cartesian_points[:, 1],
                                      s=1, c='r', marker='.')
        self._positions = self._ax.scatter(all_drone_positions[:, 0],
                                         all_drone_positions[:, 1],
                                         s=5, c='orange', marker='s')
        self._update()

    def _update(self):
        cartesian_points = self._sweep_dict[self._ind].lidar_cartesian
        drone_posision = self._sweep_dict[self._ind].drone_position
        # Update drone position and the corresposinding sweep
        self._points.set_offsets(np.c_[cartesian_points[:, 0], cartesian_points[:, 1]])
        self._positions.set_offsets(np.c_[drone_posision[0], drone_posision[1]])
        # Update label with to new sweep ID
        self._ax.set_ylabel('ID: %s' % self._ind)
        self._points.axes.figure.canvas.draw()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.inaxes == self._ax:
            if event.button == 'up':
                self._ind = (self._ind + 1) % self._frames
            else:
                self._ind = (self._ind - 1) % self._frames
            self._update()
    
    def onpick(self, event):
        ind = event.ind[0]
        self._ind = ind
        self._update()

def display_drone_data(sweep_dict):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    tracker = _IndexTracker(ax, sweep_dict)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('pick_event', tracker.onpick)
    # Get all drone positions and cartesian points
    all_drone_positions = sweep_dict.get_all_drone_positions()
    all_cartesian_points = sweep_dict.get_all_lidar_cartesian()
    # Draw a second view showing all sweeps in one
    ax[1].scatter(all_cartesian_points[:, 0], all_cartesian_points[:, 1], s=1, c='r', marker='.')
    ax[1].scatter(all_drone_positions[:, 0], all_drone_positions[:, 1], s=5, c='orange', marker='s', picker=True)
    ax[1].set_title('Click on the drone points (yellow)\nto see the sweep of that position.')
    plt.show()

if __name__ == '__main__':
    flight_path = os.path.join("data", "FlightPath.csv")
    lidar_path = os.path.join("data", "LIDARPoints.csv")

    sweep_dict = SweepDict(lidar_path, flight_path)
    display_drone_data(sweep_dict)