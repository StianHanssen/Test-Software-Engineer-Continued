# Credit: https://matplotlib.org/gallery/event_handling/image_slices_viewer.html
# Modified version

import numpy as np
import matplotlib.pyplot as plt

class IndexTracker(object):
    def __init__(self, ax, sweep_dict):
        self.ax = ax[0]
        self.ax.set_title('Use scroll wheel to scroll through sweeps.')

        self.sweep_dict = sweep_dict
        self.slices = len(sweep_dict.keys())
        self.ind = 0

        all_drone_positions = collect_all(sweep_dict, "drone_position")
        all_cartesian_points = collect_all(sweep_dict, "cartesian_points")
        # Initializing with all points to get the right scale
        self.points = self.ax.scatter(all_cartesian_points[:, 0],
                                      all_cartesian_points[:, 1],
                                      s=1, c='r', marker='.')
        self.positions = self.ax.scatter(all_drone_positions[:, 0],
                                         all_drone_positions[:, 1],
                                         s=5, c='orange', marker='s')
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.inaxes == self.ax:
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()
    
    def onpick(self, event):
        ind = event.ind[0]
        self.ind = ind
        self.update()

    def update(self):
        cartesian_points = self.sweep_dict[self.ind].lidar_cartesian
        drone_posision = self.sweep_dict[self.ind].drone_position
        # Update drone position and the corresposinding sweep
        self.points.set_offsets(np.c_[cartesian_points[:, 0], cartesian_points[:, 1]])
        self.positions.set_offsets(np.c_[drone_posision[0], drone_posision[1]])
        # Update label with to new sweep ID
        self.ax.set_ylabel('ID: %s' % self.ind)
        self.points.axes.figure.canvas.draw()

def collect_all(sweep_dict, key): # Might be better to make a real SweepDict for this...
    sweep_items = sorted(sweep_dict.items(), key=lambda x: x[0])
    collected = [sweep[key] for _, sweep in sweep_items
                 if key in sweep]
    if not collected: # No points of key found in sweep_dict
        return None
    if key == "drone_position":
        # Expand dimensions so that arrays can be concatenated
        # This would be the same as stacking them
        collected = [np.expand_dims(x, axis=0) for x in collected]
    return np.concatenate(collected, axis=0)

def display_drone_data(sweep_dict):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    tracker = IndexTracker(ax, sweep_dict)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('pick_event', tracker.onpick)
    # Get all drone positions and cartesian points
    all_drone_positions = collect_all(sweep_dict, "drone_position")
    all_cartesian_points = collect_all(sweep_dict, "cartesian_points")
    # Draw a second view showing all sweeps in one
    ax[1].scatter(all_cartesian_points[:, 0], all_cartesian_points[:, 1], s=1, c='r', marker='.')
    ax[1].scatter(all_drone_positions[:, 0], all_drone_positions[:, 1], s=5, c='orange', marker='s', picker=True)
    ax[1].set_title('Click on the drone points (yellow)\nto see the sweep of that position.')
    plt.show()