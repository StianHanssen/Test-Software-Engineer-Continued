# Test-Software-Engineer

#### General Information:
The project runs on python 3 (programmed in python version 3.6.7).
All commands seen in install instructions should be written in terminal. Using a virtual environment is optional, you can skip step 2, 3 and 4 to set up the project without it.

#### Installation instructions:
1. Install python from: https://www.python.org/downloads/
2. Get virtual environment for python: `pip install virtualenv`
3. In the project folder, create a virtual environment: `virtualenv .env/neural_nets`
4. Start virtual environment: `source .env/neural_nets/bin/activate`
5. Install packages: `pip install -r requirements.txt`

Installation can vary from computer to computer. You may need to manually install the packages by `pip install [package]`. Anaconda can make installation easier.

#### Run instructions:
Run the following commands in the project folder:
* To run assignment 1: `python -m work_dir.visualizer`
* To run assignment 4: `python -m work_dir.path_finder.py`

#### Design decisions:
##### Assignment 1:
* While working with the given data I decided to store it in a dictionary containing numpy arrays. I chose dictionary for robustness. I assumed there was a reason why ID was used instead of simply listing the data in order. I thought there may be a chance in the future that IDs are not in order or certain IDs would be missing. Dictionary would handle such a scenario better. To still maintain some performance numpy arrays was used as their operations are native and fast.
* I did not make a class for `sweep_dict`, in hindsight I believe that could make code more neat. However, outside of initialization, `sweep_dict` does not have much special functiinality that `dict` can not already do. Perhaps the better solution would have been to make functions in module private using python's conventions.
* I decided that the best way to visualize the data was to convert the LIDAR points to their cartesian coordinate version, so that they together would make a countour of the rooms. I can then plot the drone positions and get a full understanding of how the drone operated.
* From a previous assignment I displayed 3D images by scrolling through slices of the image. I imagined I would do the same here, but let a slice be a particar sweep from the drone. This way the user can animate the movement at their own speed. The scroll lacks, however, fine movement. For that I made a second view showing the whole set of sweeps in one, and let the user click on a drone to see a partical sweep. This way it is not hard to precicly pick the sweep you want to see. I let the axis stay so the user can get a sense of scale.

##### Assignment 4:
* Reading assignment 3 and 4 I quickly had an idea on my approach. I imagined if I had the walls, I would make a visibility graph. Once the visibility graph is made, I can simply use A* with euclidian distance to goal as heuristic. However, there were a few obsticles:
    * Both assignment 3 and 4 seems dependent on assignment 5. However, evaluating what I would do best in the time constraint, I decided against doing assignment 5. Instead, I manually made the wall data, in what I imagined to be a realistic dataset. That being with certain walls lacking in the room the drone did not visit. In order to do this somewhat efficiently I used code in `fake_mapping.py`.
    * I estimated that building the full framework of for visibility graph, ordering walls to make "normals", shortest path algorithm as well as handling flawed data would take too much time. I therefore decided to use a framework for visibility graphs and shortest path; `pyvisgraph`. I would then focus on processing the input data, ordering walls for "normals", so that my implementation could perform well.
* I once again used a dictionary with numpy arrays to store the data. The keys were all the points defining walls and the values were a list of walls. Walls were stored as numpy arrays of two points so that fast operations on groups of data can be used. This allowas for fast look up of which walls belonged to a particular point.
* When doing most of the operations on walls, the points defining the walls were not the real coordinate points, but rather their index in a list of all points. For most part a point was defined as a single int and a wall as a numpy array of two ints. This made it a lot easier to have an overview of what was happening while developing, and it made certain operations involving comparisons easier. It also, made implementation of `balance_dict()` simpler as a new fake point was just a number (read code comments for more details).
* In order to make the set of walls work as a visibility graph, I would need to link them together in an order so that a "normal" can be defined. Then one can efficiently know which points are visible from a given position without risking paths being made through a point on the wrong side of a wall. I therefore, used a traversal method where walls are edges and corners are vertecies.
* The idea was to create polygons from the linked walls and define a normal by being inside/outside the polygon. This requires the walls to be linked in a circle without branching. I used three methods to transform the wall data so that they would fit the criteria:
    * `balance_dict()`: To remove branching by tracing along them and once an end is met, extend it back to where the branch began and let this extension take over the third connection in the branch.
    * `shift()`: If a point came up multiple times in the graph, it would cause problems in `pyvisgraph`. Lines could them pass through these points when they are not suppose to. I therefore shift duplicate point by a very small amount so they no longer overlap.
    * `inside_out_polygon()`: `pyvisgraph` always defined the blocking from outside and in. Therefore, I could not just make a polygon for the outside wall, as it won't block view in the visibility graph. To go around this, I extended the walls from a tiny point out and around the whole floor plan. You can imagined a ring that does not close up entirly. This opening is not visible because it is so small and because there are no points to see outside, it should not inhibit performance for search algorithms that may stray to the outside of the floorplan.

#### Short review:
A glaring lack in my assignment is tests. This was sacrificed for the end result as well as time for documentation. If I were to redo the assignment, I would have focused more on this aspect as well was a bit on code structure. It is certainly clear that more classes could have been made. An option parser such as I had in the initial code put in `visualizer.py` would make for a nicer interface with only using a main file. Perhaps it would have been more ideal to have combined assignment 4 and 5. Though I had many ideas for assignment 5, I 
felt more confident in assignment 1 and 4. The timing was a bit unfortunate as I had a presentation about my general visualization tool on neural networks today (18.06.19), which divided the work sessions on this assignment.
