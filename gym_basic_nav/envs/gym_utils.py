import scipy
from scipy import stats
import numpy as np
from math import sqrt, asin, cos, sin, atan2, ceil

from .geom_utils import *

class Interactive(object):
    def __init__(self, top_right, bot_right, bot_left, top_left, interactive=False, permanent=True):
        self.interactive = interactive
        self.permanent = permanent

        self.top_right = np.array(top_right)
        self.bot_right = np.array(bot_right)
        self.bot_left  = np.array(bot_left)
        self.top_left  = np.array(top_left)

        self.center = np.mean([top_right, bot_right, bot_left, top_left])
        self.width = top_right[0]-top_left[0]
        self.height = top_right[1]-bot_right[1]

    def interact(self):
        pass

    def remove(self):
        pass

    def distance_to_point(self, p):
        return point_to_rectangle_distance(p, self.center, self.width, self.height)

    def closest_point_to(self, p):
        closest_points_to_segments = [closest_point_on_segment(p, s, t) for s, t in rectangle_edges(self.top_right,
                                                                                                    self.bot_right,
                                                                                                    self.bot_left,
                                                                                                    self.top_left)]

        distances = [np.linalg.norm(p - cp) for cp in closest_points_to_segments]
        idx = np.argmin(distances)

        return closest_points_to_segments[idx]

class WorldMap(object):
    def __init__(self, x_range, y_range, interactives):
        self.objects = interactives
        self.obstacles = list(filter(lambda o: not o.interactive, interactives))
        self.interactives = list(filter(lambda o: o.interactive, interactives))

    def location_of_interactives(self):
        locs = [i.center for i in self.interactives]
        return locs

    def range_and_bearing_to_closest_obstacle(self, x, y):
        p = np.array([x, y])
        dist = [(self.obstacles[i].distance_to_point(p), i) for i in range(len(self.obstacles))]
        distance_to_closest_obstacle, idx_closest = min(dist)

        closest_obstacle = self.obstacles[idx_closest]
        cp = closest_obstacle.closest_point_to(p)
        bearing_to_closest_obstacle = atan2(cp[1]-y, cp[0]-x)
        return distance_to_closest_obstacle, bearing_to_closest_obstacle

















































