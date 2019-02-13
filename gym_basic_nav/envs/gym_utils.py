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

        self.center = np.array([0.5*(top_right[0]+top_left[0]), 0.5*(top_right[1]+bot_right[1])])
        self.width = top_right[0]-top_left[0]
        self.height = top_right[1]-bot_right[1]

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
        self.x_range = int(x_range)
        self.y_range = int(y_range)
        self.image = 255*np.ones((self.y_range, self.x_range, int(3)))

        self.objects = interactives
        self.obstacles = list(filter(lambda o: not o.interactive, interactives))
        self.interactives = list(filter(lambda o: o.interactive, interactives))

        for obs in self.obstacles:
            r = obs.center[1]
            c = obs.center[0]
            min_row = max(int(r - obs.height/2.0), 0)
            max_row = min(int(r + obs.height/2.0), self.y_range)

            min_col = max(int(c - obs.width/2.0), 0)
            max_col = min(int(c + obs.width/2.0), self.x_range)

            if obs.permanent:
                self.image[min_row:max_row, min_col:max_col, :] = (204, 153, 102)
            else:
                self.image[min_row:max_row, min_col:max_col, :] = (179, 102, 204)

        for obs in self.interactives:
            r = obs.center[1]
            c = obs.center[0]
            min_row = max(int(r - obs.height/2.0), 0)
            max_row = min(int(r + obs.height/2.0), self.y_range-1)

            min_col = max(int(c - obs.width/2.0), 0)
            max_col = min(int(c + obs.width/2.0), self.x_range-1)

            self.image[min_row:max_row, min_col:max_col, :] = (0, 127, 0)

    def location_of_interactives(self):
        locs = np.array([i.center for i in self.interactives]).flatten()
        return locs

    def range_and_bearing_to_closest_obstacle(self, x, y):
        p = np.array([x, y])
        dist = [(self.obstacles[i].distance_to_point(p), i) for i in range(len(self.obstacles))]
        distance_to_closest_obstacle, idx_closest = min(dist)

        closest_obstacle = self.obstacles[idx_closest]
        cp = closest_obstacle.closest_point_to(p)
        bearing_to_closest_obstacle = atan2(cp[1]-y, cp[0]-x)
        return distance_to_closest_obstacle, bearing_to_closest_obstacle

    def point_is_in_obstacle(self, x, y, epsilon=0.25):
        row = int(y)
        col = int(x)

        if (row >=0 and row < self.image.shape[0] and col >= 0 and col < self.image.shape[1]):
            return (self.image[row, col, :] == (204, 153, 102)).all() or (self.image[row, col, :] == (179, 102, 204)).all()
        else:
            return True

    def segment_is_in_obstacle(self, x1,y1, x2,y2, epsilon=0.5):
        # Note: this is assuming that 1px = 1m
        a = np.array([x1,y1])
        b = np.array([x2,y2])
        L = np.linalg.norm(b-a)
        return any([self.point_is_in_obstacle(a[0] + i/L*(b[0]-a[0]), a[1] + i/L*(b[1]-a[1])) for i in range(ceil(L))])

    def point_is_in_keyzone(self, x, y):
        row = int(y)
        col = int(x)

        return (self.image[row, col, :] == (0, 127, 0)).all()

    def remove_wall(self):
        for obs in self.obstacles:
            if not obs.permanent:
                r = obs.center[1]
                c = obs.center[0]
                min_row = max(int(r - obs.height/2.0), 0)
                max_row = min(int(r + obs.height/2.0), self.y_range-1)

                min_col = max(int(c - obs.width/2.0), 0)
                max_col = min(int(c + obs.width/2.0), self.x_range-1)

                self.image[min_row:max_row, min_col:max_col, :] = (255, 255, 255)

        self.obstacles = list(filter(lambda o: o.permanent, self.obstacles))

















































