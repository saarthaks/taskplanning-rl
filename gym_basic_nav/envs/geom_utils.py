import scipy
from scipy import stats
import numpy as np
from math import sqrt
from itertools import product

def closest_point_on_segment(p, a, b):
    lsqr = (a[0]-b[0])**2 + (a[1]-b[1])**2
    pa = p-a
    ba = b-a

    if (lsqr < 1e-15):
        return a

    t_opt = pa.dot(ba) / lsqr
    t_opt = max(min(t_opt, 1), 0)
    cp = a + t_opt * (b-a)
    return cp

def point_to_segment_distance(p, a, b):
    cp = closest_point_on_segment(p, a, b)
    return np.linalg.norm(cp - p)

def point_to_rectangle_distance(p, ca, wa, ha):

    if (abs(p-ca) < np.array([wa/2.0, ha/2.0])).all():
        return 0.

    a1 = ca + np.array([wa/2.0, ha/2.0])
    a2 = ca + np.array([wa/2.0, -ha/2.0])
    a3 = ca + np.array([-wa/2.0, -ha/2.0])
    a4 = ca + np.array([-wa/2.0, ha/2.0])

    d1 = point_to_segment_distance(p, a1, a2)
    d2 = point_to_segment_distance(p, a2, a3)
    d3 = point_to_segment_distance(p, a3, a4)
    d4 = point_to_segment_distance(p, a4, a1)
    return min([d1, d2, d3, d4])

def rectangle_edges(z1,z2,z3,z4):
    yield (z1, z2)
    yield (z2, z3)
    yield (z3, z4)
    yield (z4, z1)
