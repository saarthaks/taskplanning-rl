import gym
from gym import error, spaces, utils
from gym.utils import seeding

from math import pi, cos, sin
import numpy as np

from . import gym_rendering
from .gym_utils import Interactive, WorldMap

class NavEnv(gym.Env):
    '''
    Navigate a rectangular space with a bottleneck in the middle, starting randomly on the left and the goal on the
    right. Bottleneck is closed off until the agent picks up a nearby key, after which it can cross over to the goal
    region.
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_state=(25.0, 25.0, 0.0)):
        self.min_x = 0.0
        self.max_x = 100.0
        self.min_y = 0.0
        self.max_y = 100.0
        self.radius = 1.0
        self.min_have_object = 0.0
        self.max_have_object = 1.0
        self.destination = np.array([85.0, 50.0])
        self.destination_tol = 2.0

        #TODO: Make map loadable via pickling

        obs1  = Interactive((55,45), (55,0), (45,0), (45,45), interactive=False, permanent=True)
        obs2  = Interactive((55,100), (55,55), (45,55), (45,100), interactive=False, permanent=True)
        block = Interactive((55,55), (55,45), (45,45), (45,55), interactive=False, permanent=False)
        key   = Interactive((15,15), (15,5), (5,5), (5,15), interactive=True, permanent=True)
        self.world = WorldMap(self.max_x, self.max_y, [obs1, obs2, block, key])

        self.low_state = np.array([self.min_x, self.min_y, self.min_have_object])
        self.high_state = np.array([self.max_x, self.max_x,
            self.max_have_object])

        self.init_state = np.array(initial_state)
        self.state = np.array(initial_state)

        self.min_speed = 0.0
        self.max_speed = 5.0
        self.min_angle = 0
        self.max_angle = 2*pi
        self.min_finish = 0.0
        self.max_finish = 1.0
        self.min_pickup = 0.0
        self.max_pickup = 1.0
        
        self.low_action = np.array([self.min_speed, self.min_angle,
            self.min_finish, self.min_pickup])
        self.high_action = np.array([self.max_speed, self.max_angle,
            self.max_finish, self.max_pickup])

        self.viewer = None

        self.action_space = spaces.Box(low=self.low_action,
                high=self.high_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state,
                high=self.high_state, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        old_state = self.state.copy()
        v = action[0]
        theta = action[1]
        dx = v*cos(theta)
        dy = v*sin(theta)

        pickup = (1-action[3] < 1e-6)
        attempt_finish = (1-action[4] < 1e-6)

        self.state += np.array([dx, dy, 0])
        reward = -1
        done = False
        info = {}

        if attempt_finish and (np.linalg.norm(self.destination-self.state) < self.destination_tol):
            reward = 20
            done = True

        if self.world.point_is_in_obstacle(self.state[0], self.state[1], epsilon=0.25):
            reward = -5 # for hitting an obstacle

        if self.world.segment_is_in_obstacle(old_state[0], old_state[1],
                                                   self.state[0], self.state[1],
                                                   epsilon=0.25):
            reward = -5 # for hitting an obstacle

        if reward != 5:
            if pickup and self.state[2] != 1 and self.world.point_is_in_keyzone(self.state[0], self.state[1]):
                reward = 0
                self.state[2] = 1
                self.world.remove_wall()

        self.observation = self._get_observation()
        return self.observation, reward, done, info

    def reset(self):
        obs = self._get_observation()
        self.state = self.init_state
        return obs

    def render(self, mode='human', close=False):
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return
    
        screen_width = 640
        screen_height = 640

        if self.viewer is None:
            self.viewer = gym_rendering.Viewer(screen_width, screen_height)
            self._append_elements_to_viewer(self.viewer, screen_width,
                    screen_height, objects=self.world.objects,
                    destination=self.destination,
                    destination_tol=self.destination_tol)

        self._plot_state(self.viewer, self.state)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _plot_state(self, viewer, state):
        polygon = gym_rendering.make_circle(radius=self.radius, res=30, filled=True)
        state_tr = gym_rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(state_tr)
        if state[2] == 1.0:
            polygon.set_color(0.0, 1.0, 0.0)
        else:
            polygon.set_color(0.0, 0.0, 1.0)

        viewer.add_onetime(polygon)

    def _append_elements_to_viewer(self, viewer, width, height,
            objects=None, destination=None, destination_tol=None):

        viewer.set_bounds(left=-100, right=width+100, bottom=-100,
                top=height+100)

        if not (objects is None):
            for i, ob in enumerate(objects):
                if self.state[2] == 1 and (ob.permanent and ob.interactive):     # only add key if not yet picked up
                    continue

                rect = gym_rendering.FilledPolygon([ob.top_right, ob.bot_right, ob.bot_left, ob.top_left])
                tr = gym_rendering.Transform(translation=(ob.c[0], ob.c[1]))
                rect.add_attr(tr)
                if ob.permanent and not ob.interactive:     # if object
                    rect.set_color(0.8, 0.6, 0.4)
                elif not ob.permanent and ob.interactive:   # if movable wall
                    rect.set_color(0.8, 0.8, 0.0)
                elif ob.permanent and ob.interactive:       # if key
                    rect.set_color(0.0, 0.5, 0.0)

                viewer.add_geom(rect)

        if not (destination is None):
            tr = gym_rendering.Transform(translation=(destination[0],
                destination[1]))
            polygon = gym_rendering.make_circle(radius=destination_tol, res=30,
                    filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)

    def _get_observation(self):
        
        interactives_loc = self.world.location_of_interactives() if self.state[2] == 0 else self.state[0:2]
        dist_to_closest_obs, angl_to_closest_obs = self.world.range_and_bearing_to_closest_obstacle(self.state[0], self.state[1])
        obs = np.array([self.state[0], self.state[1], self.state[2], dist_to_closest_obs, angl_to_closest_obs, self.destination[0],
            self.destination[1], interactives_loc[0], interactives_loc[1]])

        return obs

