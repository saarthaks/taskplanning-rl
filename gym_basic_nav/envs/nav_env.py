import gym
from gym import error, spaces, utils
from gym.utils import seeding

from math import pi, cos, sin
import numpy as np

class NavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_state=[0.25,0.25,0]):
        self.min_x = 0.0
        self.max_x = 1.0
        self.min_y = 0.0
        self.max_y = 1.0
        self.min_have_object = 0.0
        self.max_have_object = 1.0
        self.destination = np.array([0.85, 0.5])
        self.destination_tol = 0.02

        #TODO: Define obstacles and objects in state-space
        self.low_state = np.array([self.min_x, self.min_y, self.min_have_object])
        self.high_state = np.array([self.max_x, self.max_x,
            self.max_have_object])

        self.init_state = initial_state
        self.state = np.array(initial_state)

        self.min_speed = 0.0
        self.max_speed = 0.05
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

        #TODO: implement picking up object and finishing
        pickup = (1-action[3] < 1e-6)
        attempt_finish = (1-action[4] < 1e-6)

        self.state += np.array([dx, dy])
        reward = -1
        done = False
        info = {}

        if (np.linalg.norm(self.destination-self.state) < self.destination_tol)
        and attempt_finish:
            reward = 20
            done = True

        #TODO: implement obstacle collision

        self.observation = self._get_observation(self.state)
        return self.observation, reward, done, info

    def reset(self):
        self.state = self.init_state
        return self._get_observation(self.state)

    def render(self, mode='human', close=False):
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return
    
        screen_width = 640
        screen_height = 640
        
        #TODO: implement gym_rendering
        if self.viewer is None:
            self.viewer = gym_rendering.Viewer(screen_width, screen_height)
            self._append_elements_to_viewer(self.viewer, screen_width,
                    screen_height, obstacles=None,
                    destination=self.destination,
                    destination_tol=self.destination_tol)

        self._plot_state(self.viewer, self.state)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _plot_state(self, viewer, state):
        polygon = gym_rendering.make_circle(radius=0.01, res=30, filled=True)
        state_tr = gym_rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(state_tr)
        viewer.add_onetime(polygon)

    def append_elements_to_viewer(self, viewer, width, height,
            obstacles=None, destination=None, destination_tol=None):

        viewer.set_bounds(left=-100, right=width+100, bottom=-100,
                top=height+100)

        #TODO: add obstacles

        if not (destination is None):
            tr = gym_rendering.Transform(translation=(destination[0],
                destination[1]))
            polygon = gym_rendering.make_circle(radius=destination_tol, res=30,
                    filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)

    def _get_observation(self, state):
        
        #TODO: include distance and angle to closest obstacle 
        obs = np.array([state[0], state[1], self.destination[0],
            self.destination[1]])

        return obs

