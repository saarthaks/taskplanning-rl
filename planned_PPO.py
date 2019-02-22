import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.a2c.utils import total_episode_reward_logger

from stable_baselines.ppo2 import PPO2
from . import LatentSkillsPolicy

class PlannedPPO(PPO2):
    def __init__(self, policy, env, z_dim, traj_len, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):
        super(PlannedPPO, self).__init__(policy, env, gamma, n_steps, ent_coef, learning_rate, vf_coef, max_grad_norm,
                                         lam, nminibatches, noptepochs, cliprange, verbose, tensorboard_log,
                                         policy_kwargs, full_tensorboard_log, _init_setup_model=False)
        self.z_dim = z_dim
        self.traj_len = traj_len

        if _init_setup_model:
            self.setup_entire_model()

    #TODO
    def setup_entire_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, LatentSkillsPolicy), "Error: the input policy for the PlannedPPO model " \
                                                                "must be an instance of LatentSkillsPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.z_dim, self.traj_len,
                                        self.n_envs, 1, n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.z_dim,
                                              self.traj_len, self.n_envs // self.nminibatches, self.n_steps,
                                              n_batch_train, reuse=True, **self.policy_kwargs)
                #TODO
                with tf.variable_scope("loss", reuse=False):
                    pass

                #TODO
                with tf.variable_scope("model"):
                    pass

                #TODO: Huh??
                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))
            trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
            self._train = trainer.apply_gradients(grads)
            #TODO: MORE...

            self.train_model = train_model
            self.act_model = act_model
            self.step = act_model.step
            self.proba_step = act_model.proba_step
            self.value = act_model.value
            self.initial_state = act_model.initial_state
            tf.global_variables_initializer().run(session=self.sess)

            self.summary = tf.summary.merge_all()

        return NotImplementedError

    #TODO
    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update, writer,
                    states=None):
        return NotImplementedError

    #TODO
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PlannedPPO",
              reset_num_timesteps=True):
        return NotImplementedError

    #TODO: Check if any more needs saving
    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "z_dim": self.z_dim,
            "traj_len": self.traj_len,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)

#TODO: Edit this...
class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model
        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        """
        Run a learning step of the model
        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward