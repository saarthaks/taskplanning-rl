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

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.z_dim, self.goal_dim,
                                        self.traj_len, self.n_envs, 1, n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.z_dim, self.goal_dim,
                                              self.traj_len, self.n_envs // self.nminibatches, self.n_steps,
                                              n_batch_train, reuse=True, **self.policy_kwargs)
                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")
                    #TODO: add placeholders for vae and cvae loss functions
                    self.trajectory_ph = tf.placeholder(tf.float32, [None], name="trajectory_ph")
                    self.trajectory_hat_ph = tf.placeholder(tf.float32, [None], name="trajectory_hat_ph")
                    self.z_mean_ph = tf.placeholder(tf.float32, [None], name="z_mean_ph")
                    self.z_stddev_ph = tf.placeholder(tf.float32, [None], "z_stddev_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model._value
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model._value - self.old_vpred_ph, -self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph)))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    #TODO: add loss terms from vae and cvae to loss expression above
                    traj_loss = tf.reduce_sum(tf.squared_difference(self.trajectory_ph, self.trajectory_hat_ph), 1)
                    latent_skill_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.z_stddev_ph - tf.square(self.z_mean_ph) - \
                                                             tf.exp(2.0 * self.z_stddev_ph), 1)
                    self.skill_vae_loss = tf.reduce_mean(traj_loss + latent_skill_loss)
                    loss += self.skill_vae_loss

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leiber', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                #TODO: confirm all trainable variables are captured in scope "model"
                with tf.variable_scope("model"):
                    self.params = tf.trainable_variables()
                    if self.full_tensorboard_log:
                        for var in self.params:
                            tf.summary.histogram(var.name, var)
                grads = tf.gradients(loss, self.params)
                if self.max_grad_norm is not None:
                    grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, self.params))

            trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
            self._train = trainer.apply_gradients(grads)

            #TODO: include vae and cvae losses
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

            with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

            self.train_model = train_model
            self.act_model = act_model
            self.step = act_model.step
            self.proba_step = act_model.proba_step
            self.value = act_model.value
            self.initial_state = act_model.initial_state
            tf.global_variables_initializer().run(session=self.sess)
            #TODO: consider saving other quantities to "self"

            self.summary = tf.summary.merge_all()

    #TODO
    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update, writer,
                    states=None):
        return NotImplementedError

    #TODO
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="PlannedPPO",
              reset_num_timesteps=True):
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
            as writer:

            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            nupdates = total_timesteps // self.n_batch
            for update in range(1, nupdates + 1):
                assert self.n_batch % self.nminibatches == 0
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / nupdates
                lr_now = self.learning_rate(frac)
                cliprangenow = self.cliprange(frac)
                # true_reward is reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()

        return NotImplementedError

    #TODO
    def plan(self):
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

            new_state = self.env.state

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


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.
    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, float):
        value_schedule = constfn(value_schedule)
    else:
        assert callable(value_schedule)
    return value_schedule


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1
    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])