import gym
import numpy as np
import tensorflow as tf

from itertools import zip_longest
from stable_baselines.a2c.utils import linear, lstm
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

def observation_skill_input(ob_space, z_dim, batch_size=None, name='Obz', scale=False):
    """
    Build observation input with encoding depending on the observation space type
    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.
    :param ob_space: (Gym Space) The observation space
    :param z_dim: (int) dimensionality of the skill-space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
    #TODO: refine z_low and z_high -- [0,1] currently
    input_shape = (ob_space.shape[0] + z_dim,)
    low_in = np.zeros(input_shape)
    low_in[:ob_space.shape[0]] = ob_space.low
    high_in = np.ones(input_shape)
    high_in[:ob_space.shape[0]] = ob_space.high

    input_ph = tf.placeholder(shape=(batch_size,) + input_shape, dtype=ob_space.dtype, name=name)
    processed_input = tf.to_float(input_ph)
    # rescale to [1, 0] if the bounds are defined
    if (scale and
       not np.any(np.isinf(low_in)) and not np.any(np.isinf(high_in)) and
       np.any((high_in - low_in) != 0)):

        # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
        processed_input = ((processed_input - low_in) / (high_in - low_in))
    return input_ph, processed_input


def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:
    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].
    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value

def trajectory_input(traj_size, batch_size=None):
    """
    Build trajectory input given trajectory size
    :param traj_size: (int) dimensionality of a flattened trajectory
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :return: (TensorFlow Tensor, TensorFlow Tensor) traj_placeholder, processed_traj_tensor
    """
    trajectory_ph = tf.placeholder(shape=(batch_size,) + (traj_size,), dtype=tf.float32, name="traj")
    processed_trajectory = tf.to_float(trajectory_ph)
    return trajectory_ph, processed_trajectory


def skill_input(z_dim, batch_size=None):
    """
    Build trajectory input given trajectory size
    :param traj_size: (int) dimensionality of a flattened trajectory
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param z_dim: (int) dimensionality of flattened skill embedding
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :return: (tf.Tensor, tf.Tensor) skill_placeholder, processed_skill_tensor
    """
    skill_ph = tf.placeholder(shape=(batch_size,) + (z_dim,), dtype=tf.float32, name="z_sample")
    processed_skill = tf.to_float(skill_ph)
    return skill_ph, processed_skill


def decoder(flat_skill, dec_arch, traj_size, act_fun):
    """
    Constructs an MLP that receives skill embeddings as an input and outputs a expected trajectory. The ``dec_arch``
    parameter allows to specify the amount and size of the hidden layers.
    :param flat_skill: (tf.Tensor) The skill to base trajectory decoding on.
    :param dec_arch: ([int]) The specification of the trajectory decoding network.
    :param traj_size: (int) The dimensionality of the trajectory-space
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor) expected_trajectory of the specified network
    """
    traj_hat = flat_skill
    for idx, layer_size in enumerate(dec_arch):
        traj_hat = act_fun(linear(traj_hat, "dec_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))

    traj_hat = act_fun(linear(traj_hat, "traj_hat", traj_size, init_scale=np.sqrt(2)))
    return traj_hat


def init_goal_vae(state_dim, z_dim, goal_dim, batch_size=None):
    goal_input_ph = tf.placeholder(shape=(batch_size, z_dim+state_dim), dtype=tf.float32, name="goal_input")
    processed_goal_input = tf.to_float(goal_input_ph)

    goal_ph = tf.placeholder(shape=(batch_size, goal_dim+z_dim), dtype=tf.float32, name="g_sample")
    processed_goal = tf.to_float(goal_ph)

    return goal_input_ph, processed_goal_input, goal_ph, processed_goal


def goal_encoder(flat_goal_input, enc_arch, goal_dim, act_fun):
    latent_goal = flat_goal_input
    for idx, layer_size in enumerate(enc_arch):
        latent_goal = act_fun(linear(latent_goal, "goal_enc_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))

    mn = act_fun(linear(latent_goal, "gmean", goal_dim, init_scale=np.sqrt(2)))
    sd = 0.5 * act_fun(linear(latent_goal, "gstd", goal_dim, init_scale=np.sqrt(2)))
    eps = tf.random_normal(shape=[tf.shape(latent_goal)[0], goal_dim], mean=0.0, stddev=1.0)
    g  = mn + tf.multiply(eps, tf.exp(sd))

    return g, mn, sd


def goal_decoder(flat_goal, dec_arch, state_dim, act_fun):
    initiation_hat = flat_goal
    for idx, layer_size in enumerate(dec_arch):
        initiation_hat = act_fun(linear(initiation_hat, "goal_dec_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))

    mn = act_fun(linear(initiation_hat, "initiation_mean", state_dim, init_scale=np.sqrt(2)))
    sd = 0.5 * act_fun(linear(initiation_hat, "initiation_std", state_dim, init_scale=np.sqrt(2)))
    eps = tf.random_normal(shape=[tf.shape(initiation_hat)[0], state_dim], mean=0.0, stddev=1.0)
    state_I_hat = mn + tf.multiple(eps, tf.exp(sd))

    return state_I_hat, mn, sd


def init_skill_vae(traj_size, state_dim, z_dim, enc_num=32, dec_num=64, batch_size=None):
    """
    Builds LSTM units for encoder and decoder of skill-VAE, as well as placeholders for their inputs.
    :param traj_size: (int) The dimensionality of the trajectory-space
    :param state_dim: (int) The dimensionality of the state-space
    :param z_dim: (int) The dimensionality of the skill-space
    :param enc_num: (int) The number of hidden units in the encoding LSTM
    :param dec_num: (int) The number of hidden units in the decoding LSTM
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :return: (tf.Tensor, tf.Tensor, LSTMCell, tf.Tensor, tf.Tensor, LSTMCell) placeholders and processed placeholders
             for first the encoder LSTM, and then for the decoder LSTM
    """
    trajectory_ph = tf.placeholder(shape=(batch_size,) + (traj_size, 1), dtype=tf.float32, name="traj")
    processed_trajectory = tf.to_float(trajectory_ph)

    state_skill_ph = tf.placeholder(shape=(batch_size,) + (state_dim+z_dim, 1), dtype=tf.float32, name="z_sample")
    processed_state_skill = tf.to_float(state_skill_ph)

    lstm_enc = tf.contrib.rnn.LSTMCell(enc_num)
    lstm_dec = tf.contrib.rnn.LSTMCell(dec_num)

    return trajectory_ph, processed_trajectory, lstm_enc, state_skill_ph, processed_state_skill, lstm_dec


def skill_encoder(lstm_enc, trajectory, enc_arch, z_dim, act_fun):
    """
    Constructs an unrolled LSTM that receives trajectories as an input and outputs a latent representation for the
    skill embedding network. The ``enc_arch`` parameter allows to specify the amount and size of the hidden layers for
    the MLPs describing the mean and std NNs.
    :param lstm_enc: (LSTMCell) The encoder LSTM cell
    :param enc_arch: ([int]) The specification of the skill embedding network.
    :param act_fun: (tf function) The activation function to use for the VAE networks.
    :param z_dim: (int) The dimensionality of the skill-space.
    :param act_fun: (tf function) The activation function to use for the VAE networks.
    :return: (tf.Tensor, tf.Tensor, tf.Tensor) latent_skill, mean, standard deviation of the specified network.
    """
    _, state = tf.contrib.rnn.dynamic_rnn(lstm_enc, trajectory)

    latent_skill = state.h      # has shape (batch_size, num_hidden_units)

    for idx, layer_size in enumerate(enc_arch):
        latent_skill = act_fun(linear(latent_skill, "skill_enc_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))

    mn = act_fun(linear(latent_skill, "zmean", z_dim, init_scale=np.sqrt(2)))
    sd = 0.5 * act_fun(linear(latent_skill, "zstd", z_dim, init_scale=np.sqrt(2)))
    eps = tf.random_normal(shape=[tf.shape(latent_skill)[0], z_dim], mean=0.0, stddev=1.0)
    z  = mn + tf.multiply(eps, tf.exp(sd))

    return z, mn, sd


def skill_decoder(lstm_dec, states_skill, dec_arch, traj_len, state_dim, act_fun, batch_size=None):
    """
    Constructs an unrolled LSTM that receives skill embeddings and an observed as an input and outputs a expected
    trajectory. The ``dec_arch`` parameter allows to specify the amount and size of the hidden layers. Note that this
    decoder does not perform teacher forcing, where output at time t is used as input for time t+1.
    :param lstm_dec: (LSTMCell) The decoder LSTM cell
    :param states_skill: (tf.Tensor) The skill and initial state to base trajectory decoding on.
    :param dec_arch: ([int]) The specification of the trajectory decoding network.
    :param traj_size:
    :param act_fun:
    :return:
    """
    #TODO: refine cell_init_input, reshaping output_ta, using raw_rnn

    cell_init_state = lstm_dec.zero_state(batch_size, tf.float32)
    cell_init_input = states_skill[tf.newaxis, :, :]
    output_ta = tf.TensorArray(size=traj_len, dtype=tf.float32)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0

        if cell_output is None:
            # time=0, everything here will be used for initialization only
            next_cell_state = cell_init_state
            next_input = cell_init_input
            next_loop_state = output_ta
        else:
            # pass the last state to the next
            next_cell_state = cell_state

            next_state = cell_output
            for idx, layer_size in enumerate(dec_arch):
                next_state = act_fun(linear(next_state, "skill_dec_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))

            next_input = act_fun(linear(next_state, "next_state_hat", state_dim, init_scale=np.sqrt(2)))
            next_loop_state = loop_state.write(time - 1, next_input)

        next_loop_state = None
        elements_finished = (time >= traj_len)

        return (elements_finished, next_input, next_cell_state,
            emit_output, next_loop_state)

    outputs, _ = tf.contrib.rnn.dynamic_rnn(lstm_dec, states_skill)

    return traj_hat

class LatentSkillsPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, st_space, ac_space, z_dim, goal_dim, traj_len, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(LatentSkillsPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                 feature_extraction="mlp", **_kwargs)

        self.z_dim = z_dim
        self.goal_dim = goal_dim
        self.state_dim = st_space.shape[0]
        self.traj_len = traj_len
        self.initial_state = None
        self.current_skill = None
        self.skill_count = 0

        # TODO: Set up VAE inputs, model, and outputs
        # sets up trajectory inputs for vae encoder network
        traj_size = traj_len * self.state_dim
        with tf.variable_scope("skill_vae_inputs", reuse=False):
            self.traj_ph, self.processed_traj, self.lstm_enc, self.state_skill_ph, self.processed_state_skill, self.lstm_dec = \
            init_skill_vae(traj_size, self.state_dim, self.z_dim)

        with tf.variable_scope("goal_cvae_inputs", reuse=False):
            self.goal_input_ph, self.processed_goal_input, self.goal_ph, self.processed_goal = \
            init_goal_vae(self.state_dim, self.z_dim, self.goal_dim)

        with tf.variable_scope("skill_vae_model", reuse=reuse):
            self.z_sample, self.z_mean, self.z_stddev = \
            skill_encoder(self.lstm_enc, tf.layers.flatten(self.processed_traj), self.enc_arch, self.z_dim, self.act_fun)

        with tf.variable_scope("goal_cvae_model", reuse=reuse):
            self.g_sample, self.g_mean, self.g_stddev = \
            goal_encoder(tf.layers.flatten(self.processed_goal_input), self.enc_arch, self.goal_dim, self.act_fun)

        with tf.variable_scope("skill_vae_outputs", reuse=True):
            self.traj_hat = skill_decoder(self.lstm_dec, tf.layers.flatten(self.processed_state_skill), self.dec_arch,
                                          self.traj_len,self.state_dim, self.act_fun)

        with tf.variable_scope("goal_cvae_outputs", reuse=True):
            self.state_I_hat, self.initiation_mn, self.initiation_stddev = \
            goal_decoder(tf.layers.flatten(self.processed_goal), self.dec_arch, self.state_dim, self.act_fun)

        # sets up observation space inputs for policy and value function networks
        with tf.variable_scope("obz_input", reuse=False):
            self.input_ph, self.processed_input = observation_skill_input(ob_space, z_dim, n_batch, scale=False)     # unscaled b/c mlp, not cnn

        # sets up shapes/models of policy and value function networks
        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_input), self.net_arch, self.act_fun)

            self.value_fn = linear(vf_latent, 'vf', 1)

            # self.policy here adds a linear layer on top of pi_latent, and similar with q_value on top of vf_latent
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        # sets up the outputs self.action, self.deterministic_action, and self.neglogp from self.proba_distribution
        # and from self.policy
        # TODO: May need to change outputs of policy networks?
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):

        if self.current_skill is None or self.skill_count == self.traj_len:
            self.current_skill = np.random.randn(self.z_dim)
            self.skill_count = 0

        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.input_ph: np.hstack([obs, self.current_skill])})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.input_ph: np.hstack([obs, self.current_skill])})

        self.skill_count += 1
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        #TODO
        return NotImplementedError

    def value(self, obs, state=None, mask=None):
        #TODO
        return NotImplementedError

