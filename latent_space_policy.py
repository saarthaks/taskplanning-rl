import gym
import numpy as np
import tensorflow as tf

from itertools import zip_longest
from stable_baselines.a2c.utils import linear
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

def observation_input(ob_space, z_dim, batch_size=None, name='Ob', scale=False):
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
    #TODO: add z_dim to observation input
    observation_ph = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
    processed_observations = tf.to_float(observation_ph)
    # rescale to [1, 0] if the bounds are defined
    if (scale and
       not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
       np.any((ob_space.high - ob_space.low) != 0)):

        # equivalent to processed_observations / 255.0 when bounds are set to [255, 0]
        processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations

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

class LatentSkillsPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, z_dim, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(LatentSkillsPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                 feature_extraction="mlp", **_kwargs)

        # sets up observation space inputs for policy and value function networks
        with tf.variable_scope("input", reuse=False):
            self.obs_ph, self.processed_obs = observation_input(ob_space, z_dim, n_batch, scale=False)     # unscaled b/c mlp, not cnn

        # sets up shapes/models of policy and value function networks
        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), self.net_arch, self.act_fun)

            self.value_fn = linear(vf_latent, 'vf', 1)

            # self.policy here adds a linear layer on top of pi_latent, and similar with q_value on top of vf_latent
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.initial_state = None
        # sets up the outputs self.action, self.deterministic_action, and self.neglogp from self.proba_distribution
        # and from self.policy
        # TODO: May need to change outputs of policy networks?
        self._setup_init()

        # TODO: Set up VAE inputs, model, and outputs

    def step(self, obs, state=None, mask=None, determinstic=False):
        #TODO
        return NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        #TODO
        return NotImplementedError

    def value(self, obs, state=None, mask=None):
        #TODO
        return NotImplementedError

