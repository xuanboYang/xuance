import os
import oneflow as flow
import oneflow.nn as nn
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete
from xuance.common import Sequence, Optional, Callable, Union
from xuance.oneflow import Module, Tensor, DistributedDataParallel
from xuance.oneflow.utils import ModuleType
from .core import CategoricalActorNet as ActorNet
from .core import CategoricalActorNet_SAC as Actor_SAC
from .core import BasicQhead, CriticNet


def _init_layer(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorPolicy(Module):
    """
    Actor for stochastic policy with categorical distributions. (Discrete action space)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(ActorPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.action_dim = action_space.n
        self.actor_hidden_size = actor_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        self.device = device

        self.actor = ActorNet(self.representation_info_shape['state'][0], self.action_dim, self.actor_hidden_size,
                              normalize, initialize, activation, device)
        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict], avail_actions=None):
        """
        Returns the output of the actor network.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.
            avail_actions: The actions mask.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
        """
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'], avail_actions)
        return outputs, act_dist


class ActorCriticPolicy(Module):
    """
    Actor-critic policy with categorical distributions. (AC for discrete action spaces)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.action_dim = action_space.n
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        if critic_hidden_size is None:
            self.critic_hidden_size = [256, 256]
        self.device = device

        self.actor = ActorNet(self.representation_info_shape['state'][0], self.action_dim, self.actor_hidden_size,
                              normalize, initialize, activation, device)
        self.critic = CriticNet(self.representation_info_shape['state'][0], self.critic_hidden_size,
                                normalize, initialize, activation, device)
        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict], avail_actions=None):
        """
        Returns the outputs of the actor network.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.
            avail_actions: The actions mask.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
            value: The value of the critic network.
        """
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'], avail_actions)
        value = self.critic(outputs['state'])
        return outputs, act_dist, value


class PPGActorCritic(Module):
    """
    PPG policy with actor-critic structure. (Discrete action spaces)

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.action_dim = action_space.n
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        if critic_hidden_size is None:
            self.critic_hidden_size = [256, 256]
        self.device = device

        self.actor = ActorNet(self.representation_info_shape['state'][0], self.action_dim, self.actor_hidden_size,
                             normalize, initialize, activation, device)
        self.critic = CriticNet(self.representation_info_shape['state'][0], self.critic_hidden_size,
                               normalize, initialize, activation, device)
        self.aux_critic = CriticNet(self.representation_info_shape['state'][0], self.critic_hidden_size,
                                   normalize, initialize, activation, device)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.aux_critic = DistributedDataParallel(self.aux_critic, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict], avail_actions=None):
        """
        Returns the outputs of the actor network.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.
            avail_actions: The actions mask.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
            value: The value of the critic network.
            aux_value: The value of the auxiliary critic network.
        """
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'], avail_actions)
        value = self.critic(outputs['state'])
        aux_value = self.aux_critic(outputs['state'])
        return outputs, act_dist, value, aux_value


class SACDISPolicy(Module):
    """
    SAC policy for discrete action spaces, which outputs a distribution over the discrete action space.

    Args:
        action_space (Discrete): The discrete action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(SACDISPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.obs_dim = self.representation_info_shape['state'][0]
        self.action_dim = action_space.n

        self.representation_target = deepcopy(representation)
        # actor
        self.actor = Actor_SAC(self.obs_dim, self.action_dim, actor_hidden_size,
                               normalize, initialize, activation, device)
        # critic 1
        self.critic_1 = BasicQhead(self.obs_dim, self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation, device)
        self.critic_target_1 = deepcopy(self.critic_1)
        # critic 2
        self.critic_2 = BasicQhead(self.obs_dim, self.action_dim, critic_hidden_size,
                                   normalize, initialize, activation, device)
        self.critic_target_2 = deepcopy(self.critic_2)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic_1 = DistributedDataParallel(self.critic_1, device_ids=[device])
            self.critic_2 = DistributedDataParallel(self.critic_2, device_ids=[device])
            self.critic_target_1 = DistributedDataParallel(self.critic_target_1, device_ids=[device])
            self.critic_target_2 = DistributedDataParallel(self.critic_target_2, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])
            self.representation_target = DistributedDataParallel(self.representation_target, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the outputs of the actor network.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
        """
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        return outputs, act_dist

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        """
        Returns the Q values of all actions.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
            action_q: The Q values of all actions.
            log_pi: The log probabilities of all actions.
            actions: The sampled actions.
        """
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        actions = act_dist.stochastic_sample()
        log_pi = act_dist.log_prob(actions)
        log_pi = log_pi.reshape(log_pi.shape[0], -1)
        probs = act_dist.get_param()["probs"]

        q1_values = self.critic_1(outputs['state'])
        q2_values = self.critic_2(outputs['state'])
        # action_q = flow.min(q1_values, q2_values)
        action_q = flow.cat([q1_values.unsqueeze(0), q2_values.unsqueeze(0)], dim=0)

        return outputs, act_dist, action_q, log_pi, actions, probs

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        """
        Returns the Q values of all actions.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.

        Returns:
            outputs: The output of the representation module.
            act_dist: The categorical distribution of the actions.
            action_q: The Q values of all actions.
            log_pi: The log probabilities of all actions.
            actions: The sampled actions.
        """
        outputs = self.representation_target(observation)
        act_dist = self.actor(outputs['state'])
        actions = act_dist.stochastic_sample()
        log_pi = act_dist.log_prob(actions)
        log_pi = log_pi.reshape(log_pi.shape[0], -1)
        probs = act_dist.get_param()["probs"]

        q1_target = self.critic_target_1(outputs['state'])
        q2_target = self.critic_target_2(outputs['state'])
        action_q = flow.cat([q1_target.unsqueeze(0), q2_target.unsqueeze(0)], dim=0)

        return outputs, act_dist, action_q, log_pi, actions, probs

    def Qaction(self, observation: Union[np.ndarray, dict]):
        """
        Returns the Q values of all actions.
        Parameters:
            observation (Union[np.ndarray, dict]): The input observations.

        Returns:
            outputs: The output of the representation module.
            q1_values: The Q values of all actions from the first critic.
            q2_values: The Q values of all actions from the second critic.
        """
        outputs = self.representation(observation)
        q1_values = self.critic_1(outputs['state'])
        q2_values = self.critic_2(outputs['state'])
        return outputs, q1_values, q2_values

    def soft_update(self, tau=0.005):
        """
        Soft updating of the target networks.
        """
        for target_param, source_param in zip(self.representation_target.parameters(), self.representation.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
        for target_param, source_param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
        for target_param, source_param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
