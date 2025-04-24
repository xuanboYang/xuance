import os
import oneflow as flow
import numpy as np
from xuance.common import Sequence, Optional, Callable, Union
from copy import deepcopy
from gym.spaces import Box
from xuance.oneflow import Module, Tensor, DistributedDataParallel
from xuance.oneflow.utils import ModuleType
from .core import GaussianActorNet as ActorNet
from .core import CriticNet, GaussianActorNet_SAC


class ActorPolicy(Module):
    """
    Actor for stochastic policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 fixed_std: bool = True):
        super(ActorPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.action_dim = action_space.shape[0]
        self.actor_hidden_size = actor_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        self.device = device
        self.activation_action = activation_action
        self.fixed_std = fixed_std

        self.actor = ActorNet(self.representation_info_shape['state'][0], self.action_dim, self.actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        return outputs, act_dist


class ActorCriticPolicy(Module):
    """
    Actor-critic policy with Gaussian distributions (AC for continuous actions spaces)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(ActorCriticPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.obs_dim = self.representation_info_shape['state'][0]
        self.action_dim = action_space.shape[0]
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        if critic_hidden_size is None:
            self.critic_hidden_size = [256, 256]
        self.device = device

        self.actor = ActorNet(self.obs_dim, self.action_dim, self.actor_hidden_size,
                              normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(self.obs_dim, self.critic_hidden_size,
                                normalize, initialize, activation, device)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        return outputs, act_dist, value


class PPGActorCritic(Module):
    """
    PPG policy with actor-critic structure.

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(PPGActorCritic, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.obs_dim = self.representation_info_shape['state'][0]
        self.action_dim = action_space.shape[0]
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        if actor_hidden_size is None:
            self.actor_hidden_size = [256, 256]
        if critic_hidden_size is None:
            self.critic_hidden_size = [256, 256]
        self.device = device

        self.actor = ActorNet(self.obs_dim, self.action_dim, self.actor_hidden_size,
                             normalize, initialize, activation, activation_action, device)
        self.critic = CriticNet(self.obs_dim, self.critic_hidden_size,
                               normalize, initialize, activation, device)
        self.aux_critic = CriticNet(self.obs_dim, self.critic_hidden_size,
                                   normalize, initialize, activation, device)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.aux_critic = DistributedDataParallel(self.aux_critic, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        value = self.critic(outputs['state'])
        aux_value = self.aux_critic(outputs['state'])
        return outputs, act_dist, value, aux_value


class SACPolicy(Module):
    """
    SAC policy with Gaussian distributions. (Continuous action space)

    Args:
        action_space (Box): The continuous action space.
        representation (Module): The representation module.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Box,
                 representation: Module,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False):
        super(SACPolicy, self).__init__()
        self.action_space = action_space
        self.representation = representation
        self.representation_info_shape = representation.output_shapes
        self.obs_dim = self.representation_info_shape['state'][0]
        self.action_dim = action_space.shape[0]

        self.representation_target = deepcopy(representation)
        # actor
        self.actor = GaussianActorNet_SAC(self.obs_dim, self.action_dim, actor_hidden_size,
                                         normalize, initialize, activation, activation_action, device)
        # critic
        self.critic = []
        self.critic_target = []
        for _ in range(2):
            critic = CriticNet(self.obs_dim + self.action_dim, critic_hidden_size,
                              normalize, initialize, activation, device)
            self.critic.append(critic)
            self.critic_target.append(deepcopy(critic))
        self.critic = nn.ModuleList(self.critic)
        self.critic_target = nn.ModuleList(self.critic_target)

        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.critic_target = DistributedDataParallel(self.critic_target, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])
            self.representation_target = DistributedDataParallel(self.representation_target, device_ids=[device])

    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        return outputs, act_dist

    def Qpolicy(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        act_dist = self.actor(outputs['state'])
        act_sample, act_log = act_dist.rsample_and_logprob()
        act_log = act_log.sum(dim=-1, keepdim=True)
        eval_q = flow.cat([critic(flow.cat([outputs['state'], act_sample], dim=-1)) for critic in self.critic], dim=-1)
        eval_q = flow.min(eval_q, dim=-1, keepdim=True)[0]
        return outputs, act_dist, act_sample, eval_q, act_log

    def Qtarget(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation_target(observation)
        act_dist = self.actor(outputs['state'])
        act_sample, act_log = act_dist.rsample_and_logprob()
        act_log = act_log.sum(dim=-1, keepdim=True)
        target_q = flow.cat([critic(flow.cat([outputs['state'], act_sample], dim=-1)) for critic in self.critic_target], dim=-1)
        target_q = flow.min(target_q, dim=-1, keepdim=True)[0]
        return outputs, act_dist, act_sample, target_q, act_log

    def Qaction(self, observation: Union[np.ndarray, dict], action: Tensor):
        outputs = self.representation(observation)
        critic_input = flow.cat([outputs['state'], action], dim=-1)
        Qs = flow.cat([critic(critic_input) for critic in self.critic], dim=-1)
        return outputs, Qs

    def soft_update(self, tau=0.005):
        for target_param, source_param in zip(self.representation_target.parameters(), self.representation.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
        for i in range(len(self.critic_target)):
            for target_param, source_param in zip(self.critic_target[i].parameters(), self.critic[i].parameters()):
                target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
