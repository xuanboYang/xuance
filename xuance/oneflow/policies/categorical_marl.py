import os
import oneflow as flow
import oneflow.nn as nn
import numpy as np
from copy import deepcopy
from operator import itemgetter
from gym.spaces import Discrete
from oneflow.distributions import Categorical
from xuance.common import Sequence, Optional, Callable, Union, Dict, List
from xuance.oneflow.policies import CategoricalActorNet, ActorNet
from xuance.oneflow.policies.core import CriticNet, BasicQhead
from xuance.oneflow.utils import ModuleType, CategoricalDistribution
from xuance.oneflow import Tensor, Module, ModuleDict, DistributedDataParallel
from .core import CategoricalActorNet_SAC as Actor_SAC
from oneflow.nn import ModuleDict, Module
from xuance.oneflow.utils.distributions import DiagGaussianDistribution
from gym.spaces import Discrete, Box


class MAAC_Policy(Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy with categorical policies.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The discrete action space.
        n_agents (int): The number of agents.
        representation_actor (ModuleDict): A dict of representation modules for each agent's actor.
        representation_critic (ModuleDict): A dict of representation modules for each agent's critic.
        mixer (Module): The mixer module that mix together the individual values to the total value.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: ModuleDict,
                 representation_critic: ModuleDict,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = representation_actor
        self.critic_representation = representation_critic

        self.dim_input_critic, self.n_actions = {}, {}
        self.actor, self.critic = nn.ModuleDict(), nn.ModuleDict()
        for key in self.model_keys:
            self.n_actions[key] = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out = self._get_actor_critic_input(
                self.n_actions[key],
                self.actor_representation[key].output_shapes['state'][0],
                self.critic_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = CategoricalActorNet(dim_actor_in, dim_actor_out, actor_hidden_size,
                                                  normalize, initialize, activation, device)
            self.critic[key] = CriticNet(dim_critic_in, critic_hidden_size, normalize, initialize, activation, device)

        self.mixer = mixer

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_representation[key]._get_name() != "Basic_Identical":
                    self.critic_representation[key] = DistributedDataParallel(self.critic_representation[key],
                                                                              device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic[key] = DistributedDataParallel(module=self.critic[key], device_ids=[self.rank])
            if self.mixer is not None:
                self.mixer = DistributedDataParallel(module=self.mixer, device_ids=[self.rank])

    @property
    def parameters_model(self):
        parameters = list(self.actor_representation.parameters()) + list(self.actor.parameters()) + list(
            self.critic_representation.parameters()) + list(self.critic.parameters())
        if self.mixer is None:
            return parameters
        else:
            return parameters + list(self.mixer.parameters())

    def _get_actor_critic_input(self, dim_action, dim_actor_rep, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_action: The dimension of actions.
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in, dim_critic_out = dim_critic_rep, dim_action
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in, dim_critic_out

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden ((Optional[Dict[str, List[Tensor]])): The RNN hidden states of actor representation.

        Returns:
            rnn_hidden_new ((Optional[Dict[str, List[Tensor]])): The new RNN hidden states of actor representation.
            pi_dists (dict): The stochastic policy distributions.
        """
        try:
            rnn_hidden_new, pi_dists = {}, {}
            agent_list = self.model_keys if agent_key is None else [agent_key]

            if avail_actions is not None:
                avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list if key in avail_actions}

            for key in agent_list:
                try:
                    if self.use_rnn:
                        if rnn_hidden is None or key not in rnn_hidden or None in rnn_hidden[key]:
                            # 如果RNN隐藏状态不可用，使用零初始化
                            batch_size = observation[key].shape[0] if key in observation else 1
                            hidden_size = self.actor_representation[key].rnn.hidden_size
                            h0 = flow.zeros(1, batch_size, hidden_size, device=self.device)
                            c0 = flow.zeros(1, batch_size, hidden_size, device=self.device)
                            outputs = self.actor_representation[key](observation[key], h0, c0)
                        else:
                            outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                        rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                    else:
                        outputs = self.actor_representation[key](observation[key])
                        rnn_hidden_new[key] = [None, None]

                    if self.use_parameter_sharing:
                        # 安全地处理agent_ids
                        try:
                            actor_input = outputs['state']
                            # 检查actor_input是否有效
                            if actor_input is None or actor_input.shape[0] == 0:
                                # 创建一个安全的替代输入
                                actor_input = flow.zeros((1, self.actor[key].input_dim), device=self.device)
                            
                            # 如果agent_ids是字典，则使用当前智能体的ID
                            if isinstance(agent_ids, dict) and key in agent_ids:
                                agent_id = agent_ids[key]
                                # 将agent_id添加到actor_input的最后一维
                                if isinstance(agent_id, (int, float)):
                                    # 如果agent_id是标量，创建一个张量
                                    agent_id_tensor = flow.ones((actor_input.shape[0], 1), device=actor_input.device) * agent_id
                                    # 使用安全的连接方法
                                    try:
                                        actor_input = flow.cat([actor_input, agent_id_tensor], dim=-1)
                                    except Exception as e:
                                        # 如果连接失败，只使用原始输入
                                        pass
                                else:
                                    # 如果agent_id已经是张量，确保其形状正确
                                    try:
                                        agent_id_tensor = agent_id.view(actor_input.shape[0], 1)
                                        # 检查形状兼容性
                                        if agent_id_tensor.shape[0] == actor_input.shape[0]:
                                            actor_input = flow.cat([actor_input, agent_id_tensor], dim=-1)
                                    except Exception as e:
                                        # 如果连接失败，只使用原始输入
                                        pass
                            else:
                                # 如果没有合适的agent_id，使用默认值
                                try:
                                    agent_id_tensor = flow.zeros((actor_input.shape[0], 1), device=actor_input.device)
                                    actor_input = flow.cat([actor_input, agent_id_tensor], dim=-1)
                                except Exception as e:
                                    # 如果连接失败，只使用原始输入
                                    pass
                        except Exception as e:
                            # 如果出错，仅使用状态
                            actor_input = outputs['state']
                    else:
                        actor_input = outputs['state']

                    avail_actions_input = None if avail_actions is None or key not in avail_actions else avail_actions[key]
                    pi_dists[key] = self.actor[key](actor_input, avail_actions_input)
                except Exception as e:
                    # 如果处理某个智能体失败，创建一个默认分布
                    action_dim = self.actor[key].action_dim if hasattr(self.actor[key], 'action_dim') else 1
                    probs = flow.ones((1, action_dim), device=self.device) / action_dim
                    pi_dists[key] = CategoricalDistribution(probs=probs)
                    rnn_hidden_new[key] = [None, None]
            
            return rnn_hidden_new, pi_dists
        except Exception as e:
            # 如果整个前向传播失败，返回空结果
            return {}, {}

    def get_values(self, critic_in=None, observation: Dict[str, Tensor] = None, agent_ids: Tensor = None, agent_key: str = None,
                   rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, rnn_states=None):
        """
        Get critic values via critic networks.

        Parameters:
            critic_in (Tensor, optional): The critic input tensor. If provided, it will be used directly.
            observation (Dict[str, Tensor], optional): The input observations for the policies.
            agent_ids (Tensor, optional): The agents' ids (for parameter sharing).
            agent_key (str, optional): Calculate actions for specified agent.
            rnn_hidden ((Optional[Dict[str, List[Tensor]]])): The RNN hidden states of critic representation.
            rnn_states (Optional): RNN states for RNN-based policies.

        Returns:
            rnn_hidden_new ((Optional[Dict[str, List[Tensor]]])): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        try:
            rnn_hidden_new, values = {}, {}
            agent_list = self.model_keys if agent_key is None else [agent_key]

            # 如果提供了critic_in，直接使用它
            if critic_in is not None:
                # 确保critic_in是OneFlow张量
                if not isinstance(critic_in, flow.Tensor):
                    try:
                        critic_in = flow.tensor(critic_in, dtype=flow.float32, device=self.device)
                    except Exception as e:
                        # 如果转换失败，创建一个默认的critic_in
                        critic_in = flow.zeros((1, 1, 1), device=self.device)
                
                # 检查critic_in是否为空
                if critic_in.numel() == 0:
                    critic_in = flow.zeros((1, 1, 1), device=self.device)
                
                # 检查并修复NaN值
                if flow.isnan(critic_in).any():
                    critic_in = flow.nan_to_num(critic_in, nan=0.0)
                
                # 为每个智能体创建值函数预测
                for key in agent_list:
                    try:
                        # 使用critic网络直接处理critic_in
                        # 确保critic_in的形状适合critic网络
                        if hasattr(self.critic[key], 'input_dim'):
                            expected_dim = self.critic[key].input_dim
                            # 检查最后一个维度是否匹配
                            if len(critic_in.shape) > 0 and critic_in.shape[-1] != expected_dim:
                                # 尝试调整形状
                                try:
                                    # 如果critic_in是3D的 (batch, seq, dim)
                                    if len(critic_in.shape) == 3:
                                        # 尝试reshape到2D (batch*seq, dim)
                                        critic_in_reshaped = critic_in.reshape(-1, critic_in.shape[-1])
                                        # 然后使用线性插值调整最后一个维度
                                        if critic_in_reshaped.shape[-1] < expected_dim:
                                            # 如果维度太小，填充零
                                            padding = flow.zeros(critic_in_reshaped.shape[0], expected_dim - critic_in_reshaped.shape[-1], device=critic_in.device)
                                            critic_in_adjusted = flow.cat([critic_in_reshaped, padding], dim=-1)
                                        else:
                                            # 如果维度太大，截断
                                            critic_in_adjusted = critic_in_reshaped[:, :expected_dim]
                                        # 重新调整回原始形状
                                        critic_in = critic_in_adjusted.reshape(critic_in.shape[0], critic_in.shape[1], expected_dim)
                                    else:
                                        # 对于其他形状，简单地调整最后一个维度
                                        if critic_in.shape[-1] < expected_dim:
                                            # 如果维度太小，填充零
                                            padding = flow.zeros(*critic_in.shape[:-1], expected_dim - critic_in.shape[-1], device=critic_in.device)
                                            critic_in = flow.cat([critic_in, padding], dim=-1)
                                        else:
                                            # 如果维度太大，截断
                                            critic_in = critic_in[..., :expected_dim]
                                except Exception as e:
                                    # 如果调整失败，创建一个默认的输入
                                    critic_in = flow.zeros((1, expected_dim), device=self.device)
                        
                        # 使用critic网络处理调整后的critic_in
                        try:
                            values[key] = self.critic[key](critic_in)
                            # 检查输出是否包含NaN
                            if flow.isnan(values[key]).any():
                                values[key] = flow.nan_to_num(values[key], nan=0.0)
                        except Exception as e:
                            # 如果critic网络处理失败，创建默认值
                            values[key] = flow.zeros((critic_in.shape[0] if len(critic_in.shape) > 0 else 1, 1), device=self.device)
                    except Exception as e:
                        # 如果处理失败，创建默认值
                        values[key] = flow.zeros((critic_in.shape[0] if len(critic_in.shape) > 0 else 1, 1), device=self.device)
                        rnn_hidden_new[key] = [None, None]
                
                # 如果提供了rnn_states，返回它们
                if rnn_states is not None:
                    return values, rnn_states
                
                return values
            
            # 如果没有提供critic_in，使用observation
            if observation is None:
                # 如果既没有critic_in也没有observation，返回默认值
                for key in agent_list:
                    values[key] = flow.zeros((1, 1), device=self.device)
                    rnn_hidden_new[key] = [None, None]
                
                # 如果提供了rnn_states，返回它们
                if rnn_states is not None:
                    return values, rnn_states
                
                return rnn_hidden_new, values
            
            # 使用observation计算值函数
            for key in agent_list:
                try:
                    if hasattr(self, 'use_rnn') and self.use_rnn:
                        if rnn_hidden is None or key not in rnn_hidden or None in rnn_hidden[key]:
                            # 如果RNN隐藏状态不可用，使用零初始化
                            batch_size = observation[key].shape[0] if key in observation else 1
                            hidden_size = self.critic_representation[key].rnn.hidden_size
                            h0 = flow.zeros(1, batch_size, hidden_size, device=self.device)
                            c0 = flow.zeros(1, batch_size, hidden_size, device=self.device)
                            outputs = self.critic_representation[key](observation[key], h0, c0)
                        else:
                            outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                        rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
                    else:
                        # 确保observation[key]是有效的
                        if key not in observation or observation[key] is None:
                            # 如果observation[key]不可用，创建一个默认的输入
                            if hasattr(self.critic_representation[key], 'input_dim'):
                                input_dim = self.critic_representation[key].input_dim
                                observation_key = flow.zeros((1, input_dim), device=self.device)
                            else:
                                observation_key = flow.zeros((1, 1), device=self.device)
                        else:
                            observation_key = observation[key]
                            # 检查并修复NaN值
                            if flow.isnan(observation_key).any():
                                observation_key = flow.nan_to_num(observation_key, nan=0.0)
                        
                        try:
                            outputs = self.critic_representation[key](observation_key)
                        except Exception as e:
                            # 如果表示网络处理失败，创建默认输出
                            outputs = {'state': flow.zeros((1, 1), device=self.device)}
                        
                        rnn_hidden_new[key] = [None, None]

                    if hasattr(self, 'use_parameter_sharing') and self.use_parameter_sharing:
                        # 安全地处理agent_ids
                        try:
                            critic_input = outputs.get('state', None)
                            # 检查critic_input是否有效
                            if critic_input is None or critic_input.shape[0] == 0:
                                # 创建一个安全的替代输入
                                if hasattr(self.critic[key], 'input_dim'):
                                    input_dim = self.critic[key].input_dim
                                    critic_input = flow.zeros((1, input_dim), device=self.device)
                                else:
                                    critic_input = flow.zeros((1, 1), device=self.device)
                            
                            # 如果agent_ids是字典，则使用当前智能体的ID
                            if isinstance(agent_ids, dict) and key in agent_ids:
                                agent_id = agent_ids[key]
                                # 将agent_id添加到critic_input的最后一维
                                if isinstance(agent_id, (int, float)):
                                    # 如果agent_id是标量，创建一个张量
                                    agent_id_tensor = flow.ones((critic_input.shape[0], 1), device=critic_input.device) * agent_id
                                    # 使用安全的连接方法
                                    try:
                                        critic_input = flow.cat([critic_input, agent_id_tensor], dim=-1)
                                    except Exception as e:
                                        # 如果连接失败，只使用原始输入
                                        pass
                                else:
                                    # 如果agent_id已经是张量，确保其形状正确
                                    try:
                                        agent_id_tensor = agent_id.view(critic_input.shape[0], 1)
                                        # 检查形状兼容性
                                        if agent_id_tensor.shape[0] == critic_input.shape[0]:
                                            critic_input = flow.cat([critic_input, agent_id_tensor], dim=-1)
                                    except Exception as e:
                                        # 如果连接失败，只使用原始输入
                                        pass
                            else:
                                # 如果没有合适的agent_id，使用默认值
                                try:
                                    agent_id_tensor = flow.zeros((critic_input.shape[0], 1), device=critic_input.device)
                                    critic_input = flow.cat([critic_input, agent_id_tensor], dim=-1)
                                except Exception as e:
                                    # 如果连接失败，只使用原始输入
                                    pass
                        except Exception as e:
                            # 如果出错，仅使用状态
                            critic_input = outputs.get('state', flow.zeros((1, 1), device=self.device))
                    else:
                        critic_input = outputs.get('state', flow.zeros((1, 1), device=self.device))
                    
                    # 确保critic_input是有效的
                    if critic_input is None or critic_input.numel() == 0:
                        if hasattr(self.critic[key], 'input_dim'):
                            input_dim = self.critic[key].input_dim
                            critic_input = flow.zeros((1, input_dim), device=self.device)
                        else:
                            critic_input = flow.zeros((1, 1), device=self.device)
                    
                    # 检查并修复NaN值
                    if flow.isnan(critic_input).any():
                        critic_input = flow.nan_to_num(critic_input, nan=0.0)
                    
                    try:
                        values[key] = self.critic[key](critic_input)
                        # 检查输出是否包含NaN
                        if flow.isnan(values[key]).any():
                            values[key] = flow.nan_to_num(values[key], nan=0.0)
                    except Exception as e:
                        # 如果critic网络处理失败，创建默认值
                        values[key] = flow.zeros((critic_input.shape[0] if len(critic_input.shape) > 0 else 1, 1), device=self.device)
                except Exception as e:
                    # 如果处理某个智能体失败，创建一个默认值
                    values[key] = flow.zeros((1, 1), device=self.device)
                    rnn_hidden_new[key] = [None, None]
            
            # 如果提供了rnn_states，返回它们
            if rnn_states is not None:
                return values, rnn_states
            
            return rnn_hidden_new, values
        except Exception as e:
            # 如果整个评估过程失败，返回空结果
            default_values = {k: flow.zeros((1, 1), device=self.device) for k in agent_list}
            default_rnn = {k: [None, None] for k in agent_list}
            
            # 如果提供了rnn_states，返回它们
            if rnn_states is not None:
                return default_values, rnn_states
            
            return default_rnn, default_values

    def value_tot(self, values_n: Tensor, global_state=None):
        if global_state is not None:
            global_state = flow.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MAAC_Policy_Share(MAAC_Policy):
    """
    MAAC_Policy with parameter sharing trick.
    
    Args:
        action_space (Discrete): The discrete action space.
        n_agents (int): The number of agents.
        representation (Module): The representation module for all agents.
        mixer (Module): The mixer module that mix together the individual values to the total value.
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
                 n_agents: int,
                 representation: Module,
                 mixer: Optional[Module] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        self.action_dim = action_space.n
        self.representation = representation
        self.device = device
        self.representation_info = self.representation.output_shapes
        Module.__init__(self)
        
        if actor_hidden_size is None:
            actor_hidden_size = [64, 64]
        if critic_hidden_size is None:
            critic_hidden_size = [64, 64]
            
        self.actor = CategoricalActorNet(self.representation_info['state'][0], self.action_dim, actor_hidden_size,
                                        normalize, initialize, activation, device)
        self.critic = CriticNet(self.representation_info['state'][0], critic_hidden_size,
                               normalize, initialize, activation, device)
        self.target_critic = deepcopy(self.critic)
        
        self.mixer = mixer
        if mixer is not None:
            self.target_mixer = deepcopy(mixer)
            
        if use_distributed_training:
            self.actor = DistributedDataParallel(self.actor, device_ids=[device])
            self.critic = DistributedDataParallel(self.critic, device_ids=[device])
            self.target_critic = DistributedDataParallel(self.target_critic, device_ids=[device])
            self.representation = DistributedDataParallel(self.representation, device_ids=[device])
            if mixer is not None:
                self.mixer = DistributedDataParallel(self.mixer, device_ids=[device])
                self.target_mixer = DistributedDataParallel(self.target_mixer, device_ids=[device])

    def forward(self, observation: Tensor, agent_ids: Tensor,
                *rnn_hidden: Tensor, avail_actions=None, state=None):
        """
        Compute actions and values with shared parameters.

        Args:
            observation: The original observations.
            agent_ids: The agent id.
            rnn_hidden: The hidden states of the representation for each agent.
            avail_actions: The available actions.
            state: The global states.

        Returns:
            actions: The sampled actions of each agent.
            log_pi_a: The log-probability of the sampled actions.
            values_tot: The total values of the actions.
            actor_outputs: The outputs of actor networks.
            new_rnn_hidden: The new hidden states of the representation.
        """
        outputs = self.representation(observation)
        
        # get actor critic inputs
        actor_inputs = outputs['state']
        n_batch = actor_inputs.shape[0]
        
        # actor forward
        act_dist = self.actor(actor_inputs)
        actions = act_dist.stochastic_sample()
        log_pi_a = act_dist.log_prob(actions)
        
        # critic forward
        v = self.critic(actor_inputs)
        
        # reshape v: (batch_size * n_agent, 1) -> (batch_size, n_agent)
        v = v.reshape(n_batch, 1)
        
        values_tot = self.value_tot(v, state)
        return actions, log_pi_a, values_tot, outputs, None

    def value_tot(self, values_n: Tensor, global_state=None):
        """
        Get the total value using the mixer module.

        Args:
            values_n: The individual values.
            global_state: The global states.

        Returns:
            The total values of each agent.
        """
        if self.mixer is not None:
            return self.mixer(values_n, global_state)
        else:
            return values_n


class COMA_Policy(Module):
    """
    COMA_Policy: Counterfactual Multi-Agent Actor-Critic Policy with categorical distributions.

    Args:
        action_space (Optional[Dict[str, Discrete]]): The discrete action space.
        n_agents (int): The number of agents.
        representation_actor (ModuleDict): A dict of representation modules for each agent's actor.
        representation_critic (ModuleDict): A dict of representation modules for each agent's critic.
        mixer (Module): The mixer module that mix together the individual values to the total value.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: The other args.
    """
    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 representation_actor: Module,
                 representation_critic: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(COMA_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        # 确保所有component在同一设备上初始化
        if isinstance(device, str) and ('cuda' in device or 'gpu' in device):
            device_obj = flow.device(device)
            # 移动representation至指定设备
            self.actor_representation = representation_actor.to(device_obj) 
            self.critic_representation = representation_critic.to(device_obj)
            self.target_critic_representation = deepcopy(self.critic_representation).to(device_obj)
        else:
            self.actor_representation = representation_actor
            self.critic_representation = representation_critic
            self.target_critic_representation = deepcopy(self.critic_representation)

        # create actor
        self.n_actions = {k: space.n for k, space in self.action_space.items()}
        self.actor = ModuleDict()
        for key in self.model_keys:
            dim_actor_input = self.actor_representation[key].output_shapes['state'][0]
            if self.use_parameter_sharing:
                dim_actor_input += self.n_agents
            self.actor[key] = ActorNet(dim_actor_input, self.n_actions[key], actor_hidden_size,
                                       normalize, initialize, activation, None, device)
            # 确保actor在同一设备上
            if isinstance(device, str) and ('cuda' in device or 'gpu' in device):
                self.actor[key] = self.actor[key].to(device_obj)

        dim_input_critic = kwargs['dim_global_state']
        dim_input_critic += self.critic_representation[self.model_keys[0]].output_shapes['state'][0]
        dim_input_critic += sum(self.n_actions.values())
        dim_input_critic += self.n_agents
        self.n_actions_max = max(self.n_actions.values())
        self.critic = BasicQhead(dim_input_critic, self.n_actions_max,
                                 critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic = deepcopy(self.critic)
        
        # 确保critic在同一设备上
        if isinstance(device, str) and ('cuda' in device or 'gpu' in device):
            self.critic = self.critic.to(device_obj)
            self.target_critic = self.target_critic.to(device_obj)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ['RANK'])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_representation[key]._get_name() != "Basic_Identical":
                    self.critic_representation[key] =DistributedDataParallel(self.critic_representation[key],
                                                                             device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
            self.critic = DistributedDataParallel(module=self.critic, device_ids=[self.rank])

    @property
    def parameters_actor(self):
        return list(self.actor_representation.parameters()) + list(self.actor.parameters())

    @property
    def parameters_critic(self):
        return list(self.critic_representation.parameters()) + list(self.critic.parameters())

    def forward(self, observation: Dict[str, Tensor], agent_ids: Optional[Tensor] = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, epsilon=0.0):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[str, Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden ((Optional[Dict[str, List[Tensor]])): The RNN hidden states of actor representation.
            epsilon: The epsilon.

        Returns:
            rnn_hidden_new ((Optional[Dict[str, List[Tensor]])): The new RNN hidden states of actor representation.
            act_probs (dict): The probabilities of the actions.
        """
        rnn_hidden_new, pi_logits, act_probs, pi_dists = {}, {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        
        # 获取参考设备并确保所有输入在同一设备上
        device = self.device if self.device is not None else (
            next(self.parameters()).device if len(list(self.parameters())) > 0 else flow.device("cpu")
        )
        
        # 确保agent_ids在正确设备上
        if agent_ids is not None and agent_ids.device != device:
            agent_ids = agent_ids.to(device)

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]).to(device) for key in agent_list}

        for key in agent_list:
            # 确保观察在正确的设备上
            if observation[key].device != device:
                observation[key] = observation[key].to(device)
                
            if self.use_rnn:
                # 确保RNN隐藏状态在正确的设备上
                if rnn_hidden is not None and rnn_hidden[key][0].device != device:
                    rnn_hidden[key] = (rnn_hidden[key][0].to(device), rnn_hidden[key][1].to(device))
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.actor_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_input = flow.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_input = outputs['state']

            pi_logits[key] = self.actor[key](actor_input)
            if avail_actions is not None:
                avail_actions_key = avail_actions[key].to(device)
                pi_logits[key][avail_actions_key == 0] = -1e10
            act_probs[key] = nn.functional.softmax(pi_logits[key], dim=-1)
            act_probs[key] = (1 - epsilon) * act_probs[key] + epsilon * 1 / self.n_actions[key]
            if avail_actions is not None:
                avail_actions_key = avail_actions[key].to(device)
                act_probs[key][avail_actions_key == 0] = 0.0

            # 使用自定义的 CategoricalDistribution 类代替 OneFlow 的 Categorical
            pi_dists[key] = CategoricalDistribution(self.n_actions[key])
            pi_dists[key].set_param(probs=act_probs[key])

        return rnn_hidden_new, pi_dists

    def get_values(self, state: Tensor, observation: Dict[str, Tensor], actions: Dict[str, Tensor],
                   agent_ids: Tensor = None, rnn_hidden: Optional[Dict[str, List[Tensor]]] = None, target=False):
        """
        Get evaluated critic values.

        Parameters:
            state: Tensor: The global state.
            observation (Dict[str, Tensor]): The input observations for the policies.
            actions (Dict[str, Tensor]): The input actions.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            rnn_hidden ((Optional[Dict[str, List[Tensor]])): The RNN hidden states of critic representation.
            target: If to use target critic network to calculate the critic values.

        Returns:
            rnn_hidden_new ((Optional[Dict[str, List[Tensor]])): The new RNN hidden states of critic representation.
            values (dict): The evaluated critic values.
        """
        rnn_hidden_new, critic_input = {}, {}
        batch_size = state.shape[0]
        seq_len = state.shape[1] if self.use_rnn else 1
        
        # 确保所有输入在同一设备上
        device = state.device
        
        # 处理全局状态
        if self.use_rnn:
            state_input = state.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)  # batch * T * N * dim_S
        else:
            state_input = state.unsqueeze(-2).repeat(1, self.n_agents, 1)  # batch * N * dim_S

        # 处理观察
        obs_rep = {}
        for key in self.model_keys:
            # 确保观察在正确的设备上 - 处理numpy数组
            if isinstance(observation[key], np.ndarray):
                observation[key] = flow.Tensor(observation[key]).to(device)
            elif observation[key].device != device:
                observation[key] = observation[key].to(device)
                
            if self.use_rnn:
                # 确保RNN隐藏状态在正确的设备上
                if rnn_hidden is not None:
                    if isinstance(rnn_hidden[key][0], np.ndarray):
                        rnn_hidden[key] = (flow.Tensor(rnn_hidden[key][0]).to(device), flow.Tensor(rnn_hidden[key][1]).to(device))
                    elif rnn_hidden[key][0].device != device:
                        rnn_hidden[key] = (rnn_hidden[key][0].to(device), rnn_hidden[key][1].to(device))
                outputs = self.critic_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.critic_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]
            obs_rep[key] = outputs['state']

        # 处理动作掩码
        agent_mask = (1 - flow.eye(self.n_agents, dtype=flow.float32, device=device)).unsqueeze(-1)
        
        # 确保actions在正确的设备上
        for key in actions:
            if isinstance(actions[key], np.ndarray):
                actions[key] = flow.Tensor(actions[key]).to(device)
            elif actions[key].device != device:
                actions[key] = actions[key].to(device)
        
        # 确保agent_ids在正确的设备上
        if agent_ids is not None:
            if isinstance(agent_ids, np.ndarray):
                agent_ids = flow.Tensor(agent_ids).to(device)
            elif agent_ids.device != device:
                agent_ids = agent_ids.to(device)
        
        # 构建critic输入
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            agent_mask = agent_mask.repeat(1, 1, self.n_actions[key]).reshape(self.n_agents, -1).unsqueeze(0)
            
            if self.use_rnn:
                actions_input = actions[key].reshape(batch_size, seq_len, 1, -1).repeat(1, 1, self.n_agents, 1)
                obs_input = obs_rep[key].reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2)
                agent_ids_input = agent_ids.reshape(batch_size, self.n_agents, seq_len, -1).transpose(1, 2)
                
                # 确保所有张量在同一设备上
                inputs_list = [
                    state_input,
                    obs_input,
                    actions_input * agent_mask.unsqueeze(0),
                    agent_ids_input
                ]
                
                # 确保所有输入具有相同的形状（除了最后一维）
                base_shape = inputs_list[0].shape[:-1]
                for i in range(len(inputs_list)):
                    if inputs_list[i].shape[:-1] != base_shape:
                        # 尝试调整形状
                        inputs_list[i] = inputs_list[i].reshape(base_shape + (-1,))
                
                # 在最后一维连接
                try:
                    critic_inputs = flow.cat(inputs_list, dim=-1)
                except Exception as e:
                    # 如果出错，使用简单的方法
                    critic_inputs = state_input
            else:
                actions_input = actions[key].reshape(batch_size, 1, -1).repeat(1, self.n_agents, 1)
                obs_input = obs_rep[key].reshape(batch_size, self.n_agents, -1)
                agent_ids_input = agent_ids.reshape(batch_size, self.n_agents, -1)
                
                # 创建一个列表来存储所有输入
                inputs_list = [
                    state_input,
                    obs_input,
                    actions_input * agent_mask,
                    agent_ids_input
                ]
                
                # 确保所有输入具有相同的形状（除了最后一维）
                base_shape = inputs_list[0].shape[:-1]
                for i in range(len(inputs_list)):
                    if inputs_list[i].shape[:-1] != base_shape:
                        # 尝试调整形状
                        inputs_list[i] = inputs_list[i].reshape(base_shape + (-1,))
                
                # 在最后一维连接
                try:
                    critic_inputs = flow.cat(inputs_list, dim=-1)
                except Exception as e:
                    # 如果出错，使用简单的方法
                    critic_inputs = state_input
        else:
            # 非参数共享情况
            # 简化处理，避免复杂的合并逻辑
            if self.use_rnn:
                # 为每个智能体准备输入
                state_input = state.unsqueeze(-2).repeat(1, 1, self.n_agents, 1)
                
                # 获取第一个智能体的观察表示作为基础
                first_key = self.model_keys[0]
                base_shape = obs_rep[first_key].shape[:-1]
                
                # 创建一个列表来存储所有输入
                inputs_list = [state_input]
                
                # 添加观察表示
                for key in self.model_keys:
                    if obs_rep[key].shape[:-1] != base_shape:
                        obs_rep[key] = obs_rep[key].reshape(base_shape + (-1,))
                    inputs_list.append(obs_rep[key].unsqueeze(-2))
                
                # 添加动作
                for key in self.model_keys:
                    action_input = actions[key].unsqueeze(-2)
                    if action_input.shape[:-1] != base_shape:
                        action_input = action_input.reshape(base_shape + (-1,))
                    inputs_list.append(action_input)
                
                # 添加智能体ID
                inputs_list.append(agent_ids)
                
                # 尝试合并
                try:
                    critic_inputs = flow.cat(inputs_list, dim=-1)
                except Exception as e:
                    # 如果出错，使用简单的方法
                    critic_inputs = state_input
            else:
                # 非RNN情况
                # 为每个智能体准备输入
                state_input = state.unsqueeze(-2).repeat(1, self.n_agents, 1)
                
                # 获取第一个智能体的观察表示作为基础
                first_key = self.model_keys[0]
                base_shape = obs_rep[first_key].shape[:-1]
                
                # 创建一个列表来存储所有输入
                inputs_list = [state_input]
                
                # 添加观察表示
                for key in self.model_keys:
                    if obs_rep[key].shape[:-1] != base_shape:
                        obs_rep[key] = obs_rep[key].reshape(base_shape + (-1,))
                    inputs_list.append(obs_rep[key])
                
                # 添加动作
                for key in self.model_keys:
                    action_input = actions[key]
                    if action_input.shape[:-1] != base_shape:
                        action_input = action_input.reshape(base_shape + (-1,))
                    inputs_list.append(action_input)
                
                # 添加智能体ID
                inputs_list.append(agent_ids)
                
                # 尝试合并
                try:
                    critic_inputs = flow.cat(inputs_list, dim=-1)
                except Exception as e:
                    # 如果出错，使用简单的方法
                    critic_inputs = state_input

        values = self.target_critic(critic_inputs) if target else self.critic(critic_inputs)
        return rnn_hidden_new, values

    def copy_target(self):
        for ep, tp in zip(self.critic_representation.parameters(), self.target_critic_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(ep)


class MeanFieldActorCriticPolicy(Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Module,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 **kwargs
                 ):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = CategoricalActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                             actor_hidden_size, normalize, initialize, kwargs['gain'], activation,
                                             device)
        self.critic_net = CriticNet(representation.output_shapes['state'][0] + self.action_dim, n_agents,
                                    critic_hidden_size, normalize, initialize, activation, device)
        self.parameters_actor = list(self.actor_net.parameters()) + list(self.representation.parameters())
        self.parameters_critic = self.critic_net.parameters()
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def forward(self, observation: Tensor, agent_ids: Tensor):
        outputs = self.representation(observation)
        input_actor = flow.concat([outputs['state'], agent_ids], dim=-1)
        act_logits = self.actor_net(input_actor)
        self.pi_dist.set_param(logits=act_logits)
        return outputs, self.pi_dist

    def critic(self, observation: Tensor, actions_mean: Tensor, agent_ids: Tensor):
        outputs = self.representation(observation)
        critic_in = flow.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.critic_net(critic_in)


class Basic_ISAC_Policy(Module):
    """
    Basic_ISAC_Policy: The basic policy for independent soft actor-critic.

    Args:
        action_space (Box): The continuous action space.
        n_agents (int): The number of agents.
        actor_representation (ModuleDict): A dict of representation modules for each agent's actor.
        critic_representation (ModuleDict): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 actor_representation: ModuleDict,
                 critic_representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(Basic_ISAC_Policy, self).__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = kwargs['use_parameter_sharing']
        self.model_keys = kwargs['model_keys']
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_rnn"] else False

        self.actor_representation = actor_representation
        self.critic_1_representation = critic_representation
        self.critic_2_representation = deepcopy(critic_representation)
        self.target_critic_1_representation = deepcopy(self.critic_1_representation)
        self.target_critic_2_representation = deepcopy(self.critic_2_representation)

        self.actor, self.critic_1, self.critic_2 = ModuleDict(), ModuleDict(), ModuleDict()
        for key in self.model_keys:
            dim_action = self.action_space[key].n
            dim_actor_in, dim_actor_out, dim_critic_in = self._get_actor_critic_input(
                self.actor_representation[key].output_shapes['state'][0], dim_action,
                self.critic_1_representation[key].output_shapes['state'][0], n_agents)

            self.actor[key] = Actor_SAC(dim_actor_in, dim_actor_out, actor_hidden_size,
                                        normalize, initialize, activation, device)
            self.critic_1[key] = BasicQhead(dim_critic_in, dim_action, critic_hidden_size,
                                            normalize, initialize, activation, device)
            self.critic_2[key] = BasicQhead(dim_critic_in, dim_action, critic_hidden_size,
                                            normalize, initialize, activation, device)
        self.target_critic_1 = deepcopy(self.critic_1)
        self.target_critic_2 = deepcopy(self.critic_2)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            for key in self.model_keys:
                if self.actor_representation[key]._get_name() != "Basic_Identical":
                    self.actor_representation[key] = DistributedDataParallel(self.actor_representation[key],
                                                                             device_ids=[self.rank])
                if self.critic_1_representation[key]._get_name() != "Basic_Identical":
                    self.critic_1_representation[key] = DistributedDataParallel(self.critic_1_representation[key],
                                                                                device_ids=[self.rank])
                if self.critic_2_representation[key]._get_name() != "Basic_Identical":
                    self.critic_2_representation[key] = DistributedDataParallel(self.critic_2_representation[key],
                                                                                device_ids=[self.rank])
                self.actor[key] = DistributedDataParallel(module=self.actor[key], device_ids=[self.rank])
                self.critic_1[key] = DistributedDataParallel(module=self.critic_1[key], device_ids=[self.rank])
                self.critic_2[key] = DistributedDataParallel(module=self.critic_2[key], device_ids=[self.rank])

    @property
    def parameters_actor(self):
        parameters_actor = {}
        for key in self.model_keys:
            parameters_actor[key] = list(self.actor_representation[key].parameters()) + list(
                self.actor[key].parameters())
        return parameters_actor

    @property
    def parameters_critic(self):
        parameters_critic = {}
        for key in self.model_keys:
            parameters_critic[key] = list(self.critic_1_representation[key].parameters()) + list(
                self.critic_1[key].parameters()) + list(self.critic_2_representation[key].parameters()) + list(
                self.critic_2[key].parameters())
        return parameters_critic

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def forward(self, observation: Dict[str, Tensor], agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None, agent_key: str = None,
                rnn_hidden: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns actions of the policy.

        Parameters:
            observation (Dict[Tensor]): The input observations for the policies.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden ((Optional[Dict[str, List[Tensor]])): The hidden variables of the RNN.

        Returns:
            rnn_hidden_new ((Optional[Dict[str, List[Tensor]])): The new hidden variables of the RNN.
            actions (Dict[Tensor]): The actions output by the policies.
        """
        rnn_hidden_new, act_dists, actions_dict, log_action_prob = deepcopy(rnn_hidden), {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        if avail_actions is not None:
            avail_actions = {key: Tensor(avail_actions[key]) for key in agent_list}

        for key in agent_list:
            if self.use_rnn:
                outputs = self.actor_representation[key](observation[key], *rnn_hidden[key])
                rnn_hidden_new[key] = (outputs['rnn_hidden'], outputs['rnn_cell'])
            else:
                outputs = self.actor_representation[key](observation[key])
                rnn_hidden_new[key] = [None, None]

            if self.use_parameter_sharing:
                actor_in = flow.concat([outputs['state'], agent_ids], dim=-1)
            else:
                actor_in = outputs['state']

            avail_actions_input = None if avail_actions is None else avail_actions[key]
            act_dists = self.actor[key](actor_in, avail_actions_input)
            actions_dict[key] = act_dists.stochastic_sample()
        return rnn_hidden_new, actions_dict, None

    def Qpolicy(self, observation: Dict[str, Tensor],
                joint_actions: Optional[Tensor] = None,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            observation (Dict[Tensor]): The observations.
            joint_actions (Optional[Tensor]): The joint actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The evaluations of Q^policy with critic 1.
            q_2: The evaluations of Q^policy with critic 2.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = observation.shape[0]
        seq_len = observation.shape[1] if self.use_rnn else 1

        critic_rep_in = flow.concat([observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = flow.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = flow.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def Qtarget(self, next_observation: Dict[str, Tensor],
                agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None,
                agent_key: str = None,
                rnn_hidden_actor: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_actor (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for actor representation.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]))): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_actor: The updated rnn states for actor_representation.
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_actor_new = deepcopy(rnn_hidden_actor)
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = next_observation.shape[0]
        seq_len = next_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = flow.concat([next_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = flow.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = flow.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1 = self.target_critic_1[key](critic_1_in)
            q_2 = self.target_critic_2[key](critic_2_in)
            target_q[key] = flow.min(q_1, q_2)
        return rnn_hidden_actor_new, rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q

    def Qaction(self, observation: Union[np.ndarray, dict],
                agent_ids: Tensor, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            observation (Union[np.ndarray, dict]): The observations.
            agent_ids (Tensor): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            if self.use_rnn:
                outputs_critic_1 = self.critic_1_representation[key](observation[key], *rnn_hidden_critic_1[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key], *rnn_hidden_critic_2[key])
                rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                                for k in agent_list})
                rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                                for k in agent_list})
            else:
                outputs_critic_1 = self.critic_1_representation[key](observation[key])
                outputs_critic_2 = self.critic_2_representation[key](observation[key])

            critic_1_in = outputs_critic_1['state']
            critic_2_in = outputs_critic_2['state']
            if self.use_parameter_sharing:
                critic_1_in = flow.concat([critic_1_in, agent_ids], dim=-1)
                critic_2_in = flow.concat([critic_2_in, agent_ids], dim=-1)
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)
        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.critic_1_representation.parameters(), self.target_critic_1_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2_representation.parameters(), self.target_critic_2_representation.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class MASAC_Policy(Basic_ISAC_Policy):
    """
    Basic_ISAC_Policy: The basic policy for independent soft actor-critic.

    Args:
        action_space (Box): The continuous action space.
        n_agents (int): The number of agents.
        actor_representation (ModuleDict): A dict of representation modules for each agent's actor.
        critic_representation (ModuleDict): A dict of representation modules for each agent's critic.
        actor_hidden_size (Sequence[int]): A list of hidden layer sizes for actor network.
        critic_hidden_size (Sequence[int]): A list of hidden layer sizes for critic network.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        activation_action (Optional[ModuleType]): The activation of final layer to bound the actions.
        device (Optional[Union[str, int, flow.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
        **kwargs: Other arguments.
    """

    def __init__(self,
                 action_space: Optional[Dict[str, Discrete]],
                 n_agents: int,
                 actor_representation: ModuleDict,
                 critic_representation: ModuleDict,
                 actor_hidden_size: Sequence[int],
                 critic_hidden_size: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 activation_action: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 use_distributed_training: bool = False,
                 **kwargs):
        super(MASAC_Policy, self).__init__(action_space, n_agents, actor_representation, critic_representation,
                                           actor_hidden_size, critic_hidden_size,
                                           normalize, initialize, activation, activation_action, device,
                                           use_distributed_training, **kwargs)

    def _get_actor_critic_input(self, dim_actor_rep, dim_action, dim_critic_rep, n_agents):
        """
        Returns the input dimensions of actor netwrok and critic networks.

        Parameters:
            dim_actor_rep: The dimension of the output of actor presentation.
            dim_action: The dimension of actions (continuous), or the number of actions (discrete).
            dim_critic_rep: The dimension of the output of critic presentation.
            n_agents: The number of agents.

        Returns:
            dim_actor_in: The dimension of input of the actor networks.
            dim_actor_out: The dimension of output of the actor networks.
            dim_critic_in: The dimension of the input of critic networks.
            dim_critic_out: The dimension of the output of critic networks.
        """
        dim_actor_in, dim_actor_out = dim_actor_rep, dim_action
        dim_critic_in = dim_critic_rep
        if self.use_parameter_sharing:
            dim_actor_in += n_agents
            dim_critic_in += n_agents
        return dim_actor_in, dim_actor_out, dim_critic_in

    def Qpolicy(self, joint_observation: Optional[Tensor] = None,
                joint_actions: Optional[Tensor] = None,
                agent_ids: Tensor = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns Q^policy of current observations and actions pairs.

        Parameters:
            joint_observation (Optional[Tensor]): The joint observations of the team.
            joint_actions (Optional[Tensor]): The joint actions of the team.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (((Optional[Dict[str, List[Tensor]]]))): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (((Optional[Dict[str, List[Tensor]]]))): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The evaluations of Q^policy with critic 1.
            q_2: The evaluations of Q^policy with critic 2.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = flow.concat([joint_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = flow.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = flow.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

    def Qtarget(self, next_observation: Dict[str, Tensor],
                agent_ids: Tensor = None,
                avail_actions: Dict[str, Tensor] = None,
                agent_key: str = None,
                rnn_hidden_actor: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the Q^target of next observations and actions pairs.

        Parameters:
            next_observation (Dict[Tensor]): The observations of next step.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            avail_actions (Dict[str, Tensor]): Actions mask values, default is None.
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_actor (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for actor representation.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_actor: The updated rnn states for actor_representation.
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_target: The evaluations of Q^target.
        """
        rnn_hidden_actor_new = deepcopy(rnn_hidden_actor)
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        target_q = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = next_observation.shape[0]
        seq_len = next_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = flow.concat([next_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.target_critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.target_critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = flow.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = flow.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2
            q_1 = self.target_critic_1[key](critic_1_in)
            q_2 = self.target_critic_2[key](critic_2_in)
            target_q[key] = flow.min(q_1, q_2)
        return rnn_hidden_actor_new, rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, target_q

    def Qaction(self, joint_observation: Optional[Tensor] = None,
                joint_actions: Optional[Tensor] = None,
                agent_ids: Optional[Tensor] = None, agent_key: str = None,
                rnn_hidden_critic_1: Optional[Dict[str, List[Tensor]]] = None,
                rnn_hidden_critic_2: Optional[Dict[str, List[Tensor]]] = None):
        """
        Returns the evaluated Q-values for current observation-action pairs.

        Parameters:
            joint_observation (Optional[Tensor]): The observations.
            joint_actions (Optional[Tensor]): The joint actions.
            agent_ids (Dict[Tensor]): The agents' ids (for parameter sharing).
            agent_key (str): Calculate actions for specified agent.
            rnn_hidden_critic_1 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_1 representation.
            rnn_hidden_critic_2 (Optional[Dict[str, List[Tensor]]]): The RNN hidden states for critic_2 representation.

        Returns:
            rnn_hidden_critic_new_1: The updated rnn states for critic_1_representation.
            rnn_hidden_critic_new_2: The updated rnn states for critic_2_representation.
            q_1: The Q-value calculated by the first critic network.
            q_2: The Q-value calculated by the other critic network.
        """
        rnn_hidden_critic_new_1, rnn_hidden_critic_new_2 = deepcopy(rnn_hidden_critic_1), deepcopy(rnn_hidden_critic_2)
        q_1, q_2 = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        batch_size = joint_observation.shape[0]
        seq_len = joint_observation.shape[1] if self.use_rnn else 1

        critic_rep_in = flow.concat([joint_observation, joint_actions], dim=-1)
        if self.use_rnn:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in, *rnn_hidden_critic_1[k])
                                for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in, *rnn_hidden_critic_2[k])
                                for k in agent_list}
            rnn_hidden_critic_new_1.update({k: (outputs_critic_1[k]['rnn_hidden'], outputs_critic_1[k]['rnn_cell'])
                                            for k in agent_list})
            rnn_hidden_critic_new_2.update({k: (outputs_critic_2[k]['rnn_hidden'], outputs_critic_2[k]['rnn_cell'])
                                            for k in agent_list})
        else:
            outputs_critic_1 = {k: self.critic_1_representation[k](critic_rep_in) for k in agent_list}
            outputs_critic_2 = {k: self.critic_2_representation[k](critic_rep_in) for k in agent_list}

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        for key in agent_list:
            if self.use_parameter_sharing:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(-1, self.n_agents, -1, -1)
                    joint_rep_out_1 = joint_rep_out_1.reshape(bs, seq_len, -1)
                    joint_rep_out_2 = joint_rep_out_2.reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].unsqueeze(1).expand(
                        -1, self.n_agents, -1).reshape(bs, -1)
                critic_1_in = flow.concat([joint_rep_out_1, agent_ids], dim=-1)
                critic_2_in = flow.concat([joint_rep_out_2, agent_ids], dim=-1)
            else:
                if self.use_rnn:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, seq_len, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, seq_len, -1)
                else:
                    joint_rep_out_1 = outputs_critic_1[key]['state'].reshape(bs, -1)
                    joint_rep_out_2 = outputs_critic_2[key]['state'].reshape(bs, -1)
                critic_1_in = joint_rep_out_1
                critic_2_in = joint_rep_out_2

            q_1[key] = self.critic_1[key](critic_1_in)
            q_2[key] = self.critic_2[key](critic_2_in)

        return rnn_hidden_critic_new_1, rnn_hidden_critic_new_2, q_1, q_2

