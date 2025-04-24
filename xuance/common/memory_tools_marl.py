import numpy as np
import numpy as np
from abc import ABC, abstractmethod
from xuance.common import List, Dict, Optional, Union
from gym.spaces import Space
from xuance.common import space2shape, create_memory


class BaseBuffer(ABC):
    """
    Basic buffer for MARL algorithms.
    """
    def __init__(self, *args):
        self.agent_keys, self.state_space, self.obs_space, self.act_space, self.n_envs, self.buffer_size = args
        assert self.buffer_size % self.n_envs == 0, "buffer_size must be divisible by the number of envs (parallels)"
        self.n_size = self.buffer_size // self.n_envs
        self.ptr = 0  # last data pointer
        self.size = 0  # current buffer size

    @property
    def full(self):
        return self.size >= self.n_size

    @abstractmethod
    def store(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear(self, *args):
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args):
        raise NotImplementedError

    @abstractmethod
    def finish_path(self, *args, **kwargs):
        raise NotImplementedError


class MARL_OnPolicyBuffer(BaseBuffer):
    """
    Replay buffer for on-policy MARL algorithms.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        use_gae (bool): Whether to use GAE trick.
        use_advnorm (bool): Whether to use Advantage normalization trick.
        gamma (float): Discount factor.
        gae_lam (float): gae lambda.
        max_episode_steps (int): The sequence length of each episode data.
        **kwargs: Other arguments.

    Example:
        $ state_space=None
        $ obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                     'agent_1': Box(-inf, inf, (18,), float32),
                     'agent_2': Box(-inf, inf, (18,), float32)},
        $ act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                     'agent_1': Box(0.0, 1.0, (5,), float32),
                     'agent_2': Box(0.0, 1.0, (5,), float32)},
        $ n_envs=16,
        $ buffer_size=1600,
        $ agent_keys=['agent_0', 'agent_1', 'agent_2'],
        $ memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space, obs_space=obs_space,
                                        act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                        use_gae=False, use_advnorm=False, gamma=0.99, gae_lam=0.95)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 use_gae: Optional[bool] = False,
                 use_advnorm: Optional[bool] = False,
                 gamma: Optional[float] = None,
                 gae_lam: Optional[float] = None,
                 max_episode_steps: Optional[int] = None,
                 **kwargs):
        """Initialize the on-policy buffer for multi-agent reinforcement learning.

        Args:
            agent_keys: The names of agents.
            state_space: The global state spaces.
            obs_space: The observation spaces for all agents.
            act_space: The action spaces for all agents.
            n_envs: The number of parallel environments.
            buffer_size: The buffer size.
            use_gae: Whether to use GAE to calculate the advantage function.
            use_advnorm: Whether to use the advantage normalization.
            gamma: The discount factor.
            gae_lam: The factor of GAE.
        """
        self.agent_keys = agent_keys
        self.model_keys = agent_keys if kwargs.get("use_parameter_sharing", False) is False else [agent_keys[0], ]
        self.n_envs = n_envs  # 环境数量
        self.n_size = buffer_size  # 每个环境的缓冲区大小
        self.buffer_size = self.n_envs * self.n_size  # 总缓冲区大小
        self.max_eps_len = max_episode_steps if max_episode_steps is not None else buffer_size
        
        # 设置空间属性
        self.state_space = state_space
        self.obs_space = obs_space
        self.action_space = act_space
        self.act_space = act_space  # 添加别名以兼容
        
        self.store_global_state = False if self.state_space is None else True
        self.use_actions_mask = kwargs['use_actions_mask'] if 'use_actions_mask' in kwargs else False
        self.avail_actions_shape = kwargs['avail_actions_shape'] if 'avail_actions_shape' in kwargs else None
        self.use_gae = use_gae
        self.use_advnorm = use_advnorm
        self.gamma = gamma
        self.gae_lambda = gae_lam
        self.reward_space = ()
        self.values = {key: () for key in self.agent_keys}
        self.log_pis = {key: () for key in self.agent_keys}
        self.returns = {key: () for key in self.agent_keys}
        self.advantages = {key: () for key in self.agent_keys}
        self.terminal_space = {key: () for key in self.agent_keys}
        self.agent_mask_space = {key: () for key in self.agent_keys}
        self.clear()
        self.data_keys = self.data.keys()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (n_env=16, buffer_size=1600, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[16, 100, 18],
                            'agent_1': shape=[16, 100, 18],
                            'agent_2': shape=[16, 100, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[16, 100, 5],
                                'agent_1': shape=[16, 100, 5],
                                'agent_2': shape=[16, 100, 5]},  # dim_act: 5
                     ...}
        """
        # 打印调试信息
        print(f"初始化缓冲区: n_envs={self.n_envs}, n_size={self.n_size}, agent_keys={self.agent_keys}")
        
        try:
            # 创建基础数据结构
            self.data = {}
            
            # 创建观察空间
            self.data['obs'] = {}
            for k in self.agent_keys:
                if k in self.obs_space:
                    shape = space2shape(self.obs_space[k])
                    self.data['obs'][k] = np.zeros((self.n_envs, self.n_size) + shape, dtype=np.float32)
                else:
                    print(f"警告: 智能体 {k} 不在观察空间中")
                    # 创建一个默认的观察空间
                    self.data['obs'][k] = np.zeros((self.n_envs, self.n_size, 1), dtype=np.float32)
            
            # 创建动作空间
            self.data['actions'] = {}
            for k in self.agent_keys:
                if k in self.action_space:
                    if hasattr(self.action_space[k], 'n'):  # 离散动作空间
                        self.data['actions'][k] = np.zeros((self.n_envs, self.n_size), dtype=np.int32)
                    else:  # 连续动作空间
                        act_shape = space2shape(self.action_space[k])
                        self.data['actions'][k] = np.zeros((self.n_envs, self.n_size) + act_shape, dtype=np.float32)
                else:
                    print(f"警告: 智能体 {k} 不在动作空间中")
                    # 创建一个默认的动作空间
                    self.data['actions'][k] = np.zeros((self.n_envs, self.n_size), dtype=np.int32)
            
            # 创建其他必要的数据结构
            self.data['rewards'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['returns'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['values'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['log_pi_old'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['advantages'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['terminals'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys}
            self.data['agent_mask'] = {k: np.ones((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys}
            
            # 如果使用全局状态，创建状态空间
            if self.store_global_state and self.state_space is not None:
                state_shape = self.state_space.shape if hasattr(self.state_space, 'shape') else (1,)
                self.data['state'] = np.zeros((self.n_envs, self.n_size) + state_shape, dtype=np.float32)
                print(f"创建状态空间: 形状={state_shape}")
            
            # 如果使用动作掩码，创建动作掩码空间
            if self.use_actions_mask and self.avail_actions_shape is not None:
                self.data['avail_actions'] = {}
                for k in self.agent_keys:
                    if k in self.avail_actions_shape:
                        self.data['avail_actions'][k] = np.ones((self.n_envs, self.n_size) + self.avail_actions_shape[k], dtype=np.bool_)
                    else:
                        print(f"警告: 智能体 {k} 不在动作掩码形状中")
                        # 创建一个默认的动作掩码
                        self.data['avail_actions'][k] = np.ones((self.n_envs, self.n_size, 1), dtype=np.bool_)
            
            # 重置指针和大小
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)  # 每个环境最后一个episode的起始索引
            
            print("缓冲区初始化完成")
            # 打印数据结构
            # for k, v in self.data.items():
            #     if isinstance(v, dict):
            #         print(f"  {k}:")
            #         for agent_key, data in v.items():
            #             print(f"    {agent_key}: shape={data.shape}, dtype={data.dtype}")
            #     else:
            #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        
        except Exception as e:
            print(f"初始化缓冲区时出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 回退到最基本的数据结构
            self.data = {
                'obs': {k: np.zeros((self.n_envs, self.n_size, 1), dtype=np.float32) for k in self.agent_keys},
                'actions': {k: np.zeros((self.n_envs, self.n_size), dtype=np.int32) for k in self.agent_keys},
                'rewards': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'returns': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'values': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'log_pi_old': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'advantages': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'terminals': {k: np.zeros((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys},
                'agent_mask': {k: np.ones((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys},
            }
            if self.store_global_state:
                self.data['state'] = np.zeros((self.n_envs, self.n_size, 10), dtype=np.float32)  # 假设状态维度为10
            if self.use_actions_mask:
                self.data['avail_actions'] = {k: np.ones((self.n_envs, self.n_size, 5), dtype=np.bool_) for k in self.agent_keys}  # 假设动作数为5
            
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)
            print("使用基本数据结构初始化缓冲区")

    def store(self, **step_data):
        """ Stores a step of data into the replay buffer. """
        for data_key, data_value in step_data.items():
            if data_key in ['state']:
                # 确保数据形状与缓冲区匹配
                if data_value.shape[0] != self.n_envs:
                    raise ValueError(f"输入状态数据的环境数量({data_value.shape[0]})与缓冲区的环境数量({self.n_envs})不匹配")
                
                # 如果传入了state但数据结构中没有state字段，则动态创建
                if data_key not in self.data:
                    print(f"动态创建{data_key}字段在缓冲区中")
                    # 根据数据形状创建存储空间
                    shape = data_value.shape[1:] if len(data_value.shape) > 1 else (1,)
                    self.data[data_key] = np.zeros((self.n_envs, self.n_size) + shape, dtype=data_value.dtype)
                    # 更新数据键列表
                    self.data_keys = self.data.keys()
                    # 设置标志使得后续可以使用该字段
                    self.store_global_state = True
                
                self.data[data_key][:, self.ptr] = data_value
                continue
                
            for agt_key in self.agent_keys:
                # 检查数据形状
                if data_value[agt_key].shape[0] != self.n_envs:
                    raise ValueError(f"输入{data_key}的环境数量({data_value[agt_key].shape[0]})与缓冲区的环境数量({self.n_envs})不匹配")
                
                # 确保数据结构是正确的
                if agt_key not in self.data[data_key]:
                    raise KeyError(f"缓冲区中不存在键 {agt_key}")
                
                # 检查数据类型和键是否匹配
                try:
                    # 存储数据
                    self.data[data_key][agt_key][:, self.ptr] = data_value[agt_key]
                except (IndexError, ValueError, TypeError) as e:
                    print(f"存储数据时出错: 键={data_key}, 智能体={agt_key}, 错误={str(e)}")
                    print(f"数据形状: {data_value[agt_key].shape}, 缓冲区形状: {self.data[data_key][agt_key].shape}")
                    # 尝试用不同的存储方式
                    if isinstance(self.data[data_key], dict):
                        if isinstance(self.data[data_key][agt_key], np.ndarray):
                            if len(self.data[data_key][agt_key].shape) >= 2:
                                self.data[data_key][agt_key][:, self.ptr] = data_value[agt_key]
                            else:
                                self.data[data_key][agt_key] = data_value[agt_key]
                        else:
                            self.data[data_key][agt_key] = data_value[agt_key]
                    else:
                        raise
        
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def finish_path(self, i_env, value_next=None, value_normalizer=None):
        """
        Calculate the returns and advantages of the transitions in a path.

        Parameters:
            i_env (int): The index of the environment.
            value_next (dict): The next values of the path.
            value_normalizer (ValueNorm): The normalizer for values.
        """
        if self.ptr > self.start_ids[i_env]:
            path_slice = slice(self.start_ids[i_env], self.ptr)
            if value_next is None:
                value_next = {k: 0.0 for k in self.agent_keys}
            for key in self.agent_keys:
                use_value_norm = (value_normalizer is not None)
                rewards = self.data['rewards'][key][i_env, path_slice]
                values = self.data['values'][key][i_env, path_slice]
                if use_value_norm:
                    # 缓存所有智能体的value_normalizer
                    if isinstance(value_normalizer, dict):
                        if key not in value_normalizer:
                            default_key = None
                            # 优先使用agent_0作为默认标准化器
                            if 'agent_0' in value_normalizer:
                                default_key = 'agent_0'
                            # 如果没有agent_0，使用第一个可用的键
                            elif len(value_normalizer) > 0:
                                default_key = next(iter(value_normalizer))
                            
                            if default_key is not None:
                                try:
                                    values = value_normalizer[default_key].denormalize(values)
                                except AttributeError:
                                    # 如果denormalize失败，使用原始值
                                    pass
                        else:
                            try:
                                values = value_normalizer[key].denormalize(values)
                            except AttributeError:
                                # 如果denormalize失败，使用原始值
                                pass
                    else:
                        # 如果value_normalizer不是字典，尝试直接调用
                        try:
                            values = value_normalizer.denormalize(values)
                        except AttributeError:
                            # 如果denormalize失败，使用原始值
                            pass
                
                # 使用terminals代替masks，并且需要转换逻辑（终端状态的掩码应该是0而不是1）
                terminals = self.data['terminals'][key][i_env, path_slice]
                masks = 1.0 - terminals.astype(np.float32)  # 将终端状态转换为掩码（1-终端状态）
                
                # 安全地处理value_next
                if isinstance(value_next, dict) and key in value_next:
                    value_next_key = value_next[key]
                else:
                    value_next_key = 0.0
                
                # 确保value_next_key是标量
                if hasattr(value_next_key, 'shape') and value_next_key.shape:
                    try:
                        value_next_key = float(value_next_key.item())
                    except (ValueError, AttributeError, TypeError):
                        # 如果无法转换为标量，使用第一个元素
                        try:
                            value_next_key = float(value_next_key.flatten()[0])
                        except (IndexError, ValueError, AttributeError, TypeError):
                            # 如果仍然失败，使用0.0
                            value_next_key = 0.0
                
                # 计算GAE或普通回报
                if self.use_gae:
                    values_append = np.append(values, value_next_key)
                    gae = 0.0
                    for t in reversed(range(len(rewards))):
                        # 安全地处理masks
                        mask_t = masks[t]
                        if hasattr(mask_t, 'shape') and mask_t.shape:
                            try:
                                mask_t = float(mask_t.item())
                            except (ValueError, AttributeError, TypeError):
                                mask_t = float(mask_t.flatten()[0])
                        
                        # 安全地处理delta
                        try:
                            delta = rewards[t] + self.gamma * values_append[t + 1] * mask_t - values_append[t]
                            gae = delta + self.gamma * self.gae_lambda * mask_t * gae
                            self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = gae
                        except (IndexError, ValueError, TypeError) as e:
                            # 如果计算失败，使用0.0
                            self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = 0.0
                        
                        # 安全地计算returns
                        try:
                            # 确保t不超出values的范围
                            if t < len(values):
                                self.data['returns'][key][i_env, self.start_ids[i_env] + t] = gae + values[t]
                            else:
                                self.data['returns'][key][i_env, self.start_ids[i_env] + t] = gae
                        except (IndexError, ValueError, TypeError) as e:
                            # 如果计算失败，使用一个默认值
                            self.data['returns'][key][i_env, self.start_ids[i_env] + t] = gae
                else:
                    returns_ = np.zeros_like(rewards)
                    for t in reversed(range(len(rewards))):
                        # 安全地处理masks
                        mask_t = masks[t]
                        if hasattr(mask_t, 'shape') and mask_t.shape:
                            try:
                                mask_t = float(mask_t.item())
                            except (ValueError, AttributeError, TypeError):
                                mask_t = float(mask_t.flatten()[0])
                        
                        if t == len(rewards) - 1:
                            # 安全地处理reward_term
                            reward_term = rewards[t]
                            if hasattr(reward_term, 'shape') and reward_term.shape:
                                try:
                                    reward_term = float(reward_term.item())
                                except (ValueError, AttributeError, TypeError):
                                    reward_term = float(reward_term.flatten()[0])
                            
                            # 安全地处理gamma_term
                            gamma_term = self.gamma * value_next_key * mask_t
                            if hasattr(gamma_term, 'shape') and gamma_term.shape:
                                try:
                                    gamma_term = float(gamma_term.item())
                                except (ValueError, AttributeError, TypeError):
                                    gamma_term = float(gamma_term.flatten()[0])
                            
                            returns_[t] = reward_term + gamma_term
                        else:
                            # 安全地处理reward_term
                            reward_term = rewards[t]
                            if hasattr(reward_term, 'shape') and reward_term.shape:
                                try:
                                    reward_term = float(reward_term.item())
                                except (ValueError, AttributeError, TypeError):
                                    reward_term = float(reward_term.flatten()[0])
                            
                            # 安全地处理gamma_term
                            gamma_term = self.gamma * returns_[t + 1] * mask_t
                            if hasattr(gamma_term, 'shape') and gamma_term.shape:
                                try:
                                    gamma_term = float(gamma_term.item())
                                except (ValueError, AttributeError, TypeError):
                                    gamma_term = float(gamma_term.flatten()[0])
                            
                            returns_[t] = reward_term + gamma_term
                    
                    # 安全地设置returns
                    try:
                        self.data['returns'][key][i_env, path_slice] = returns_[:-1]
                    except (IndexError, ValueError, TypeError) as e:
                        # 如果设置失败，逐个设置
                        for t in range(len(returns_) - 1):
                            try:
                                self.data['returns'][key][i_env, self.start_ids[i_env] + t] = returns_[t]
                            except (IndexError, ValueError, TypeError):
                                pass
                    
                    # 安全地计算advantages
                    try:
                        if use_value_norm:
                            try:
                                denorm_vs = value_normalizer[key_vn].denormalize(values[:-1])
                            except (KeyError, AttributeError) as e:
                                # 如果找不到对应的键，尝试使用默认键或第一个键
                                if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                    first_key = next(iter(value_normalizer))
                                    denorm_vs = value_normalizer[first_key].denormalize(values[:-1])
                                    print(f"警告: 在value_normalizer中找不到键 {key_vn}，使用 {first_key} 代替")
                                else:
                                    # 如果无法处理，则保持values不变
                                    denorm_vs = values[:-1]
                                    print(f"警告: 无法在value_normalizer中找到有效的键: {e}")
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(denorm_vs))
                            advantages = returns_[:min_len] - denorm_vs[:min_len]
                        else:
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(values))
                            advantages = returns_[:min_len] - values[:min_len]
                        
                        # 确保advantages是有效的数组
                        if isinstance(advantages, np.ndarray):
                            # 检查并修复NaN和无限值
                            advantages = np.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 确保path_slice和advantages长度匹配
                        slice_len = path_slice.stop - path_slice.start
                        if len(advantages) < slice_len:
                            # 如果advantages长度不足，填充0
                            padded_advantages = np.zeros(slice_len)
                            padded_advantages[:len(advantages)] = advantages
                            self.data['advantages'][key][i_env, path_slice] = padded_advantages
                        else:
                            self.data['advantages'][key][i_env, path_slice] = advantages[:slice_len]
                    except (ValueError, TypeError) as e:
                        # 如果计算失败，逐个设置
                        for t in range(min(len(returns_) - 1, path_slice.stop - path_slice.start)):
                            try:
                                if use_value_norm and t < len(values):
                                    try:
                                        denorm_v = value_normalizer[key_vn].denormalize(values[t])
                                    except (KeyError, AttributeError) as e:
                                        # 如果找不到对应的键，尝试使用默认键或第一个键
                                        if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                            first_key = next(iter(value_normalizer))
                                            denorm_v = value_normalizer[first_key].denormalize(values[t])
                                        else:
                                            # 如果无法处理，则保持values不变
                                            denorm_v = values[t]
                                    advantage = returns_[t] - denorm_v
                                elif t < len(values):
                                    advantage = returns_[t] - values[t]
                                else:
                                    advantage = returns_[t]
                                
                                # 检查并修复NaN和无限值
                                if np.isnan(advantage) or np.isinf(advantage):
                                    advantage = 0.0
                                
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = advantage
                            except (IndexError, ValueError, TypeError):
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = 0.0
        self.start_ids[i_env] = self.ptr

    def sample(self, indexes: Optional[np.ndarray] = None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            indexes (int): The indexes of the data in the buffer that will be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.full, "Not enough transitions for on-policy buffer to random sample."
        samples_dict = {}
        env_choices, step_choices = divmod(indexes, self.n_size)
        for data_key in self.data_keys:
            if data_key == "advantages":
                adv_batch_dict = {}
                for agt_key in self.agent_keys:
                    adv_batch = self.data[data_key][agt_key][env_choices, step_choices]
                samples_dict[data_key] = adv_batch_dict
            elif data_key == "state":
                samples_dict[data_key] = self.data[data_key][env_choices, step_choices]
            else:
                samples_dict[data_key] = {k: self.data[data_key][k][env_choices, step_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = len(indexes)
        return samples_dict


class MARL_OnPolicyBuffer_RNN(MARL_OnPolicyBuffer):
    """
    Replay buffer for on-policy MARL algorithms with DRQN trick.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        max_episode_steps (int): The sequence length of each episode data.
        use_gae (bool): Whether to use GAE trick.
        use_advnorm (bool): Whether to use Advantage normalization trick.
        gamma (float): Discount factor.
        gae_lam (float): gae lambda.
        **kwargs: Other arguments.

    Example:
        >> state_space=None
        >> obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                      'agent_1': Box(-inf, inf, (18,), float32),
                      'agent_2': Box(-inf, inf, (18,), float32)},
        >> act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                      'agent_1': Box(0.0, 1.0, (5,), float32),
                      'agent_2': Box(0.0, 1.0, (5,), float32)},
        >> n_envs=16,
        >> buffer_size=1600,
        >> agent_keys=['agent_0', 'agent_1', 'agent_2'],
        >> max_episode_steps = 100
        >> memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space,
                                         act_space=act_space, n_envs=n_envs, buffer_size=buffer_size,
                                         max_episode_steps=max_episode_steps,
                                         use_gae=False, use_advnorm=False, gamma=0.99, gae_lam=0.95)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 max_episode_steps: int = 1,
                 use_gae: Optional[bool] = False,
                 use_advnorm: Optional[bool] = False,
                 gamma: Optional[float] = None,
                 gae_lam: Optional[float] = None,
                 **kwargs):
        self.max_eps_len = max_episode_steps
        self.n_actions = kwargs['n_actions'] if 'n_actions' in kwargs else None
        self.obs_shape = {k: space2shape(obs_space[k]) for k in agent_keys}
        self.act_shape = {k: space2shape(act_space[k]) for k in agent_keys}
        super(MARL_OnPolicyBuffer_RNN, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs,
                                                      buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)
        self.episode_data = {}
        self.clear_episodes()

    @property
    def full(self):
        return self.size >= self.buffer_size

    def clear(self):
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.buffer_size, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.data.update({
                'state': np.zeros((self.buffer_size, self.max_eps_len) + self.state_space.shape, np.float32)
            })
            self.data.update({
                'state_next': np.zeros((self.buffer_size, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                      dtype=np.bool_) for k in self.agent_keys}
            })
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'returns': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'values': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'advantages': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'log_pi_old': {k: np.zeros((self.n_envs, self.max_eps_len,), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), np.bool_)
        }
        if self.store_global_state:
            self.episode_data.update({
                'state': np.zeros((self.n_envs, self.max_eps_len) + self.state_space.shape, np.float32)
            })
            self.episode_data.update({
                'state_next': np.zeros((self.n_envs, self.max_eps_len) + self.state_space.shape, np.float32)
            })
        if self.use_actions_mask:
            self.episode_data.update({
                'avail_actions': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                      dtype=np.bool_) for k in self.agent_keys}
            })

    def store(self, **step_data):
        """
        Stores a step of data for each environment.

        Parameters:
            step_data (dict): A dict of step data that to be stored into self.episode_data.
        """
        envs_step = step_data['episode_steps']
        envs_choice = range(self.n_envs)
        self.episode_data["filled"][envs_choice, envs_step] = True
        for data_key, data_value in step_data.items():
            if data_key == "episode_steps":
                continue
            if data_key in ['state', 'state_next']:
                self.episode_data[data_key][envs_choice, envs_step] = data_value
                continue
            for agt_key in self.agent_keys:
                self.episode_data[data_key][agt_key][envs_choice, envs_step] = data_value[agt_key]

    def store_episodes(self, i_env):
        """
        Stores the episode of data for ith environment into the self.data.

        Parameters:
            i_env (int): The ith environment.
        """
        for data_key in self.data_keys:
            if data_key == "filled":
                self.data["filled"][self.ptr] = self.episode_data["filled"][i_env].copy()
                continue
            if data_key in ['state', 'state_next']:
                self.data[data_key][self.ptr] = self.episode_data[data_key][i_env].copy()
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][self.ptr] = self.episode_data[data_key][agt_key][i_env].copy()
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        # clear the filled values for ith env.
        self.episode_data['filled'][i_env] = np.zeros(self.max_eps_len, dtype=np.bool_)

    def finish_path(self,
                    i_env: Optional[int] = None,
                    i_step: Optional[int] = None,
                    value_next: Optional[Union[dict, float]] = None,
                    value_normalizer: Optional[dict] = None):
        """
        Calculates and stores the returns and advantages when an episode is finished.

        Parameters:
            i_env (int): The index of environment.
            i_step (int): The index of step for current environment.
            value_next (Optional[dict or float]): The critic values of the terminal state. Can be a dict or a scalar value (0.0 for terminal states).
            value_normalizer (Optional[dict]): The value normalizer method, default is None.
        """
        env_step = i_step if i_step < self.max_eps_len else self.max_eps_len
        path_slice = np.arange(0, env_step).astype(np.int32)

        # calculate advantages and returns
        use_value_norm = False if (value_normalizer is None) else True
        use_parameter_sharing = False
        if use_value_norm:
            if value_normalizer.keys() != set(self.agent_keys):
                use_parameter_sharing = True
                
        # 处理value_next为标量值的情况
        if value_next is None or isinstance(value_next, (int, float)):
            # 如果value_next是None或标量值，为每个智能体创建相同的值
            value_next_dict = {key: value_next if value_next is not None else 0.0 for key in self.agent_keys}
        else:
            # 如果已经是字典，直接使用
            value_next_dict = value_next
            
        for key in self.agent_keys:
            try:
                rewards = np.array(self.episode_data['rewards'][key][i_env, path_slice])
                vs = np.append(np.array(self.episode_data['values'][key][i_env, path_slice]),
                            [value_next_dict[key]], axis=0)
                dones = np.array(self.episode_data['terminals'][key][i_env, path_slice])
                # 确保dones的形状正确
                if len(dones.shape) > 1:
                    dones = dones.reshape(-1)
                returns = np.zeros_like(vs)
                last_gae_lam = 0
                step_nums = len(path_slice)
                key_vn = self.agent_keys[0] if use_parameter_sharing else key

                if self.use_gae:
                    for t in reversed(range(step_nums)):
                        try:
                            if use_value_norm:
                                vs_t = value_normalizer[key_vn].denormalize(vs[t])
                                vs_next = value_normalizer[key_vn].denormalize(vs[t + 1])
                            else:
                                vs_t, vs_next = vs[t], vs[t + 1]
                                
                            # 确保vs_t和vs_next是标量
                            if isinstance(vs_t, np.ndarray) and vs_t.size == 1:
                                vs_t = float(vs_t.item())
                            if isinstance(vs_next, np.ndarray) and vs_next.size == 1:
                                vs_next = float(vs_next.item())
                                
                            # 确保rewards[t]和dones[t]是标量
                            reward_t = rewards[t]
                            if isinstance(reward_t, np.ndarray) and reward_t.size == 1:
                                reward_t = float(reward_t.item())
                                
                            done_t = dones[t]
                            if isinstance(done_t, np.ndarray) and done_t.size == 1:
                                done_t = float(done_t.item())
                                
                            delta = reward_t + (1 - done_t) * self.gamma * vs_next - vs_t
                            last_gae_lam = delta + (1 - done_t) * self.gamma * self.gae_lambda * last_gae_lam
                            
                            # 确保last_gae_lam是标量
                            if isinstance(last_gae_lam, np.ndarray) and last_gae_lam.size == 1:
                                last_gae_lam = float(last_gae_lam.item())
                                
                            returns[t] = last_gae_lam + vs_t
                        except (IndexError, ValueError, TypeError) as e:
                            # 如果计算失败，使用0.0
                            returns[t] = 0.0
                    
                    # 安全地计算advantages
                    try:
                        if use_value_norm:
                            try:
                                denorm_vs = value_normalizer[key_vn].denormalize(vs[:-1])
                            except (KeyError, AttributeError) as e:
                                # 如果找不到对应的键，尝试使用默认键或第一个键
                                if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                    first_key = next(iter(value_normalizer))
                                    denorm_vs = value_normalizer[first_key].denormalize(vs[:-1])
                                    print(f"警告: 在value_normalizer中找不到键 {key_vn}，使用 {first_key} 代替")
                                else:
                                    # 如果无法处理，则保持values不变
                                    denorm_vs = vs[:-1]
                                    print(f"警告: 无法在value_normalizer中找到有效的键: {e}")
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(denorm_vs))
                            advantages = returns_[:min_len] - denorm_vs[:min_len]
                        else:
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(values))
                            advantages = returns_[:min_len] - values[:min_len]
                        
                        # 确保advantages是有效的数组
                        if isinstance(advantages, np.ndarray):
                            # 检查并修复NaN和无限值
                            advantages = np.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 确保path_slice和advantages长度匹配
                        slice_len = path_slice.stop - path_slice.start
                        if len(advantages) < slice_len:
                            # 如果advantages长度不足，填充0
                            padded_advantages = np.zeros(slice_len)
                            padded_advantages[:len(advantages)] = advantages
                            self.data['advantages'][key][i_env, path_slice] = padded_advantages
                        else:
                            self.data['advantages'][key][i_env, path_slice] = advantages[:slice_len]
                    except (ValueError, TypeError) as e:
                        # 如果计算失败，逐个设置
                        for t in range(min(len(returns_) - 1, path_slice.stop - path_slice.start)):
                            try:
                                if use_value_norm and t < len(values):
                                    try:
                                        denorm_v = value_normalizer[key_vn].denormalize(values[t])
                                    except (KeyError, AttributeError) as e:
                                        # 如果找不到对应的键，尝试使用默认键或第一个键
                                        if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                            first_key = next(iter(value_normalizer))
                                            denorm_v = value_normalizer[first_key].denormalize(values[t])
                                        else:
                                            # 如果无法处理，则保持values不变
                                            denorm_v = values[t]
                                    advantage = returns_[t] - denorm_v
                                elif t < len(values):
                                    advantage = returns_[t] - values[t]
                                else:
                                    advantage = returns_[t]
                                
                                # 检查并修复NaN和无限值
                                if np.isnan(advantage) or np.isinf(advantage):
                                    advantage = 0.0
                                
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = advantage
                            except (IndexError, ValueError, TypeError):
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = 0.0
                else:
                    returns_ = np.zeros_like(vs)
                    for t in reversed(range(step_nums)):
                        try:
                            # 确保rewards[t]和dones[t]是标量
                            reward_t = rewards[t]
                            if isinstance(reward_t, np.ndarray) and reward_t.size == 1:
                                reward_t = float(reward_t.item())
                                
                            done_t = dones[t]
                            if isinstance(done_t, np.ndarray) and done_t.size == 1:
                                done_t = float(done_t.item())
                            
                            if t == step_nums - 1:
                                # 处理最后一步
                                value_next_key = value_next_dict[key]
                                if isinstance(value_next_key, np.ndarray) and value_next_key.size == 1:
                                    value_next_key = float(value_next_key.item())
                                    
                                gamma_term = (1 - done_t) * self.gamma * value_next_key
                            else:
                                # 处理中间步骤
                                return_next = returns_[t + 1]
                                if isinstance(return_next, np.ndarray) and return_next.size == 1:
                                    return_next = float(return_next.item())
                                    
                                gamma_term = (1 - done_t) * self.gamma * return_next
                                
                            # 确保gamma_term是标量
                            if isinstance(gamma_term, np.ndarray) and gamma_term.size == 1:
                                gamma_term = float(gamma_term.item())
                                
                            returns_[t] = reward_t + gamma_term
                        except (IndexError, ValueError, TypeError) as e:
                            # 如果计算失败，使用0.0
                            returns_[t] = 0.0
                    
                    # 安全地计算advantages
                    try:
                        if use_value_norm:
                            try:
                                denorm_vs = value_normalizer[key_vn].denormalize(vs[:-1])
                            except (KeyError, AttributeError) as e:
                                # 如果找不到对应的键，尝试使用默认键或第一个键
                                if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                    first_key = next(iter(value_normalizer))
                                    denorm_vs = value_normalizer[first_key].denormalize(vs[:-1])
                                    print(f"警告: 在value_normalizer中找不到键 {key_vn}，使用 {first_key} 代替")
                                else:
                                    # 如果无法处理，则保持values不变
                                    denorm_vs = vs[:-1]
                                    print(f"警告: 无法在value_normalizer中找到有效的键: {e}")
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(denorm_vs))
                            advantages = returns_[:min_len] - denorm_vs[:min_len]
                        else:
                            # 确保长度匹配
                            min_len = min(len(returns_) - 1, len(values))
                            advantages = returns_[:min_len] - values[:min_len]
                        
                        # 确保advantages是有效的数组
                        if isinstance(advantages, np.ndarray):
                            # 检查并修复NaN和无限值
                            advantages = np.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 确保path_slice和advantages长度匹配
                        slice_len = path_slice.stop - path_slice.start
                        if len(advantages) < slice_len:
                            # 如果advantages长度不足，填充0
                            padded_advantages = np.zeros(slice_len)
                            padded_advantages[:len(advantages)] = advantages
                            self.data['advantages'][key][i_env, path_slice] = padded_advantages
                        else:
                            self.data['advantages'][key][i_env, path_slice] = advantages[:slice_len]
                    except (ValueError, TypeError) as e:
                        # 如果计算失败，逐个设置
                        for t in range(min(len(returns_) - 1, path_slice.stop - path_slice.start)):
                            try:
                                if use_value_norm and t < len(values):
                                    try:
                                        denorm_v = value_normalizer[key_vn].denormalize(values[t])
                                    except (KeyError, AttributeError) as e:
                                        # 如果找不到对应的键，尝试使用默认键或第一个键
                                        if isinstance(value_normalizer, dict) and len(value_normalizer) > 0:
                                            first_key = next(iter(value_normalizer))
                                            denorm_v = value_normalizer[first_key].denormalize(values[t])
                                        else:
                                            # 如果无法处理，则保持values不变
                                            denorm_v = values[t]
                                    advantage = returns_[t] - denorm_v
                                elif t < len(values):
                                    advantage = returns_[t] - values[t]
                                else:
                                    advantage = returns_[t]
                                
                                # 检查并修复NaN和无限值
                                if np.isnan(advantage) or np.isinf(advantage):
                                    advantage = 0.0
                                
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = advantage
                            except (IndexError, ValueError, TypeError):
                                self.data['advantages'][key][i_env, self.start_ids[i_env] + t] = 0.0
                    
                    returns = returns_[:-1]

                # 安全地设置returns和advantages
                try:
                    self.episode_data['returns'][key][i_env, path_slice] = returns if not self.use_gae else returns[:-1]
                    self.episode_data['advantages'][key][i_env, path_slice] = self.data['advantages'][key][i_env, path_slice]
                except (IndexError, ValueError, TypeError) as e:
                    # 如果设置失败，逐个设置
                    for t in range(len(path_slice)):
                        try:
                            self.episode_data['returns'][key][i_env, t] = returns[t] if t < len(returns) else 0.0
                            self.episode_data['advantages'][key][i_env, t] = self.data['advantages'][key][i_env, t]
                        except (IndexError, ValueError, TypeError):
                            pass
            except Exception as e:
                # 如果整个处理过程失败，跳过这个智能体
                continue
                
        self.store_episodes(i_env)

    def sample(self, indexes: Optional[np.ndarray] = None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            indexes (int): The indexes of the data in the buffer that will be sampled.

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.full, "Not enough transitions for on-policy buffer to random sample"
        episode_choices = indexes
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key == "filled":
                samples_dict["filled"] = self.data['filled'][episode_choices]
                continue
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][episode_choices]
                continue
            samples_dict[data_key] = {k: self.data[data_key][k][episode_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = len(indexes)
        samples_dict['sequence_length'] = self.max_eps_len
        return samples_dict


class MeanField_OnPolicyBuffer(MARL_OnPolicyBuffer):
    """
    Replay buffer for on-policy Mean-Field MARL algorithms (Mean-Field Actor-Critic).

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        n_size: buffer size of trajectory data for one environment.
        use_gae: whether to use GAE trick.
        use_advnorm: whether to use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
        kwargs: the other arguments.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs,
                 n_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        self.prob_space = kwargs['prob_space']
        super(MeanField_OnPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                       done_space, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam,
                                                       **kwargs)

    def clear(self):
        self.data.update({
            'obs': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.obs_space).astype(np.float32),
            'actions': np.zeros((self.n_envs, self.n_size, self.n_agents) + self.act_space).astype(np.float32),
            'act_mean': np.zeros((self.n_envs, self.n_size,) + self.prob_space).astype(np.float32),
            'rewards': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
            'returns': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
            'values': np.zeros((self.n_envs, self.n_size, self.n_agents, 1)).astype(np.float32),
            'advantages': np.zeros((self.n_envs, self.n_size,) + self.rew_space).astype(np.float32),
            'terminals': np.zeros((self.n_envs, self.n_size,) + self.done_space).astype(np.bool_),
            'agent_mask': np.ones((self.n_envs, self.n_size, self.n_agents)).astype(np.bool_),
        })
        if self.state_space is not None:
            self.data.update({'state': np.zeros((self.n_envs, self.n_size,) + self.state_space).astype(np.float32)})
        self.ptr, self.size = 0, 0
        self.start_ids = np.zeros(self.n_envs)

    def finish_ac_path(self, value, i_env):  # when an episode is finished
        if self.size == 0:
            return
        self.start_ids[i_env] = self.ptr


class COMA_Buffer(MARL_OnPolicyBuffer):
    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        # 将n_agents转换为agent_keys列表
        agent_keys = [f"agent_{i}" for i in range(n_agents)]
        self.n_agents = n_agents
        
        super(COMA_Buffer, self).__init__(
            agent_keys=agent_keys,
            state_space=state_space,
            obs_space=obs_space,
            act_space=act_space,
            n_envs=n_envs,
            buffer_size=buffer_size,
            use_gae=use_gae,
            use_advnorm=use_advnorm,
            gamma=gamma,
            gae_lam=gae_lam,
            **kwargs
        )
        print(f"COMA_Buffer初始化完成: n_agents={n_agents}, n_envs={n_envs}, buffer_size={buffer_size}")

    def clear(self):
        """
        为COMA算法专门设计的缓冲区结构
        """
        print(f"初始化COMA缓冲区: n_agents={self.n_agents}, n_envs={self.n_envs}, n_size={self.n_size}")
        
        try:
            # 创建基础数据结构
            self.data = {}
            
            # 创建观察空间
            self.data['obs'] = {}
            for k in self.agent_keys:
                if k in self.obs_space:
                    shape = space2shape(self.obs_space[k])
                    self.data['obs'][k] = np.zeros((self.n_envs, self.n_size) + shape, dtype=np.float32)
                else:
                    print(f"警告: 智能体 {k} 不在观察空间中")
                    # 创建一个默认的观察空间
                    self.data['obs'][k] = np.zeros((self.n_envs, self.n_size, 1), dtype=np.float32)
            
            # 创建动作空间
            self.data['actions'] = {}
            for k in self.agent_keys:
                if k in self.action_space:
                    if hasattr(self.action_space[k], 'n'):  # 离散动作空间
                        self.data['actions'][k] = np.zeros((self.n_envs, self.n_size), dtype=np.int32)
                    else:  # 连续动作空间
                        act_shape = space2shape(self.action_space[k])
                        self.data['actions'][k] = np.zeros((self.n_envs, self.n_size) + act_shape, dtype=np.float32)
                else:
                    print(f"警告: 智能体 {k} 不在动作空间中")
                    # 创建一个默认的动作空间
                    self.data['actions'][k] = np.zeros((self.n_envs, self.n_size), dtype=np.int32)
            
            # 创建其他必要的数据结构
            self.data['rewards'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['returns'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['values'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['advantages'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys}
            self.data['terminals'] = {k: np.zeros((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys}
            self.data['agent_mask'] = {k: np.ones((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys}
            
            # 如果使用全局状态，创建状态空间
            if self.store_global_state and self.state_space is not None:
                state_shape = self.state_space.shape if hasattr(self.state_space, 'shape') else (1,)
                self.data['state'] = np.zeros((self.n_envs, self.n_size) + state_shape, dtype=np.float32)
                print(f"创建状态空间: 形状={state_shape}")
            
            # 如果使用动作掩码，创建动作掩码空间
            if self.use_actions_mask and self.avail_actions_shape is not None:
                self.data['avail_actions'] = {}
                for k in self.agent_keys:
                    if k in self.avail_actions_shape:
                        self.data['avail_actions'][k] = np.ones((self.n_envs, self.n_size) + self.avail_actions_shape[k], dtype=np.bool_)
                    else:
                        print(f"警告: 智能体 {k} 不在动作掩码形状中")
                        # 创建一个默认的动作掩码
                        self.data['avail_actions'][k] = np.ones((self.n_envs, self.n_size, 1), dtype=np.bool_)
            
            # 重置指针和大小
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)  # 每个环境最后一个episode的起始索引
            
            print("COMA缓冲区初始化完成")
            
        except Exception as e:
            print(f"初始化COMA缓冲区时出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 回退到最基本的数据结构
            self.data = {
                'obs': {k: np.zeros((self.n_envs, self.n_size, 1), dtype=np.float32) for k in self.agent_keys},
                'actions': {k: np.zeros((self.n_envs, self.n_size), dtype=np.int32) for k in self.agent_keys},
                'rewards': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'returns': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'values': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'advantages': {k: np.zeros((self.n_envs, self.n_size), dtype=np.float32) for k in self.agent_keys},
                'terminals': {k: np.zeros((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys},
                'agent_mask': {k: np.ones((self.n_envs, self.n_size), dtype=np.bool_) for k in self.agent_keys},
            }
            if self.store_global_state:
                self.data['state'] = np.zeros((self.n_envs, self.n_size, 10), dtype=np.float32)  # 假设状态维度为10
            if self.use_actions_mask:
                self.data['avail_actions'] = {k: np.ones((self.n_envs, self.n_size, 5), dtype=np.bool_) for k in self.agent_keys}  # 假设动作数为5
            
            self.ptr, self.size = 0, 0
            self.start_ids = np.zeros(self.n_envs, np.int64)
            print("使用基本数据结构初始化COMA缓冲区")

    def finish_path(self, i_env, value_next=None, value_normalizer=None):  # when an episode is finished
        """特别适用于COMA算法的经验回放完成方法"""
        print(f"完成路径: i_env={i_env}")
        
        try:
            if self.ptr > self.start_ids[i_env]:
                path_slice = slice(self.start_ids[i_env], self.ptr)
                if value_next is None:
                    value_next = {k: 0.0 for k in self.agent_keys}
                
                for key in self.agent_keys:
                    if value_normalizer is not None:
                        key_vn = key if isinstance(value_normalizer, dict) else 0
                        values = value_normalizer[key_vn].denormalize(self.data['values'][key][i_env, path_slice])
                    else:
                        values = self.data['values'][key][i_env, path_slice]
                    
                    rewards = self.data['rewards'][key][i_env, path_slice]
                    terminals = 1.0 - self.data['terminals'][key][i_env, path_slice].astype(np.float32)
                    
                    # 确保value_next是标量
                    value_next_key = value_next[key] if isinstance(value_next, dict) and key in value_next else 0.0
                    
                    # 计算回报
                    returns = np.zeros_like(rewards)
                    gae = 0.0
                    for t in reversed(range(len(rewards))):
                        if t == len(rewards) - 1:
                            value_next_t = value_next_key
                        else:
                            value_next_t = values[t + 1]
                        delta = rewards[t] + self.gamma * value_next_t * terminals[t] - values[t]
                        gae = delta + self.gamma * self.gae_lambda * terminals[t] * gae
                        returns[t] = gae + values[t]
                    
                    self.data['advantages'][key][i_env, path_slice] = returns - values
                    self.data['returns'][key][i_env, path_slice] = returns
            
            # 更新环境的起始索引
            self.start_ids[i_env] = self.ptr
        
        except Exception as e:
            print(f"完成路径时出现错误: {str(e)}")
            import traceback
            traceback.print_exc()


class COMA_Buffer_RNN(MARL_OnPolicyBuffer_RNN):
    """
    Replay buffer for on-policy MARL algorithms.

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size of transition data for one environment.
        use_gae: whether to use GAE trick.
        use_advnorm: whether to use Advantage normalization trick.
        gamma: discount factor.
        gae_lam: gae lambda.
        **kwargs: other args.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, rew_space, done_space, n_envs, buffer_size,
                 use_gae, use_advnorm, gamma, gae_lam, **kwargs):
        self.td_lambda = kwargs['td_lambda']
        super(COMA_Buffer_RNN, self).__init__(n_agents, state_space, obs_space, act_space, rew_space, done_space,
                                              n_envs, buffer_size, use_gae, use_advnorm, gamma, gae_lam, **kwargs)

    def clear(self):
        self.data = {
            'obs': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len + 1) + self.obs_space, np.float32),
            'actions': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.act_space, np.float32),
            'actions_onehot': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len, self.dim_act)).astype(
                np.float32),
            'rewards': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'returns': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'values': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'advantages': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'log_pi_old': np.zeros((self.buffer_size, self.n_agents, self.max_eps_len,), np.float32),
            'terminals': np.zeros((self.buffer_size, self.max_eps_len) + self.done_space, np.bool_),
            'avail_actions': np.ones((self.buffer_size, self.n_agents, self.max_eps_len + 1, self.dim_act), np.bool_),
            'filled': np.zeros((self.buffer_size, self.max_eps_len, 1), np.bool_)
        }
        if self.state_space is not None:
            self.data.update({'state': np.zeros(
                (self.buffer_size, self.max_eps_len + 1) + self.state_space, np.float32)})
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        self.episode_data = {
            'obs': np.zeros((self.n_envs, self.n_agents, self.max_eps_len + 1) + self.obs_space, dtype=np.float32),
            'actions': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.act_space, dtype=np.float32),
            'actions_onehot': np.zeros((self.n_envs, self.n_agents, self.max_eps_len, self.dim_act), dtype=np.float32),
            'rewards': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, dtype=np.float32),
            'returns': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'values': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'advantages': np.zeros((self.n_envs, self.n_agents, self.max_eps_len) + self.rew_space, np.float32),
            'log_pi_old': np.zeros((self.n_envs, self.n_agents, self.max_eps_len,), np.float32),
            'terminals': np.zeros((self.n_envs, self.max_eps_len) + self.done_space, dtype=np.bool_),
            'avail_actions': np.ones((self.n_envs, self.n_agents, self.max_eps_len + 1, self.dim_act), dtype=np.bool_),
            'filled': np.zeros((self.n_envs, self.max_eps_len, 1), dtype=np.bool_),
        }
        if self.state_space is not None:
            self.episode_data.update({
                'state': np.zeros((self.n_envs, self.max_eps_len + 1) + self.state_space, dtype=np.float32)
            })
            self.episode_data.update({
                'state_next': np.zeros((self.n_envs, self.max_eps_len) + self.state_space, np.float32)
            })
        if self.use_actions_mask:
            self.episode_data.update({
                'avail_actions': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                      dtype=np.bool_) for k in self.agent_keys}
            })

    def store_transitions(self, t_envs, *transition_data):
        obs_n, actions_dict, state, rewards, terminated, avail_actions = transition_data
        self.episode_data['obs'][:, :, t_envs] = obs_n
        self.episode_data['actions'][:, :, t_envs] = actions_dict['actions_n']
        self.episode_data['actions_onehot'][:, :, t_envs] = actions_dict['act_n_onehot']
        self.episode_data['rewards'][:, :, t_envs] = rewards
        self.episode_data['values'][:, :, t_envs] = actions_dict['values']
        self.episode_data['log_pi_old'][:, :, t_envs] = actions_dict['log_pi']
        self.episode_data['terminals'][:, t_envs] = terminated
        self.episode_data['avail_actions'][:, :, t_envs] = avail_actions
        if self.state_space is not None:
            self.episode_data['state'][:, t_envs] = state

    def finish_path(self, i_env, next_t, *terminal_data, value_next=None, value_normalizer=None):
        obs_next, state_next, available_actions, filled = terminal_data
        self.episode_data['obs'][i_env, :, next_t] = obs_next[i_env]
        self.episode_data['state'][i_env, next_t] = state_next[i_env]
        self.episode_data['avail_actions'][i_env, :, next_t] = available_actions[i_env]
        self.episode_data['filled'][i_env] = filled[i_env]

        """
        when an episode is finished, build td-lambda targets.
        """
        if next_t > self.max_eps_len:
            path_slice = np.arange(0, self.max_eps_len).astype(np.int32)
        else:
            path_slice = np.arange(0, next_t).astype(np.int32)
        # calculate advantages and returns
        rewards = np.array(self.episode_data['rewards'][i_env, :, path_slice])
        vs = np.append(np.array(self.episode_data['values'][i_env, :, path_slice]),
                       [value_next.reshape(self.n_agents, 1)], axis=0)
        dones = np.array(self.episode_data['terminals'][i_env, path_slice])
        # 确保dones的形状正确
        if len(dones.shape) > 1:
            dones = dones.reshape(-1)
        returns = np.zeros_like(vs)
        step_nums = len(path_slice)

        for t in reversed(range(step_nums)):
            # 修复：确保正确处理多维数组
            gamma_term = self.td_lambda * self.gamma * returns[t + 1]
            reward_term = rewards[t]
            vs_term = (1 - self.td_lambda) * self.gamma * vs[t + 1] * (1 - dones[t])
            
            # 处理可能的数组到标量转换
            if isinstance(gamma_term, np.ndarray) and gamma_term.size == 1:
                gamma_term = float(gamma_term.item())
            if isinstance(reward_term, np.ndarray) and reward_term.size == 1:
                reward_term = float(reward_term.item())
            if isinstance(vs_term, np.ndarray) and vs_term.size == 1:
                vs_term = float(vs_term.item())
                
            returns[t] = reward_term + gamma_term
        self.episode_data['returns'][i_env, :, path_slice] = returns[:-1]
        self.store_episodes(i_env)


class MARL_OffPolicyBuffer(BaseBuffer):
    """
    Replay buffer for off-policy MARL algorithms.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        batch_size (int): Batch size of transition data for a sample.
        **kwargs: Other arguments.

    Example:
        >> state_space=None
        >> obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                      'agent_1': Box(-inf, inf, (18,), float32),
                      'agent_2': Box(-inf, inf, (18,), float32)},
        >> act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                      'agent_1': Box(0.0, 1.0, (5,), float32),
                      'agent_2': Box(0.0, 1.0, (5,), float32)},
        >> n_envs=50,
        >> buffer_size=10000,
        >> batch_size=256,
        >> agent_keys=['agent_0', 'agent_1', 'agent_2'],
        >> memory = MARL_OffPolicyBuffer(agent_keys=agent_keys, state_space=state_space,
                                         obs_space=obs_space, act_space=act_space,
                                         n_envs=n_envs, buffer_size=buffer_size, batch_size=batch_size)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 **kwargs):
        super(MARL_OffPolicyBuffer, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs, buffer_size)
        self.batch_size = batch_size
        self.store_global_state = False if self.state_space is None else True
        self.use_actions_mask = kwargs['use_actions_mask'] if 'use_actions_mask' in kwargs else False
        self.avail_actions_shape = kwargs['avail_actions_shape'] if 'avail_actions_shape' in kwargs else None
        self.data = {}
        self.clear()
        self.data_keys = self.data.keys()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (n_env=50, buffer_size=10000, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[50, 200, 18],
                            'agent_1': shape=[50, 200, 18],
                            'agent_2': shape=[50, 200, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[50, 200, 5],
                                'agent_1': shape=[50, 200, 5],
                                'agent_2': shape=[50, 200, 5]},  # dim_act: 5
                     ...}
        """
        reward_space = {key: () for key in self.agent_keys}
        terminal_space = {key: () for key in self.agent_keys}
        agent_mask_space = {key: () for key in self.agent_keys}

        self.data = {
            'obs': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'actions': create_memory(space2shape(self.act_space), self.n_envs, self.n_size),
            'obs_next': create_memory(space2shape(self.obs_space), self.n_envs, self.n_size),
            'rewards': create_memory(reward_space, self.n_envs, self.n_size),
            'terminals': create_memory(terminal_space, self.n_envs, self.n_size, np.bool_),
            'agent_mask': create_memory(agent_mask_space, self.n_envs, self.n_size, np.bool_),
        }
        if self.store_global_state:
            self.data.update({
                'state': create_memory(space2shape(self.state_space), self.n_envs, self.n_size),
                'state_next': create_memory(space2shape(self.state_space), self.n_envs, self.n_size)
            })
        if self.use_actions_mask:
            self.data.update({
                "avail_actions": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_),
                "avail_actions_next": create_memory(self.avail_actions_shape, self.n_envs, self.n_size, np.bool_)
            })
        self.ptr, self.size = 0, 0

    def store(self, **step_data):
        """ Stores a step of data into the replay buffer. """
        for data_key, data_values in step_data.items():
            if data_key in ['state', 'state_next']:
                self.data[data_key][:, self.ptr] = data_values
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][:, self.ptr] = data_values[agt_key]
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = np.min([self.size + 1, self.n_size])

    def sample(self, batch_size=None):
        """
        Samples a batch of data from the replay buffer.

        Parameters:
            batch_size (int): The size of the data batch, default is self.batch_size (recommended).

        Returns:
            samples_dict (dict): The sampled data.
        """
        assert self.size > 0, "Not enough transitions for off-policy buffer to random sample."
        if batch_size is None:
            batch_size = self.batch_size
        env_choices = np.random.choice(self.n_envs, batch_size)
        step_choices = np.random.choice(self.size, batch_size)
        samples_dict = {}
        for data_key in self.data_keys:
            if data_key in ['state', 'state_next']:
                samples_dict[data_key] = self.data[data_key][env_choices, step_choices]
                continue
            samples_dict[data_key] = {k: self.data[data_key][k][env_choices, step_choices] for k in self.agent_keys}
        samples_dict['batch_size'] = batch_size
        return samples_dict

    def finish_path(self, *args, **kwargs):
        return


class MARL_OffPolicyBuffer_RNN(MARL_OffPolicyBuffer):
    """
    Replay buffer for off-policy MARL algorithms with DRQN trick.

    Args:
        agent_keys (List[str]): Keys that identify each agent.
        state_space (Dict[str, Space]): Global state space, type: Discrete, Box.
        obs_space (Dict[str, Dict[str, Space]]): Observation space for one agent (suppose same obs space for group agents).
        act_space (Dict[str, Dict[str, Space]]): Action space for one agent (suppose same actions space for group agents).
        n_envs (int): Number of parallel environments.
        buffer_size (int): Buffer size of total experience data.
        batch_size (int): Batch size of episodes for a sample.
        max_episode_steps (int): The sequence length of each episode data.
        **kwargs: Other arguments.

    Example:
        $ state_space=None
        $ obs_space={'agent_0': Box(-inf, inf, (18,), float32),
                     'agent_1': Box(-inf, inf, (18,), float32),
                     'agent_2': Box(-inf, inf, (18,), float32)},
        $ act_space={'agent_0': Box(0.0, 1.0, (5,), float32),
                     'agent_1': Box(0.0, 1.0, (5,), float32),
                     'agent_2': Box(0.0, 1.0, (5,), float32)},
        $ n_envs=50,
        $ buffer_size=10000,
        $ batch_size=256,
        $ agent_keys=['agent_0', 'agent_1', 'agent_2'],
        $ max_episode_steps=60
        $ memory = MARL_OffPolicyBuffer_RNN(agent_keys=agent_keys, state_space=state_space,
                                            obs_space=obs_space, act_space=act_space,
                                            n_envs=n_envs, buffer_size=buffer_size, batch_size=batch_size,
                                            max_episode_steps=max_episode_steps,
                                            agent_keys=agent_keys)
    """

    def __init__(self,
                 agent_keys: List[str],
                 state_space: Dict[str, Space] = None,
                 obs_space: Dict[str, Dict[str, Space]] = None,
                 act_space: Dict[str, Dict[str, Space]] = None,
                 n_envs: int = 1,
                 buffer_size: int = 1,
                 batch_size: int = 1,
                 max_episode_steps: int = 1,
                 **kwargs):
        self.max_eps_len = max_episode_steps
        self.obs_shape = {k: space2shape(obs_space[k]) for k in agent_keys}
        self.act_shape = {k: space2shape(act_space[k]) for k in agent_keys}
        super(MARL_OffPolicyBuffer_RNN, self).__init__(agent_keys, state_space, obs_space, act_space, n_envs,
                                                       buffer_size, batch_size, **kwargs)
        self.episode_data = {}
        self.clear_episodes()

    def clear(self):
        """
        Clears the memory data in the replay buffer.

        Example:
        An example shows the data shape: (buffer_size=10000, max_eps_len=60,
                                          agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[10000, 61, 18],
                            'agent_1': shape=[10000, 61, 18],
                            'agent_2': shape=[10000, 61, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[10000, 60, 5],
                                'agent_1': shape=[10000, 60, 5],
                                'agent_2': shape=[10000, 60, 5]},  # dim_act: 5
                     ...
                     'filled': shape=[10000, 60],  # Step mask values. True means current step is not terminated.
                     }
        """
        self.data = {
            'obs': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.obs_shape[k], dtype=np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.buffer_size, self.max_eps_len) + self.act_shape[k], dtype=np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.buffer_size, self.max_eps_len), dtype=np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.buffer_size, self.max_eps_len + 1) + space2shape(self.state_space)
            self.data.update({'state': np.zeros(state_shape, dtype=np.float32)})
        if self.use_actions_mask:
            self.data.update({
                'avail_actions': {k: np.zeros((self.buffer_size, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                      dtype=np.bool_) for k in self.agent_keys}
            })
        self.ptr, self.size = 0, 0

    def clear_episodes(self):
        """
        Clears an episode of data for multiple environments in the replay buffer.

        Example:
        An example shows the data shape: (n_envs=16, max_eps_len=60, agent_keys=['agent_0', 'agent_1', 'agent_2']).
        self.data: {'obs': {'agent_0': shape=[16, 61, 18],
                            'agent_1': shape=[16, 61, 18],
                            'agent_2': shape=[16, 61, 18]},  # dim_obs: 18
                    'actions': {'agent_0': shape=[16, 60, 5],
                                'agent_1': shape=[16, 60, 5],
                                'agent_2': shape=[16, 60, 5]},  # dim_act: 5
                     ...
                     'filled': shape=[16, 60],  # Step mask values. True means current step is not terminated.
                     }
        """
        self.episode_data = {
            'obs': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.obs_shape[k], np.float32)
                    for k in self.agent_keys},
            'actions': {k: np.zeros((self.n_envs, self.max_eps_len) + self.act_shape[k], np.float32)
                        for k in self.agent_keys},
            'rewards': {k: np.zeros((self.n_envs, self.max_eps_len), np.float32) for k in self.agent_keys},
            'terminals': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'agent_mask': {k: np.zeros((self.n_envs, self.max_eps_len), np.bool_) for k in self.agent_keys},
            'filled': np.zeros((self.n_envs, self.max_eps_len), np.bool_),
        }

        if self.store_global_state:
            state_shape = (self.n_envs, self.max_eps_len + 1) + space2shape(self.state_space)
            self.episode_data.update({'state': np.zeros(state_shape, dtype=np.float32)})
            self.episode_data.update({
                'state_next': np.zeros((self.n_envs, self.max_eps_len) + self.state_space, np.float32)
            })
        if self.use_actions_mask:
            self.episode_data.update({
                'avail_actions': {k: np.zeros((self.n_envs, self.max_eps_len + 1) + self.avail_actions_shape[k],
                                      dtype=np.bool_) for k in self.agent_keys}
            })

    def store(self, **step_data):
        """
        Stores a step of data for each environment.

        Parameters:
            step_data (dict): A dict of step data that to be stored into self.episode_data.
        """
        envs_step = step_data['episode_steps']
        envs_choice = range(self.n_envs)
        self.episode_data["filled"][envs_choice, envs_step] = True
        for data_key, data_value in step_data.items():
            if data_key == "episode_steps":
                continue
            if data_key in ['state', 'state_next']:
                self.episode_data[data_key][envs_choice, envs_step] = data_value
                continue
            for agt_key in self.agent_keys:
                self.episode_data[data_key][agt_key][envs_choice, envs_step] = data_value[agt_key]

    def store_episodes(self, i_env):
        """
        Stores the episode of data for ith environment into the self.data.

        Parameters:
            i_env (int): The ith environment.
        """
        for data_key in self.data_keys:
            if data_key == "filled":
                self.data["filled"][self.ptr] = self.episode_data["filled"][i_env].copy()
                continue
            if data_key in ['state', 'state_next']:
                self.data[data_key][self.ptr] = self.episode_data[data_key][i_env].copy()
                continue
            for agt_key in self.agent_keys:
                self.data[data_key][agt_key][self.ptr] = self.episode_data[data_key][agt_key][i_env].copy()
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = np.min([self.size + 1, self.buffer_size])
        # clear the filled values for ith env.
        self.episode_data['filled'][i_env] = np.zeros(self.max_eps_len, dtype=np.bool_)

    def finish_path(self, i_env, **terminal_data):
        """
        Address the terminal states, including store the terminal observations, avail_actions, and others.

        Parameters:
            i_env (int): The i-th environment.
            terminal_data (dict): The terminal states.
        """
        env_step = terminal_data['episode_step']
        # Store terminal data into self.episode_data.
        if self.store_global_state:
            self.episode_data['state'][i_env, env_step] = terminal_data['state']
        for agt_key in self.agent_keys:
            self.episode_data['obs'][agt_key][i_env, env_step] = terminal_data['obs'][agt_key]
            if self.use_actions_mask:
                self.episode_data['avail_actions'][agt_key][i_env, env_step] = terminal_data['avail_actions'][agt_key]
        # Store the episode data of ith env into self.data.
        self.store_episodes(i_env)


class MeanField_OffPolicyBuffer(MARL_OffPolicyBuffer):
    """
    Replay buffer for off-policy Mean-Field MARL algorithms (Mean-Field Q-Learning).

    Args:
        n_agents: number of agents.
        state_space: global state space, type: Discrete, Box.
        obs_space: observation space for one agent (suppose same obs space for group agents).
        act_space: action space for one agent (suppose same actions space for group agents).
        prob_shape: the data shape of the action probabilities.
        rew_space: reward space.
        done_space: terminal variable space.
        n_envs: number of parallel environments.
        buffer_size: buffer size of total experience data.
        batch_size: batch size of transition data for a sample.
    """

    def __init__(self, n_agents, state_space, obs_space, act_space, prob_shape, rew_space, done_space,
                 n_envs, buffer_size, batch_size):
        self.prob_shape = prob_shape
        super(MeanField_OffPolicyBuffer, self).__init__(n_agents, state_space, obs_space, act_space, rew_space,
                                                        done_space, n_envs, buffer_size, batch_size)

    def clear(self):
        super(MeanField_OffPolicyBuffer, self).clear()
        self.data.update({"act_mean": np.zeros((self.n_envs, self.n_size,) + self.prob_shape).astype(np.float32)})

    def sample(self):
        env_choices = np.random.choice(self.n_envs, self.batch_size)
        step_choices = np.random.choice(self.size, self.batch_size)
        samples = {k: self.data[k][env_choices, step_choices] for k in self.keys}
        next_index = (step_choices + 1) % self.n_size
        samples.update({'act_mean_next': self.data['act_mean'][env_choices, next_index]})
        return samples
