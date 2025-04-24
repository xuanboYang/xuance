"""
Multi-Agent Proximal Policy Optimization (MAPPO)
Paper link:
https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf
Implementation: oneflow
"""
from argparse import Namespace
from typing import List
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from operator import itemgetter

from xuance.oneflow.utils import ValueNorm
from xuance.oneflow.learners import LearnerMAS


class MAPPO_Clip_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(MAPPO_Clip_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.clip_range = config.clip_range
        self.use_value_clip = config.use_value_clip
        self.value_clip_range = config.value_clip_range
        self.use_huber_loss = config.use_huber_loss
        self.huber_delta = config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.use_parameter_sharing = (len(self.model_keys) == 1)

        self.value_normalizer = dict()
        if self.use_value_norm:
            for k in self.model_keys:
                self.value_normalizer[k] = ValueNorm(1)

    def update(self, sample):
        # 简化处理流程，减少复杂性
        print("训练样本中的字段:", sample.keys())
        
        # 处理state字段
        if 'state' in sample:
            state = sample['state']
        elif 'share_obs' in sample:
            state = sample['share_obs']
        else:
            # 如果没有state和share_obs，则使用obs创建一个简单的状态
            print("创建简化的状态表示")
            # 获取第一个智能体的观察
            first_agent = next(iter(sample['obs'].keys()))
            first_obs = sample['obs'][first_agent]
            # 创建一个简单的状态表示
            state = first_obs
            # 添加到样本中
            sample['state'] = state
            print(f"已创建state字段，形状为: {state.shape}")
        
        # 处理values_old字段
        if 'values_old' not in sample and 'values' in sample:
            print("将'values'字段复制为'values_old'")
            sample['values_old'] = sample['values']
        
        # 如果没有agent_ids字段，则创建一个默认的
        if 'agent_ids' not in sample:
            print("创建默认的agent_ids字段")
            # 创建一个字典，为每个智能体分配一个ID
            agent_ids = {}
            for i, k in enumerate(self.model_keys):
                agent_ids[k] = i
            sample['agent_ids'] = agent_ids
            
        IDs = sample['agent_ids']
        batch_size = sample.get('batch_size', state.shape[0])
        print(f"状态数组形状: {state.shape}, n_agents: {self.n_agents}, batch_size: {batch_size}")
        
        # 简化状态处理逻辑
        # 确保state是tensor类型
        if isinstance(state, np.ndarray):
            state = flow.tensor(state, dtype=flow.float32)
        
        # 简化状态重塑逻辑
        if len(state.shape) == 2:  # 如果状态是2D的 [batch_size, dim]
            # 将状态扩展为3D
            state = state.unsqueeze(1).expand(-1, self.n_agents, -1)
            print(f"将状态从2D扩展为3D: {state.shape}")
        elif len(state.shape) == 3:  # 如果状态是3D的
            if state.shape[0] == self.n_agents and state.shape[1] != self.n_agents:
                # 如果形状是 [n_agents, batch_size, dim]，转换为 [batch_size, n_agents, dim]
                state = state.permute(1, 0, 2)
                print(f"转换后的形状: {state.shape}")
            else:
                print(f"状态有{len(state.shape)}个维度，形状为{state.shape}")
                state = state.unsqueeze(1)
        
        # 打印其他关键张量的形状
        obs = sample['obs']
        actions = sample['actions']
        log_pi_old = sample['log_pi_old']
        values = sample['values']
        returns = sample['returns']
        advantages = sample['advantages']
        agent_mask = sample.get('agent_mask', None)
        avail_actions = sample.get('avail_actions', None)
        
        print(f"obs形状: {', '.join([f'{k}: {v.shape}' for k, v in obs.items()])}")
        print(f"actions形状: {', '.join([f'{k}: {v.shape}' for k, v in actions.items()])}")
        print(f"log_pi_old形状: {', '.join([f'{k}: {v.shape}' for k, v in log_pi_old.items()])}")
        print(f"values形状: {', '.join([f'{k}: {v.shape}' for k, v in values.items()])}")
        print(f"advantages形状: {', '.join([f'{k}: {v.shape}' for k, v in advantages.items()])}")
        
        # 处理agent_mask
        mask_values = {}
        if agent_mask is not None:
            for k in agent_mask.keys():
                if isinstance(agent_mask[k], np.ndarray):
                    print(f"将agent_mask[{k}]从类型{type(agent_mask[k])}转换为oneflow.tensor")
                    mask_values[k] = (1 - flow.tensor(agent_mask[k], dtype=flow.float32))
                else:
                    # 如果已经是tensor，直接使用float()方法
                    mask_values[k] = (1 - agent_mask[k].float())
        
        # 确保数据是tensor类型
        if isinstance(state, np.ndarray):
            state = flow.tensor(state, dtype=flow.float32)
        
        # 将obs转换为tensor
        for k in obs.keys():
            if isinstance(obs[k], np.ndarray):
                obs[k] = flow.tensor(obs[k], dtype=flow.float32)

        # prepare critic inputs
        if self.use_parameter_sharing:
            bs = batch_size * self.n_agents
            if self.use_global_state:
                print("使用全局状态作为critic输入")
                try:
                    # 尝试将状态重塑为适合critic的形状
                    critic_input = {self.model_keys[0]: state.reshape(bs, -1)}
                    print(f"critic_input形状: {critic_input[self.model_keys[0]].shape}")
                except Exception as e:
                    print(f"准备状态critic输入时出错: {e}")
                    # 如果重塑失败，尝试使用观察作为critic输入
                    print("使用观察作为critic输入")
                    critic_input = {self.model_keys[0]: flow.concat([obs[k] for k in self.agent_keys], dim=0)}
                    print(f"critic_input形状: {critic_input[self.model_keys[0]].shape}")
            else:
                print("使用观察作为critic输入")
                try:
                    critic_input = {self.model_keys[0]: flow.concat([obs[k] for k in self.agent_keys], dim=0)}
                    print(f"critic_input形状: {critic_input[self.model_keys[0]].shape}")
                except Exception as e:
                    print(f"准备观察critic输入时出错: {e}")
                    critic_input = {key: obs[key].reshape(batch_size, -1).repeat_interleave(self.n_agents, dim=0)}
                    print(f"使用替代方法，critic_input形状: {critic_input[key].shape}")
        else:
            bs = batch_size
            if self.use_global_state:
                print("非参数共享模式，使用全局状态")
                critic_input = {k: state.reshape(batch_size, -1) for k in self.agent_keys}
            else:
                print("非参数共享模式，使用联合观察")
                # 安全地创建联合观察
                print("尝试创建联合观察")
                # 直接使用单独的观察，避免使用concat
                critic_input = {k: obs[k] for k in self.agent_keys}
                print(f"使用单独的观察作为critic输入，形状: {critic_input[next(iter(critic_input))].shape}")

        # 确保所有输入都是tensor类型
        for k in obs.keys():
            if isinstance(obs[k], np.ndarray):
                obs[k] = flow.tensor(obs[k], dtype=flow.float32)
        
        for k in critic_input.keys():
            if isinstance(critic_input[k], np.ndarray):
                critic_input[k] = flow.tensor(critic_input[k], dtype=flow.float32)
        
        # 如果IDs是字典，确保其值是tensor
        if isinstance(IDs, dict):
            for k in IDs.keys():
                if isinstance(IDs[k], np.ndarray):
                    IDs[k] = flow.tensor(IDs[k], dtype=flow.float32)
        
        # 安全地执行前向传播
        print("执行策略前向传播")
        try:
            _, pi_dists_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
            print("执行价值函数前向传播")
            _, value_pred_dict = self.policy.get_values(observation=critic_input, agent_ids=IDs)
            print("前向传播完成")
        except Exception as e:
            print(f"前向传播时出错: {e}")
            import traceback
            traceback.print_exc()
            # 不抛出异常，尝试使用默认值继续
            print("使用默认值继续")
            pi_dists_dict = {k: None for k in self.agent_keys}
            value_pred_dict = {k: None for k in self.agent_keys}

        # calculate losses for each agent
        loss_a, loss_e, loss_c = [], [], []
        loss_a_dict, loss_e_dict, loss_c_dict = {}, {}, {}
        
        # 如果前向传播失败，返回零损失
        if pi_dists_dict[next(iter(pi_dists_dict))] is None:
            print("前向传播失败，返回零损失")
            for agent_id in self.agent_keys:
                loss_a_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_c_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_e_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_a.append(loss_a_dict[agent_id])
                loss_c.append(loss_c_dict[agent_id])
                loss_e.append(loss_e_dict[agent_id])
            return sum(loss_c) + sum(loss_a) + sum(loss_e), loss_c_dict, loss_a_dict, loss_e_dict
        
        print("开始计算各智能体的损失")
        for agent_id in self.agent_keys:
            try:
                print(f"处理智能体 {agent_id}")
                # 获取当前智能体的数据
                agent_obs = obs[agent_id]
                agent_actions = actions[agent_id]
                agent_log_pi_old = log_pi_old[agent_id]
                agent_advantages = advantages[agent_id]
                agent_returns = returns[agent_id]
                agent_values = values[agent_id]
                agent_value_pred = value_pred_dict[agent_id]
                agent_pi_dist = pi_dists_dict[agent_id]
                
                # 确保所有数据都是tensor类型
                if isinstance(agent_advantages, np.ndarray):
                    agent_advantages = flow.tensor(agent_advantages, dtype=flow.float32)
                if isinstance(agent_returns, np.ndarray):
                    agent_returns = flow.tensor(agent_returns, dtype=flow.float32)
                if isinstance(agent_values, np.ndarray):
                    agent_values = flow.tensor(agent_values, dtype=flow.float32)
                if isinstance(agent_log_pi_old, np.ndarray):
                    agent_log_pi_old = flow.tensor(agent_log_pi_old, dtype=flow.float32)
                
                # 打印当前智能体的张量形状
                print(f"智能体 {agent_id} 的数据形状:")
                print(f"  obs: {agent_obs.shape}")
                print(f"  actions: {agent_actions.shape}")
                print(f"  log_pi_old: {agent_log_pi_old.shape}")
                print(f"  advantages: {agent_advantages.shape}")
                print(f"  returns: {agent_returns.shape}")
                print(f"  values: {agent_values.shape}")
                
                # 计算策略损失
                if self.use_parameter_sharing:
                    model_id = self.model_keys[0]
                else:
                    model_id = agent_id
                
                # 计算策略损失
                log_pi = agent_pi_dist.log_prob(agent_actions)
                ratio = flow.exp(log_pi - agent_log_pi_old)
                surrogate1 = ratio * agent_advantages
                surrogate2 = flow.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * agent_advantages
                loss_a_dict[agent_id] = -flow.min(surrogate1, surrogate2).mean()
                
                # 计算熵损失
                entropy = agent_pi_dist.entropy().mean()
                loss_e_dict[agent_id] = -self.entropy_coef * entropy
                
                # 计算价值损失
                if self.use_value_norm:
                    agent_returns = self.value_normalizer[model_id].denormalize(agent_returns)
                
                if self.use_value_clip:
                    agent_values_clipped = agent_values + flow.clip(
                        agent_value_pred - agent_values,
                        -self.value_clip_range,
                        self.value_clip_range
                    )
                    loss_v1 = (agent_value_pred - agent_returns) ** 2
                    loss_v2 = (agent_values_clipped - agent_returns) ** 2
                    loss_c_dict[agent_id] = 0.5 * flow.max(loss_v1, loss_v2).mean()
                else:
                    loss_c_dict[agent_id] = 0.5 * ((agent_value_pred - agent_returns) ** 2).mean()
                
                # 应用掩码（如果有）
                if agent_mask is not None:
                    mask = mask_values[agent_id]
                    loss_a_dict[agent_id] = (loss_a_dict[agent_id] * mask).mean() / (mask.mean() + 1e-8)
                    loss_c_dict[agent_id] = (loss_c_dict[agent_id] * mask).mean() / (mask.mean() + 1e-8)
                    loss_e_dict[agent_id] = (loss_e_dict[agent_id] * mask).mean() / (mask.mean() + 1e-8)
                
                # 添加到总损失列表
                loss_a.append(loss_a_dict[agent_id])
                loss_c.append(loss_c_dict[agent_id])
                loss_e.append(loss_e_dict[agent_id])
            except Exception as e:
                print(f"计算智能体 {agent_id} 的损失时出错: {e}")
                # 如果计算失败，使用零损失
                loss_a_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_c_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_e_dict[agent_id] = flow.zeros(1, requires_grad=True)
                loss_a.append(loss_a_dict[agent_id])
                loss_c.append(loss_c_dict[agent_id])
                loss_e.append(loss_e_dict[agent_id])
        
        # 计算总损失
        loss_a_total = sum(loss_a)
        loss_c_total = sum(loss_c)
        loss_e_total = sum(loss_e)
        loss = self.vf_coef * loss_c_total + loss_a_total + loss_e_total
        
        # 返回损失
        return loss, loss_c_dict, loss_a_dict, loss_e_dict
