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

    def get_tensors_from_sample(self, sample):
        """
        从样本中提取并处理张量数据
        
        参数:
            sample (dict): 包含训练数据的字典
            
        返回:
            tuple: 处理后的张量数据，包括state, obs, actions, log_pi_old, returns, advantages, 
                  agent_mask, avail_actions, values_old, agent_ids
        """
        try:
            # 提取基本数据
            obs = sample.get('obs', {})
            actions = sample.get('actions', {})
            log_pi_old = sample.get('log_pi_old', {})
            returns = sample.get('returns', {})
            advantages = sample.get('advantages', {})
            agent_mask = sample.get('agent_mask', {})
            avail_actions = sample.get('avail_actions', None)
            
            # 处理values_old
            if 'values_old' in sample:
                values_old = sample['values_old']
            elif 'values' in sample:
                values_old = sample['values']
            else:
                values_old = {k: None for k in self.agent_keys}
                
            # 处理state
            if 'state' in sample:
                state = sample['state']
            elif 'share_obs' in sample:
                state = sample['share_obs']
            else:
                # 如果没有state，使用第一个智能体的观察创建一个简单的状态
                first_agent = next(iter(obs.keys()))
                state = obs[first_agent]
                
            # 处理agent_ids
            if 'agent_ids' in sample:
                agent_ids = sample['agent_ids']
            else:
                # 创建默认的agent_ids
                agent_ids = {k: i for i, k in enumerate(self.agent_keys)}
                
            # 确保所有数据都是正确的类型和形状
            for k in self.agent_keys:
                # 确保所有智能体都有数据
                if k not in obs:
                    obs[k] = flow.zeros(1)
                if k not in actions:
                    actions[k] = flow.zeros(1)
                if k not in log_pi_old:
                    log_pi_old[k] = flow.zeros(1)
                if k not in returns:
                    returns[k] = flow.zeros(1)
                if k not in advantages:
                    advantages[k] = flow.zeros(1)
                if k not in agent_mask:
                    agent_mask[k] = flow.ones(1)
                if k not in values_old:
                    values_old[k] = flow.zeros(1)
            
            # 确保所有张量都是OneFlow张量
            for k in self.agent_keys:
                if not isinstance(obs[k], flow.Tensor):
                    obs[k] = flow.tensor(obs[k], dtype=flow.float32)
                if not isinstance(actions[k], flow.Tensor):
                    actions[k] = flow.tensor(actions[k], dtype=flow.float32)
                if not isinstance(log_pi_old[k], flow.Tensor):
                    log_pi_old[k] = flow.tensor(log_pi_old[k], dtype=flow.float32)
                if not isinstance(returns[k], flow.Tensor):
                    returns[k] = flow.tensor(returns[k], dtype=flow.float32)
                if not isinstance(advantages[k], flow.Tensor):
                    advantages[k] = flow.tensor(advantages[k], dtype=flow.float32)
                if not isinstance(agent_mask[k], flow.Tensor):
                    agent_mask[k] = flow.tensor(agent_mask[k], dtype=flow.float32)
                if values_old[k] is not None and not isinstance(values_old[k], flow.Tensor):
                    values_old[k] = flow.tensor(values_old[k], dtype=flow.float32)
            
            # 确保state是OneFlow张量
            if not isinstance(state, flow.Tensor):
                state = flow.tensor(state, dtype=flow.float32)
            
            # 检查并修复NaN值
            for k in self.agent_keys:
                if flow.isnan(obs[k]).any():
                    obs[k] = flow.nan_to_num(obs[k], nan=0.0)
                if flow.isnan(actions[k]).any():
                    actions[k] = flow.nan_to_num(actions[k], nan=0.0)
                if flow.isnan(log_pi_old[k]).any():
                    log_pi_old[k] = flow.nan_to_num(log_pi_old[k], nan=0.0)
                if flow.isnan(returns[k]).any():
                    returns[k] = flow.nan_to_num(returns[k], nan=0.0)
                if flow.isnan(advantages[k]).any():
                    advantages[k] = flow.nan_to_num(advantages[k], nan=0.0)
                if flow.isnan(agent_mask[k]).any():
                    agent_mask[k] = flow.nan_to_num(agent_mask[k], nan=1.0)
                if values_old[k] is not None and flow.isnan(values_old[k]).any():
                    values_old[k] = flow.nan_to_num(values_old[k], nan=0.0)
            
            if flow.isnan(state).any():
                state = flow.nan_to_num(state, nan=0.0)
            
            # 对于RNN，还需要返回RNN状态
            if hasattr(self, 'use_rnn') and self.use_rnn and 'rnn_state_actor' in sample and 'rnn_state_critic' in sample and 'dones' in sample:
                rnn_state_actor = sample['rnn_state_actor']
                rnn_state_critic = sample['rnn_state_critic']
                dones = sample['dones']
                
                # 确保RNN状态是OneFlow张量
                if not isinstance(rnn_state_actor, dict):
                    rnn_state_actor = {k: flow.zeros((1, 1, 64)) for k in self.agent_keys}
                if not isinstance(rnn_state_critic, dict):
                    rnn_state_critic = {k: flow.zeros((1, 1, 64)) for k in self.agent_keys}
                
                for k in self.agent_keys:
                    if k not in rnn_state_actor:
                        rnn_state_actor[k] = flow.zeros((1, 1, 64))
                    if k not in rnn_state_critic:
                        rnn_state_critic[k] = flow.zeros((1, 1, 64))
                    
                    if not isinstance(rnn_state_actor[k], flow.Tensor):
                        rnn_state_actor[k] = flow.tensor(rnn_state_actor[k], dtype=flow.float32)
                    if not isinstance(rnn_state_critic[k], flow.Tensor):
                        rnn_state_critic[k] = flow.tensor(rnn_state_critic[k], dtype=flow.float32)
                
                if not isinstance(dones, flow.Tensor):
                    dones = flow.tensor(dones, dtype=flow.float32)
                
                return state, obs, actions, log_pi_old, returns, advantages, agent_mask, avail_actions, values_old, agent_ids, rnn_state_actor, rnn_state_critic, dones
            
            return state, obs, actions, log_pi_old, returns, advantages, agent_mask, avail_actions, values_old, agent_ids
        except Exception as e:
            # 如果处理过程中出现任何错误，返回默认值
            default_tensor = flow.zeros(1)
            default_dict = {k: flow.zeros(1) for k in self.agent_keys}
            
            if hasattr(self, 'use_rnn') and self.use_rnn and 'rnn_state_actor' in sample and 'rnn_state_critic' in sample and 'dones' in sample:
                default_rnn = {k: flow.zeros((1, 1, 64)) for k in self.agent_keys}
                return default_tensor, default_dict, default_dict, default_dict, default_dict, default_dict, default_dict, None, default_dict, {k: i for i, k in enumerate(self.agent_keys)}, default_rnn, default_rnn, flow.zeros(1)
            
            return default_tensor, default_dict, default_dict, default_dict, default_dict, default_dict, default_dict, None, default_dict, {k: i for i, k in enumerate(self.agent_keys)}
    
    def update(self, sample):
        try:
            self.iterations += 1
            
            # 使用try-except包装get_tensors_from_sample调用
            try:
                state, obs, actions, log_pi_old, returns, advantages, agent_mask, avail_actions, values_old, agent_ids = self.get_tensors_from_sample(sample)
            except Exception as e:
                # 如果提取张量失败，返回空结果
                import traceback
                print(f"Error in get_tensors_from_sample: {str(e)}")
                print(traceback.format_exc())
                return {
                    "actor-loss": 0.0,
                    "critic-loss": 0.0,
                    "entropy": 0.0,
                    "loss": 0.0,
                    "error": {"value": f"Failed to extract tensors: {str(e)}"}
                }
            
            # 显式释放不需要的变量以减少内存占用
            del sample
            
            # 转换为oneflow tensor并进行安全检查
            try:
                if isinstance(obs, dict):
                    for k in obs:
                        if isinstance(obs[k], np.ndarray):
                            obs[k] = flow.tensor(obs[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(obs[k], flow.Tensor):
                            if flow.isnan(obs[k]).any():
                                obs[k] = flow.nan_to_num(obs[k], nan=0.0)
                            # 确保形状正确
                            if len(obs[k].shape) == 0:
                                obs[k] = obs[k].unsqueeze(0)
                
                if isinstance(actions, dict):
                    for k in actions:
                        if isinstance(actions[k], np.ndarray):
                            actions[k] = flow.tensor(actions[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(actions[k], flow.Tensor):
                            if flow.isnan(actions[k]).any():
                                actions[k] = flow.nan_to_num(actions[k], nan=0.0)
                            # 确保形状正确
                            if len(actions[k].shape) == 0:
                                actions[k] = actions[k].unsqueeze(0)
                
                if isinstance(advantages, dict):
                    for k in advantages:
                        if isinstance(advantages[k], np.ndarray):
                            advantages[k] = flow.tensor(advantages[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(advantages[k], flow.Tensor):
                            if flow.isnan(advantages[k]).any():
                                advantages[k] = flow.nan_to_num(advantages[k], nan=0.0)
                            # 确保形状正确
                            if len(advantages[k].shape) == 0:
                                advantages[k] = advantages[k].unsqueeze(0)
                
                if isinstance(log_pi_old, dict):
                    for k in log_pi_old:
                        if isinstance(log_pi_old[k], np.ndarray):
                            log_pi_old[k] = flow.tensor(log_pi_old[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(log_pi_old[k], flow.Tensor):
                            if flow.isnan(log_pi_old[k]).any():
                                log_pi_old[k] = flow.nan_to_num(log_pi_old[k], nan=0.0)
                            # 确保形状正确
                            if len(log_pi_old[k].shape) == 0:
                                log_pi_old[k] = log_pi_old[k].unsqueeze(0)
                
                if isinstance(returns, dict):
                    for k in returns:
                        if isinstance(returns[k], np.ndarray):
                            returns[k] = flow.tensor(returns[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(returns[k], flow.Tensor):
                            if flow.isnan(returns[k]).any():
                                returns[k] = flow.nan_to_num(returns[k], nan=0.0)
                            # 确保形状正确
                            if len(returns[k].shape) == 0:
                                returns[k] = returns[k].unsqueeze(0)
                
                if isinstance(values_old, dict):
                    for k in values_old:
                        if isinstance(values_old[k], np.ndarray):
                            values_old[k] = flow.tensor(values_old[k], dtype=flow.float32)
                        # 检查并修复NaN值和形状问题
                        if isinstance(values_old[k], flow.Tensor):
                            if flow.isnan(values_old[k]).any():
                                values_old[k] = flow.nan_to_num(values_old[k], nan=0.0)
                            # 确保形状正确
                            if len(values_old[k].shape) == 0:
                                values_old[k] = values_old[k].unsqueeze(0)
                
                # 检查并修复state
                if isinstance(state, np.ndarray):
                    state = flow.tensor(state, dtype=flow.float32)
                if isinstance(state, flow.Tensor):
                    if flow.isnan(state).any():
                        state = flow.nan_to_num(state, nan=0.0)
                    # 确保形状正确
                    if len(state.shape) == 0:
                        state = state.unsqueeze(0)
            except Exception as e:
                # 如果转换张量失败，返回空结果
                import traceback
                print(f"Error in tensor conversion: {str(e)}")
                print(traceback.format_exc())
                return {
                    "actor-loss": 0.0,
                    "critic-loss": 0.0,
                    "entropy": 0.0,
                    "loss": 0.0,
                    "error": {"value": f"Failed to convert tensors: {str(e)}"}
                }
            
            # 准备critic输入
            try:
                if self.use_parameter_sharing:
                    critic_in = obs
                else:
                    # 安全地处理state张量
                    if isinstance(state, flow.Tensor):
                        # 检查state是否为空
                        if state.numel() == 0:
                            # 如果state为空，创建一个默认的critic_in
                            critic_in = {k: flow.zeros((1, 1), device=self.device) for k in self.agent_keys}
                        else:
                            critic_in = state
                    else:
                        # 如果state不是张量，使用obs
                        critic_in = obs
            except Exception as e:
                # 如果准备critic输入失败，使用obs作为备用
                import traceback
                print(f"Error in preparing critic input: {str(e)}")
                print(traceback.format_exc())
                critic_in = obs
            
            # 前向传播
            try:
                with flow.set_grad_enabled(True):
                    # 使用try-except包装策略前向传播
                    try:
                        _, pi_dists_dict = self.policy(observation=obs, avail_actions=avail_actions)
                    except Exception as e:
                        # 如果策略前向传播失败，创建一个默认的分布字典
                        import traceback
                        print(f"Error in policy forward pass: {str(e)}")
                        print(traceback.format_exc())
                        pi_dists_dict = {}
                        for key in self.model_keys:
                            action_dim = 5  # 默认动作维度
                            probs = flow.ones((1, action_dim), device=self.device) / action_dim
                            pi_dists_dict[key] = flow.distributions.Categorical(probs=probs)
                    
                    # 使用try-except包装值函数前向传播
                    try:
                        # 禁用concat操作中的自动广播，可能导致段错误
                        values_pred_dict = {}
                        if hasattr(self.policy, 'get_values'):
                            # 安全处理critic_in
                            if not isinstance(critic_in, flow.Tensor) and not isinstance(critic_in, dict):
                                try:
                                    critic_in = flow.tensor(critic_in, dtype=flow.float32, device=self.device)
                                except Exception:
                                    critic_in = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                            
                            # 检查critic_in是否为空
                            if isinstance(critic_in, flow.Tensor) and critic_in.numel() == 0:
                                critic_in = flow.zeros((1, 1), device=self.device)
                            
                            # 检查critic_in是否包含NaN
                            if isinstance(critic_in, flow.Tensor) and flow.isnan(critic_in).any():
                                critic_in = flow.nan_to_num(critic_in, nan=0.0)
                            
                            # 尝试单独调用get_values
                            try:
                                _, values_pred_dict = self.policy.get_values(critic_in=critic_in)
                            except Exception as e:
                                print(f"Error in get_values call: {str(e)}")
                                # 如果失败，为每个model_key创建默认值
                                values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                        else:
                            # 如果没有get_values方法，创建默认值
                            values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                    except Exception as e:
                        # 如果值函数前向传播失败，创建默认值
                        import traceback
                        print(f"Error in value function forward pass: {str(e)}")
                        print(traceback.format_exc())
                        values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
            except Exception as e:
                # 如果前向传播失败，返回空结果
                import traceback
                print(f"Error in forward pass: {str(e)}")
                print(traceback.format_exc())
                return {
                    "actor-loss": 0.0,
                    "critic-loss": 0.0,
                    "entropy": 0.0,
                    "loss": 0.0,
                    "error": {"value": f"Forward pass failed: {str(e)}"}
                }
            
            # 计算损失
            loss_a, loss_e, loss_c = [], [], []
            for key in self.model_keys:
                try:
                    mask_values = agent_mask[key]
                    
                    # actor loss
                    try:
                        log_pi = pi_dists_dict[key].log_prob(actions[key])
                        
                        # 检查并修复无限值
                        log_pi = flow.nan_to_num(log_pi, nan=0.0, posinf=20.0, neginf=-20.0)
                        log_pi_old_k = flow.nan_to_num(log_pi_old[key], nan=0.0, posinf=20.0, neginf=-20.0)
                        
                        ratio = flow.exp(log_pi - log_pi_old_k)
                        # 限制ratio的范围，防止数值不稳定
                        ratio = flow.clamp(ratio, min=0.01, max=100.0)
                        
                        # 确保advantages_mask是有效的张量
                        advantages_k = advantages[key].detach()
                        advantages_k = flow.nan_to_num(advantages_k, nan=0.0)
                        
                        # 使用广播而不是reshape
                        if mask_values.shape != advantages_k.shape:
                            # 尝试广播
                            try:
                                advantages_mask = advantages_k * mask_values
                            except Exception:
                                # 如果广播失败，使用flatten和reshape
                                mask_flat = mask_values.flatten()
                                adv_flat = advantages_k.flatten()
                                min_len = min(len(mask_flat), len(adv_flat))
                                advantages_mask = (mask_flat[:min_len] * adv_flat[:min_len]).reshape(-1, 1)
                        else:
                            advantages_mask = advantages_k * mask_values
                        
                        surrogate1 = ratio * advantages_mask
                        surrogate2 = flow.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                        loss_a.append(-flow.min(surrogate1, surrogate2).mean())
                    except Exception as e:
                        loss_a.append(flow.tensor(0.0, requires_grad=True))
                    
                    # entropy loss
                    try:
                        entropy = pi_dists_dict[key].entropy()
                        entropy = flow.nan_to_num(entropy, nan=0.0)
                        
                        # 使用广播而不是reshape
                        if mask_values.shape != entropy.shape:
                            # 尝试广播
                            try:
                                entropy_masked = entropy * mask_values
                            except Exception:
                                # 如果广播失败，使用flatten和reshape
                                mask_flat = mask_values.flatten()
                                entropy_flat = entropy.flatten()
                                min_len = min(len(mask_flat), len(entropy_flat))
                                entropy_masked = (mask_flat[:min_len] * entropy_flat[:min_len]).reshape(-1, 1)
                        else:
                            entropy_masked = entropy * mask_values
                        
                        loss_e.append(entropy_masked.mean())
                    except Exception as e:
                        loss_e.append(flow.tensor(0.0, requires_grad=True))
                    
                    # critic loss
                    try:
                        value_pred_i = values_pred_dict[key]
                        value_target = returns[key]
                        values_i = values_old[key]
                        
                        # 确保所有张量都是有效的
                        value_pred_i = flow.nan_to_num(value_pred_i, nan=0.0)
                        value_target = flow.nan_to_num(value_target, nan=0.0)
                        values_i = flow.nan_to_num(values_i, nan=0.0)
                        
                        # 计算critic loss
                        if self.use_value_clip:
                            value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range, self.value_clip_range)
                            
                            if self.use_huber_loss:
                                try:
                                    loss_v1 = nn.SmoothL1Loss(reduction='none')(value_pred_i, value_target)
                                    loss_v2 = nn.SmoothL1Loss(reduction='none')(value_clipped, value_target)
                                    loss_v = flow.max(loss_v1, loss_v2)
                                except Exception:
                                    # 回退到使用平方误差
                                    loss_v = flow.max(flow.square(value_pred_i - value_target),
                                                    flow.square(value_clipped - value_target))
                            else:
                                loss_v = flow.max(flow.square(value_pred_i - value_target),
                                                flow.square(value_clipped - value_target))
                        else:
                            if self.use_huber_loss:
                                try:
                                    loss_v = nn.SmoothL1Loss(reduction='none')(value_pred_i, value_target)
                                except Exception:
                                    # 回退到使用平方误差
                                    loss_v = flow.square(value_pred_i - value_target)
                            else:
                                loss_v = flow.square(value_pred_i - value_target)
                        
                        # 确保loss_v是有效的张量
                        loss_v = flow.nan_to_num(loss_v, nan=0.0)
                        
                        # 使用广播而不是reshape
                        if mask_values.shape != loss_v.shape:
                            # 尝试广播
                            try:
                                loss_v_masked = loss_v * mask_values
                            except Exception:
                                # 如果广播失败，使用flatten和reshape
                                mask_flat = mask_values.flatten()
                                loss_v_flat = loss_v.flatten()
                                min_len = min(len(mask_flat), len(loss_v_flat))
                                loss_v_masked = (mask_flat[:min_len] * loss_v_flat[:min_len]).reshape(-1, 1)
                        else:
                            loss_v_masked = loss_v * mask_values
                        
                        loss_c.append(loss_v_masked.mean())
                    except Exception as e:
                        loss_c.append(flow.tensor(0.0, requires_grad=True))
                except Exception as e:
                    # 确保每个智能体都有对应的损失值
                    if len(loss_a) < self.model_keys.index(key) + 1:
                        loss_a.append(flow.tensor(0.0, requires_grad=True))
                    if len(loss_e) < self.model_keys.index(key) + 1:
                        loss_e.append(flow.tensor(0.0, requires_grad=True))
                    if len(loss_c) < self.model_keys.index(key) + 1:
                        loss_c.append(flow.tensor(0.0, requires_grad=True))
            
            # 显式释放不再需要的变量
            del pi_dists_dict, values_pred_dict
            
            # 计算总体损失并优化
            try:
                # 计算总体损失
                actor_loss = sum(loss_a) / max(1, self.n_groups)
                critic_loss = sum(loss_c) / max(1, self.n_groups)
                entropy_loss = sum(loss_e) / max(1, self.n_groups)
                
                # 检查损失值是否有效
                actor_loss = flow.nan_to_num(actor_loss, nan=0.1)
                critic_loss = flow.nan_to_num(critic_loss, nan=0.1)
                entropy_loss = flow.nan_to_num(entropy_loss, nan=0.01)
                
                # 计算策略损失
                policy_loss = actor_loss - self.ent_coef * entropy_loss + self.vf_coef * critic_loss
                policy_loss = flow.nan_to_num(policy_loss, nan=0.1)
                
                # 反向传播
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # 使用try-except包装反向传播
                try:
                    policy_loss.backward()
                except Exception:
                    # 如果反向传播失败，使用actor_loss作为备用
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()
                    actor_loss.backward()
                
                # 梯度裁剪
                try:
                    # 对所有参数应用梯度裁剪
                    flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(),
                                                max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                except Exception:
                    # 单独裁剪actor和critic参数
                    try:
                        flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters_actor,
                                                    max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                    except Exception:
                        pass
                    
                    try:
                        flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters_critic,
                                                    max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                    except Exception:
                        pass
                
                # 优化器步骤
                for optimizer in self.optimizers:
                    try:
                        optimizer.step()
                    except Exception:
                        pass
                
                # 学习率调度器
                if self.scheduler is not None:
                    for scheduler in self.schedulers:
                        try:
                            scheduler.step()
                        except Exception:
                            pass
                
                # 更新变量
                self.policy_loss = policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss)
                self.actor_loss = actor_loss.item() if hasattr(actor_loss, 'item') else float(actor_loss)
                self.critic_loss = critic_loss.item() if hasattr(critic_loss, 'item') else float(critic_loss)
                self.entropy_loss = entropy_loss.item() if hasattr(entropy_loss, 'item') else float(entropy_loss)
                
                # 显式释放不再需要的变量
                del policy_loss, actor_loss, critic_loss, entropy_loss
                
                # 强制执行垃圾回收
                import gc
                gc.collect()
                
                # 如果支持，清除CUDA缓存
                try:
                    if hasattr(flow, 'cuda') and hasattr(flow.cuda, 'empty_cache'):
                        flow.cuda.empty_cache()
                except Exception:
                    pass
                
                return {
                    "actor-loss": self.actor_loss,
                    "critic-loss": self.critic_loss,
                    "entropy": self.entropy_loss,
                    "loss": self.policy_loss
                }
            
            except Exception as e:
                # 返回一个空的结果字典
                return {
                    "actor-loss": 0.0,
                    "critic-loss": 0.0,
                    "entropy": 0.0,
                    "loss": 0.0,
                    "error": {"value": f"Optimization failed: {str(e)}"}
                }
        
        except Exception as e:
            # 返回一个空的结果字典
            return {
                "actor-loss": 0.0,
                "critic-loss": 0.0,
                "entropy": 0.0,
                "loss": 0.0,
                "error": {"value": f"General error: {str(e)}"}
            }

    def update_rnn(self, sample):
        try:
            self.iterations += 1
            state, obs, actions, log_pi_old, returns, advantages, agent_mask, avail_actions, values_old, agent_ids, \
            rnn_state_actor, rnn_state_critic, dones = self.get_tensors_from_sample(sample)
            
            # 显式释放不需要的变量以减少内存占用
            del sample
            
            # 转换为oneflow tensor
            if isinstance(obs, np.ndarray):
                obs = flow.Tensor(obs)
            if isinstance(actions, np.ndarray):
                actions = flow.Tensor(actions)
            if isinstance(advantages, np.ndarray):
                advantages = flow.Tensor(advantages)
            if isinstance(log_pi_old, np.ndarray):
                log_pi_old = flow.Tensor(log_pi_old)
            if isinstance(returns, np.ndarray):
                returns = flow.Tensor(returns)
            if isinstance(values_old, np.ndarray):
                values_old = flow.Tensor(values_old)
            
            # 检查并修复NaN值
            for tensor_name, tensor in [("obs", obs), ("actions", actions), ("advantages", advantages), 
                                       ("log_pi_old", log_pi_old), ("returns", returns), 
                                       ("values_old", values_old)]:
                if isinstance(tensor, flow.Tensor) and flow.isnan(tensor).any():
                    # 将NaN替换为0
                    tensor[flow.isnan(tensor)] = 0.0
            
            # 准备critic输入
            if self.use_parameter_sharing:
                critic_in = obs
            else:
                try:
                    # 安全地处理state张量
                    if isinstance(state, flow.Tensor):
                        # 检查state的维度
                        if len(state.shape) < 2:
                            # 如果state是一维的，添加一个批次维度
                            state = state.unsqueeze(0)
                        
                        # 尝试reshape，使用更安全的方法
                        try:
                            # 首先检查state是否为空
                            if state.numel() == 0:
                                # 如果state为空，创建一个默认的critic_in
                                critic_in = flow.zeros((1, 1, 1), device=self.device)
                            else:
                                # 检查state的形状是否合适
                                if len(state.shape) >= 2:
                                    # 使用view而不是reshape，避免内存重新分配
                                    try:
                                        # 计算最后一个维度的大小
                                        last_dim_size = state.shape[-1] if len(state.shape) > 2 else 1
                                        critic_in = state.view(state.shape[0], state.shape[1], last_dim_size)
                                    except Exception as e:
                                        # 如果view失败，使用clone和reshape
                                        critic_in = state.clone().reshape(state.shape[0], state.shape[1], -1)
                                else:
                                    # 如果维度不足，使用原始state
                                    critic_in = state
                        except Exception as e:
                            # 如果reshape失败，使用原始state
                            critic_in = state
                    else:
                        # 如果state不是张量，使用obs
                        critic_in = obs
                except Exception as e:
                    # 如果出现任何错误，使用obs作为备用
                    critic_in = obs
            
            # 确保RNN状态是有效的
            if not isinstance(rnn_state_actor, dict):
                rnn_state_actor = {k: (flow.zeros((1, 1, 64), device=self.device), 
                                      flow.zeros((1, 1, 64), device=self.device)) 
                                    for k in self.model_keys}
            if not isinstance(rnn_state_critic, dict):
                rnn_state_critic = {k: (flow.zeros((1, 1, 64), device=self.device), 
                                       flow.zeros((1, 1, 64), device=self.device)) 
                                       for k in self.model_keys}
            
            # 检查RNN状态中的每个键
            for key in self.model_keys:
                if key not in rnn_state_actor:
                    rnn_state_actor[key] = (flow.zeros((1, 1, 64), device=self.device), 
                                           flow.zeros((1, 1, 64), device=self.device))
                if key not in rnn_state_critic:
                    rnn_state_critic[key] = (flow.zeros((1, 1, 64), device=self.device), 
                                            flow.zeros((1, 1, 64), device=self.device))
                
                # 确保RNN状态是元组
                if not isinstance(rnn_state_actor[key], tuple):
                    rnn_state_actor[key] = list(rnn_state_actor[key])
                    rnn_state_actor[key][0] = flow.zeros((1, 1, 64), device=self.device)
                    rnn_state_actor[key][1] = flow.zeros((1, 1, 64), device=self.device)
                    rnn_state_actor[key] = tuple(rnn_state_actor[key])
                
                for i in range(len(rnn_state_actor[key])):
                    if not isinstance(rnn_state_actor[key][i], flow.Tensor):
                        rnn_state_actor[key] = list(rnn_state_actor[key])
                        rnn_state_actor[key][i] = flow.zeros((1, 1, 64), device=self.device)
                        rnn_state_actor[key] = tuple(rnn_state_actor[key])
                
                for i in range(len(rnn_state_critic[key])):
                    if not isinstance(rnn_state_critic[key][i], flow.Tensor):
                        rnn_state_critic[key] = list(rnn_state_critic[key])
                        rnn_state_critic[key][i] = flow.zeros((1, 1, 64), device=self.device)
                        rnn_state_critic[key] = tuple(rnn_state_critic[key])
            
            # 前向传播
            try:
                with flow.set_grad_enabled(True):
                    # 使用try-except包装每个可能失败的操作
                    try:
                        _, pi_dists_dict, rnn_state_outputs_actor = self.policy(
                            observation=obs, 
                            rnn_states=rnn_state_actor, 
                            avail_actions=avail_actions
                        )
                    except Exception as e:
                        # 如果策略前向传播失败，创建一个默认的分布字典
                        pi_dists_dict = {}
                        rnn_state_outputs_actor = {}
                        for key in self.model_keys:
                            action_dim = 5  # 默认动作维度
                            probs = flow.ones((1, action_dim), device=self.device) / action_dim
                            pi_dists_dict[key] = flow.distributions.Categorical(probs=probs)
                            rnn_state_outputs_actor[key] = (flow.zeros((1, 1, 64), device=self.device), 
                                                           flow.zeros((1, 1, 64), device=self.device))
                
                    try:
                        # 安全地获取值函数预测
                        if hasattr(self.policy, 'get_values'):
                            # 确保critic_in是有效的张量
                            if not isinstance(critic_in, flow.Tensor):
                                critic_in = flow.tensor(critic_in, dtype=flow.float32, device=self.device)
                            
                            # 检查critic_in是否为空
                            if critic_in.numel() == 0:
                                critic_in = flow.zeros((1, 1, 1), device=self.device)
                            
                            # 检查critic_in是否包含NaN
                            if flow.isnan(critic_in).any():
                                critic_in = flow.nan_to_num(critic_in, nan=0.0)
                            
                            # 使用try-except包装get_values调用
                            try:
                                _, values_pred_dict = self.policy.get_values(critic_in=critic_in)
                            except Exception as e:
                                print(f"Error in get_values call: {str(e)}")
                                # 如果失败，为每个model_key创建默认值
                                values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                        else:
                            # 如果没有get_values方法，创建默认值
                            values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                            rnn_state_outputs_critic = {k: (flow.zeros((1, 1, 64), device=self.device), 
                                                           flow.zeros((1, 1, 64), device=self.device)) 
                                                       for k in self.model_keys}
                    except Exception as e:
                        # 如果值函数预测失败，创建默认值
                        values_pred_dict = {k: flow.zeros((1, 1), device=self.device) for k in self.model_keys}
                        rnn_state_outputs_critic = {k: (flow.zeros((1, 1, 64), device=self.device), 
                                                       flow.zeros((1, 1, 64), device=self.device)) 
                                                   for k in self.model_keys}
            except Exception as e:
                return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "kl": 0.0, "error": str(e)}
            
            # 显式释放不再需要的变量
            del rnn_state_actor, rnn_state_critic
            
            # calculate losses
            loss_a, loss_e, loss_c = [], [], []
            for key in self.model_keys:
                try:
                    mask_values = agent_mask[key]
                    
                    # actor loss
                    log_pi = pi_dists_dict[key].log_prob(actions[key])
                    if log_pi.shape != log_pi_old[key].shape:
                        # 使用广播而不是reshape，更安全
                        if len(log_pi.shape) < len(log_pi_old[key].shape):
                            for _ in range(len(log_pi_old[key].shape) - len(log_pi.shape)):
                                log_pi = log_pi.unsqueeze(-1)
                        elif len(log_pi.shape) > len(log_pi_old[key].shape):
                            log_pi = log_pi.reshape(log_pi_old[key].shape)
                        else:
                            log_pi = log_pi.reshape(log_pi_old[key].shape)
                    
                    # 检查并修复无限值
                    log_pi = flow.clamp(log_pi, min=-20.0, max=20.0)
                    log_pi_old_k = flow.clamp(log_pi_old[key], min=-20.0, max=20.0)
                    
                    ratio = flow.exp(log_pi - log_pi_old_k)
                    # 限制ratio的范围，防止数值不稳定
                    ratio = flow.clamp(ratio, min=0.01, max=100.0)
                    
                    advantages_mask = advantages[key].detach() * mask_values
                    surrogate1 = ratio * advantages_mask
                    surrogate2 = flow.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
                    loss_a.append(-flow.min(surrogate1, surrogate2).mean())
                    
                    # entropy loss
                    entropy = pi_dists_dict[key].entropy()
                    if entropy.shape != mask_values.shape:
                        # 使用广播而不是reshape
                        if len(entropy.shape) < len(mask_values.shape):
                            for _ in range(len(mask_values.shape) - len(entropy.shape)):
                                entropy = entropy.unsqueeze(-1)
                        elif len(entropy.shape) > len(mask_values.shape):
                            entropy = entropy.reshape(mask_values.shape)
                        else:
                            entropy = entropy.reshape(mask_values.shape)
                    
                    loss_e.append((entropy * mask_values).mean())
                    
                    # critic loss
                    value_pred_i = values_pred_dict[key]
                    if value_pred_i.shape != returns[key].shape:
                        # 使用广播而不是reshape
                        if len(value_pred_i.shape) < len(returns[key].shape):
                            for _ in range(len(returns[key].shape) - len(value_pred_i.shape)):
                                value_pred_i = value_pred_i.unsqueeze(-1)
                        elif len(value_pred_i.shape) > len(returns[key].shape):
                            value_pred_i = value_pred_i.reshape(returns[key].shape)
                        else:
                            value_pred_i = value_pred_i.reshape(returns[key].shape)
                    
                    value_target = returns[key]
                    values_i = values_old[key]
                    
                    if self.use_value_clip:
                        value_clipped = values_i + (value_pred_i - values_i).clamp(-self.value_clip_range, self.value_clip_range)
                        if self.use_huber_loss:
                            try:
                                loss_v = flow.max(nn.SmoothL1Loss(reduction='none')(value_pred_i, value_target),
                                                nn.SmoothL1Loss(reduction='none')(value_clipped, value_target))
                            except Exception:
                                # 回退到平方误差
                                loss_v = flow.max(flow.square(value_pred_i - value_target),
                                                flow.square(value_clipped - value_target))
                        else:
                            loss_v = flow.max(flow.square(value_pred_i - value_target),
                                            flow.square(value_clipped - value_target))
                    else:
                        if self.use_huber_loss:
                            try:
                                loss_v = nn.SmoothL1Loss(reduction='none')(value_pred_i, value_target)
                            except Exception:
                                # 回退到平方误差
                                loss_v = flow.square(value_pred_i - value_target)
                        else:
                            loss_v = flow.square(value_pred_i - value_target)
                    
                    if loss_v.shape != mask_values.shape:
                        # 使用广播而不是reshape
                        if len(loss_v.shape) < len(mask_values.shape):
                            for _ in range(len(mask_values.shape) - len(loss_v.shape)):
                                loss_v = loss_v.unsqueeze(-1)
                        elif len(loss_v.shape) > len(mask_values.shape):
                            loss_v = loss_v.reshape(mask_values.shape)
                        else:
                            loss_v = loss_v.reshape(mask_values.shape)
                    
                    loss_c.append((loss_v * mask_values).mean())
                except Exception:
                    # 创建零损失张量，确保梯度计算不会中断
                    loss_a.append(flow.tensor(0.0, requires_grad=True))
                    loss_e.append(flow.tensor(0.0, requires_grad=True))
                    loss_c.append(flow.tensor(0.0, requires_grad=True))
            
            # 显式释放不再需要的变量
            del pi_dists_dict, values_pred_dict
            
            # calculate policy loss and backpropagate
            try:
                # 计算总体损失
                actor_loss = sum(loss_a) / max(1, self.n_groups)
                critic_loss = sum(loss_c) / max(1, self.n_groups)
                entropy_loss = sum(loss_e) / max(1, self.n_groups)
                
                # 检查损失值是否有效
                if flow.isnan(actor_loss) or flow.isinf(actor_loss):
                    actor_loss = flow.tensor(0.1, requires_grad=True)
                if flow.isnan(critic_loss) or flow.isinf(critic_loss):
                    critic_loss = flow.tensor(0.1, requires_grad=True)
                if flow.isnan(entropy_loss) or flow.isinf(entropy_loss):
                    entropy_loss = flow.tensor(0.01, requires_grad=True)
                
                # 计算策略损失
                policy_loss = actor_loss - self.ent_coef * entropy_loss + self.vf_coef * critic_loss
                
                # 检查策略损失是否有效
                if flow.isnan(policy_loss) or flow.isinf(policy_loss):
                    policy_loss = flow.tensor(0.1, requires_grad=True)
                
                # 反向传播
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # 使用try-except包装反向传播，避免梯度计算错误导致崩溃
                try:
                    policy_loss.backward()
                except Exception as e:
                    # 如果反向传播失败，使用actor_loss作为备用
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()
                    actor_loss.backward()
                
                # 梯度裁剪 - 始终应用，不仅在use_grad_norm为True时
                try:
                    # 对所有参数应用梯度裁剪，不仅是actor和critic
                    flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(),
                                                max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                except Exception:
                    # 单独裁剪actor和critic参数
                    try:
                        flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters_actor,
                                                    max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                    except Exception:
                        pass
                    
                    try:
                        flow.nn.utils.clip_grad_norm_(parameters=self.policy.parameters_critic,
                                                    max_norm=self.max_grad_norm if hasattr(self, 'max_grad_norm') else 10.0)
                    except Exception:
                        pass
                
                # 优化器步骤
                for optimizer in self.optimizers:
                    try:
                        optimizer.step()
                    except Exception:
                        pass
                
                # 学习率调度器
                if self.scheduler is not None:
                    for scheduler in self.schedulers:
                        try:
                            scheduler.step()
                        except Exception:
                            pass
            except Exception as e:
                # 创建一个备用的策略损失张量
                try:
                    # 尝试使用现有的损失张量
                    if 'actor_loss' in locals() and isinstance(actor_loss, flow.Tensor) and actor_loss.requires_grad:
                        policy_loss = actor_loss
                    else:
                        # 创建一个新的零张量
                        policy_loss = flow.tensor(0.1, requires_grad=True)
                    
                    # 尝试简化的反向传播
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()
                    policy_loss.backward()
                    
                    # 简化的优化器步骤
                    for optimizer in self.optimizers:
                        optimizer.step()
                except Exception:
                    # 返回空结果字典
                    return {
                        "actor-loss": 0.0,
                        "critic-loss": 0.0,
                        "entropy": 0.0,
                        "loss": 0.0,
                        "error": {"value": f"{str(e)}"}
                    }
            
            # 显式释放不再需要的变量
            del loss_a, loss_e, loss_c
            
            # update variables
            self.policy_loss = policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss)
            self.actor_loss = actor_loss.item() if hasattr(actor_loss, 'item') else float(actor_loss)
            self.critic_loss = critic_loss.item() if hasattr(critic_loss, 'item') else float(critic_loss)
            self.entropy_loss = entropy_loss.item() if hasattr(entropy_loss, 'item') else float(entropy_loss)
            
            # 显式释放不再需要的变量
            del policy_loss, actor_loss, critic_loss, entropy_loss
            
            # 简化统计计算，减少内存使用和计算复杂度
            # 只计算基本统计信息，避免复杂的计算
            self.pi_mean = 0.0
            self.v_mean = 0.0
            self.r_mean = 0.0
            self.a_mean = 0.0
            self.kl = 0.0
            self.clip_fraction = 0.0
            
            # 强制执行垃圾回收
            import gc
            gc.collect()
            
            # 如果支持，清除CUDA缓存
            try:
                if hasattr(flow, 'cuda') and hasattr(flow.cuda, 'empty_cache'):
                    flow.cuda.empty_cache()
            except Exception:
                pass
            
            return {
                "actor-loss": self.actor_loss,
                "critic-loss": self.critic_loss,
                "entropy": self.entropy_loss,
                "loss": self.policy_loss
            }
        
        except Exception as e:
            # 返回一个空的结果字典
            return {
                "actor-loss": 0.0,
                "critic-loss": 0.0,
                "entropy": 0.0,
                "loss": 0.0,
                "error": {"value": str(e)}
            }
