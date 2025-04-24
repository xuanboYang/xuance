"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: oneflow
"""
import oneflow as flow
from oneflow import nn
from oneflow.nn.functional import one_hot
from xuance.common import List
from argparse import Namespace
from xuance.oneflow.learners.multi_agent_rl.iac_learner import IAC_Learner
from typing import Optional
import numpy as np
from operator import itemgetter
from oneflow import Tensor


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.mse_loss = nn.MSELoss()

    def build_optimizer(self):
        self.optimizer = {
            'actor': flow.optim.Adam(self.policy.parameters_actor, self.config.learning_rate_actor, eps=1e-5),
            'critic': flow.optim.Adam(self.policy.parameters_critic, self.config.learning_rate_critic, eps=1e-5)
        }
        self.scheduler = {
            'actor': flow.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
            'critic': flow.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)
        }

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        returns = sample_Tensor['returns']
        IDs = sample_Tensor['agent_ids']

        # 确保所有输入在同一设备上
        device = self.device
        for key in obs:
            if obs[key].device != device:
                obs[key] = obs[key].to(device)
        
        if state.device != device:
            state = state.to(device)
        
        for key in actions:
            if actions[key].device != device:
                actions[key] = actions[key].to(device)
        
        for key in agent_mask:
            if agent_mask[key].device != device:
                agent_mask[key] = agent_mask[key].to(device)
        
        if avail_actions is not None:
            for key in avail_actions:
                if avail_actions[key].device != device:
                    avail_actions[key] = avail_actions[key].to(device)
        
        for key in returns:
            if returns[key].device != device:
                returns[key] = returns[key].to(device)
        
        if IDs.device != device:
            IDs = IDs.to(device)

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=epsilon)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: one_hot(actions[key].long(), self.n_actions[key])}
        else:
            IDs = flow.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).reshape(bs, -1).to(device)
            actions_onehot = {k: one_hot(actions[k].long(), self.n_actions[k]) for k in self.agent_keys}

        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {k: values_pred.reshape(bs, -1) for k in self.model_keys}
        else:
            values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]

            pi_probs = pi_dist_dict[key].probs
            if self.use_actions_mask:
                pi_probs[avail_actions[key] == 0] = 0
            baseline = (pi_probs * values_pred_dict[key]).sum(-1).reshape(bs)
            pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs)
            log_pi_taken = flow.log(pi_taken).reshape(bs)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].zero_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            grad_norm = flow.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = flow.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        }

        return info

    def update_rnn(self, sample, epsilon=0.0):
        self.iterations += 1
        info = {}

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        returns = sample_Tensor['returns']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        # 确保所有输入在同一设备上
        device = self.device
        for key in obs:
            if obs[key].device != device:
                obs[key] = obs[key].to(device)
        
        if state.device != device:
            state = state.to(device)
        
        for key in actions:
            if actions[key].device != device:
                actions[key] = actions[key].to(device)
        
        for key in agent_mask:
            if agent_mask[key].device != device:
                agent_mask[key] = agent_mask[key].to(device)
        
        if avail_actions is not None:
            for key in avail_actions:
                if avail_actions[key].device != device:
                    avail_actions[key] = avail_actions[key].to(device)
        
        for key in returns:
            if returns[key].device != device:
                returns[key] = returns[key].to(device)
        
        if filled.device != device:
            filled = filled.to(device)
        
        if IDs.device != device:
            IDs = IDs.to(device)

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape(bs_rnn, seq_len)
        else:
            IDs = flow.eye(self.n_agents).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(device)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        
        # 确保RNN隐藏状态在正确的设备上
        for key in self.model_keys:
            rnn_hidden_actor[key] = (rnn_hidden_actor[key][0].to(device), rnn_hidden_actor[key][1].to(device))
            rnn_hidden_critic[key] = (rnn_hidden_critic[key][0].to(device), rnn_hidden_critic[key][1].to(device))

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                      rnn_hidden=rnn_hidden_actor, epsilon=epsilon)
        actions_onehot = {k: one_hot(actions[k].long(), self.n_actions[k]) for k in self.model_keys}
        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, rnn_hidden=rnn_hidden_critic, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {self.model_keys[0]: values_pred.transpose(1, 2).reshape(bs_rnn, seq_len, -1)}
        else:
            values_pred_dict = {k: values_pred[:, :, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled

            pi_probs = pi_dist_dict[key].probs
            if self.use_actions_mask:
                pi_probs[avail_actions[key] == 0] = 0
            baseline = (pi_probs * values_pred_dict[key]).sum(-1).reshape(bs_rnn, seq_len)
            pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).long())
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).long()).reshape(bs_rnn, seq_len)
            log_pi_taken = flow.log(pi_taken).reshape(bs_rnn, seq_len)
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].zero_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            grad_norm = flow.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].zero_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            grad_norm = flow.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            info["gradient_norm_actor"] = grad_norm.item()
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        }

        return info

    def build_training_data(self, sample: Optional[dict],
                            use_parameter_sharing: Optional[bool] = False,
                            use_actions_mask: Optional[bool] = False,
                            use_global_state: Optional[bool] = False):
        """
        Build training data for COMA learner, different from IAC_learner.

        Parameters:
            sample (dict): The experience sample to build training data.
            use_parameter_sharing (bool): Whether to use parameter sharing for agents.
            use_actions_mask (bool): Whether to use a mask for available actions.
            use_global_state (bool): Whether to use the global state as a basis.

        Returns:
            training_data: The dictionary of training data.
        """
        batch_size = sample['batch_size']
        seq_length = sample['sequence_length'] if self.use_rnn else 1
        state, avail_actions, filled, IDs = None, None, None, None
        if use_parameter_sharing:
            k = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['obs']), axis=1)).to(self.device)
            actions_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['actions']), axis=1)).to(self.device)
            values_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['values']), axis=1)).to(self.device)
            returns_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['returns']), axis=1)).to(self.device)
            
            # 检查 'log_pi_old' 键是否存在
            if 'log_pi_old' in sample:
                log_pi_old_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['log_pi_old']), 1)).to(self.device)
            else:
                # 如果不存在，则使用零值替代
                log_pi_old_tensor = flow.zeros_like(values_tensor)
                
            ter_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['terminals']), 1)).float().to(self.device)
            msk_tensor = Tensor(np.stack(itemgetter(*self.agent_keys)(sample['agent_mask']), 1)).float().to(self.device)
            if self.use_rnn:
                obs = {k: obs_tensor.reshape(bs, seq_length, -1)}
                if len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, seq_length)}
                elif len(actions_tensor.shape) == 4:
                    actions = {k: actions_tensor.reshape(bs, seq_length, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape(bs, seq_length)}
                returns = {k: returns_tensor.reshape(bs, seq_length)}
                log_pi_old = {k: log_pi_old_tensor.reshape(bs, seq_length)}
                terminals = {k: ter_tensor.reshape(bs, seq_length)}
                agent_mask = {k: msk_tensor.reshape(bs, seq_length)}
                IDs = flow.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(
                    batch_size, -1, seq_length, -1).reshape(bs, seq_length, self.n_agents).to(self.device)
            else:
                obs = {k: obs_tensor.reshape(bs, -1)}
                if len(actions_tensor.shape) == 2:
                    actions = {k: actions_tensor.reshape(bs)}
                elif len(actions_tensor.shape) == 3:
                    actions = {k: actions_tensor.reshape(bs, -1)}
                else:
                    raise AttributeError("Wrong actions shape.")
                values = {k: values_tensor.reshape(bs)}
                returns = {k: returns_tensor.reshape(bs)}
                log_pi_old = {k: log_pi_old_tensor.reshape(bs)}
                terminals = {k: ter_tensor.reshape(bs)}
                agent_mask = {k: msk_tensor.reshape(bs)}
                IDs = flow.eye(self.n_agents).unsqueeze(0).expand(
                    batch_size, -1, -1).reshape(bs, self.n_agents).to(self.device)

            if use_actions_mask:
                avail_a = np.stack(itemgetter(*self.agent_keys)(sample['avail_actions']), axis=1)
                if self.use_rnn:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, seq_length, -1])).float().to(self.device)}
                else:
                    avail_actions = {k: Tensor(avail_a.reshape([bs, -1])).float().to(self.device)}

        else:
            # parse the agent experiences and convert to tensors
            obs = {}
            actions = {}
            values = {}
            returns = {}
            log_pi_old = {}
            terminals = {}
            agent_mask = {}
            # iterate each agent
            for agent_id in self.model_keys:
                obs[agent_id] = Tensor(sample['obs'][agent_id]).to(self.device)
                actions[agent_id] = Tensor(sample['actions'][agent_id]).to(self.device)
                values[agent_id] = Tensor(sample['values'][agent_id]).to(self.device)
                returns[agent_id] = Tensor(sample['returns'][agent_id]).to(self.device)
                
                # 检查 'log_pi_old' 键是否存在
                if 'log_pi_old' in sample:
                    log_pi_old[agent_id] = Tensor(sample['log_pi_old'][agent_id]).to(self.device)
                else:
                    # 如果不存在，则使用零值替代
                    log_pi_old[agent_id] = flow.zeros_like(values[agent_id])
                
                terminals[agent_id] = Tensor(sample['terminals'][agent_id]).float().to(self.device)
                agent_mask[agent_id] = Tensor(sample['agent_mask'][agent_id]).float().to(self.device)
            
            if use_actions_mask:
                avail_actions = {k: Tensor(sample['avail_actions'][k]).float().to(self.device) for k in self.agent_keys}
            
            if self.use_rnn:
                filled = Tensor(sample['filled']).bool().to(self.device)
                rnn_hidden = sample['rnn_hidden']

        if use_global_state:
            state = Tensor(sample['state']).to(self.device)

        if self.use_rnn:
            filled = Tensor(sample['filled']).float().to(self.device)

        sample_Tensor = {
            'batch_size': batch_size,
            'state': state,
            'obs': obs,
            'actions': actions,
            'values': values,
            'returns': returns,
            'log_pi_old': log_pi_old,
            'terminals': terminals,
            'agent_mask': agent_mask,
            'avail_actions': avail_actions,
            'agent_ids': IDs,
            'filled': filled,
            'seq_length': seq_length,
        }
        return sample_Tensor
