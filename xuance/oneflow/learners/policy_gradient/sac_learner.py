"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: oneflow
"""
import oneflow as flow
from oneflow import nn
import numpy as np
from xuance.oneflow.learners.learner import Learner
from argparse import Namespace


class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(SAC_Learner, self).__init__(config, policy)
        self.optimizer = {
            'actor': flow.optim.Adam(self.policy.actor_parameters, self.config.learning_rate_actor),
            'critic': flow.optim.Adam(self.policy.critic_parameters, self.config.learning_rate_critic)}
        self.scheduler = {
            'actor': flow.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
                                                       start_factor=1.0,
                                                       end_factor=self.end_factor_lr_decay,
                                                       total_iters=self.config.running_steps),
            'critic': flow.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
                                                        start_factor=1.0,
                                                        end_factor=self.end_factor_lr_decay,
                                                        total_iters=self.config.running_steps)}
        self.mse_loss = nn.MSELoss()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(policy.action_space.shape).item()
            self.log_alpha = nn.Parameter(flow.zeros(1, requires_grad=True, device=self.device))
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = flow.optim.Adam([self.log_alpha], lr=config.learning_rate_actor)

    def update(self, **samples):
        info = {}
        self.iterations += 1
        obs_batch = flow.as_tensor(samples['obs'], device=self.device)
        act_batch = flow.as_tensor(samples['actions'], device=self.device)
        next_batch = flow.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = flow.as_tensor(samples['rewards'], device=self.device)
        ter_batch = flow.as_tensor(samples['terminals'], dtype=flow.float, device=self.device)

        # actor update
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
        policy_q = flow.min(policy_q_1, policy_q_2).reshape([-1])
        p_loss = (self.alpha * log_pi.reshape([-1]) - policy_q).mean()
        self.optimizer['actor'].zero_grad()
        p_loss.backward()
        if self.use_grad_clip:
            flow.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
        self.optimizer['actor'].step()

        # critic update
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_value = target_q - self.alpha * log_pi_next.reshape([-1])
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        q_loss = self.mse_loss(action_q_1, backup.detach()) + self.mse_loss(action_q_2, backup.detach())
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            flow.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = flow.zeros([])

        if self.scheduler is not None:
            self.scheduler['actor'].step()
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info = {
                f"Qloss/rank_{self.rank}": q_loss.item(),
                f"Ploss/rank_{self.rank}": p_loss.item(),
                f"Qvalue/rank_{self.rank}": policy_q.mean().item(),
                f"actor_lr/rank_{self.rank}": actor_lr,
                f"critic_lr/rank_{self.rank}": critic_lr,
            }
        else:
            info = {
                "Qloss": q_loss.item(),
                "Ploss": p_loss.item(),
                "Qvalue": policy_q.mean().item(),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
            }
        if self.use_automatic_entropy_tuning:
            if self.distributed_training:
                info.update({f"alpha_loss/rank_{self.rank}": alpha_loss.item(),
                             f"alpha/rank_{self.rank}": self.alpha.item()})
            else:
                info.update({"alpha_loss": alpha_loss.item(),
                             "alpha": self.alpha.item()})

        return info
