"""
Deep Deterministic Policy Gradient (DDPG)
Paper link: https://arxiv.org/pdf/1509.02971.pdf
Implementation: oneflow
"""
import oneflow as flow
from oneflow import nn
from xuance.oneflow.learners.learner import Learner
from argparse import Namespace


class DDPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(DDPG_Learner, self).__init__(config, policy)
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
        self.tau = config.tau
        self.gamma = config.gamma
        self.mse_loss = nn.MSELoss()

    def update(self, **samples):
        self.iterations += 1
        obs_batch = flow.as_tensor(samples['obs'], device=self.device)
        act_batch = flow.as_tensor(samples['actions'], device=self.device)
        next_batch = flow.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = flow.as_tensor(samples['rewards'], device=self.device)
        ter_batch = flow.as_tensor(samples['terminals'], dtype=flow.float, device=self.device)

        # critic update
        action_q = self.policy.Qaction(obs_batch, act_batch).reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + (1 - ter_batch) * self.gamma * next_q
        q_loss = self.mse_loss(action_q, target_q.detach())
        self.optimizer['critic'].zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            flow.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
        self.optimizer['critic'].step()

        # actor update
        policy_q = self.policy.Qpolicy(obs_batch)
        p_loss = -policy_q.mean()
        self.optimizer['actor'].zero_grad()
        p_loss.backward()
        if self.use_grad_clip:
            flow.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
        self.optimizer['actor'].step()

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
                f"Qvalue/rank_{self.rank}": action_q.mean().item(),
                f"actor_lr/rank_{self.rank}": actor_lr,
                f"critic_lr/rank_{self.rank}": critic_lr
            }
        else:
            info = {
                "Qloss": q_loss.item(),
                "Ploss": p_loss.item(),
                "Qvalue": action_q.mean().item(),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr
            }

        return info
