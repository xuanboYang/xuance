"""
Soft Actor-Critic with continuous action spaces (SAC)
Paper link: http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
import numpy as np
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup

class SAC_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(SAC_Learner, self).__init__(config, policy)
        # self.optimizer = {
        #     'actor': torch.optim.Adam(self.policy.actor_parameters, self.config.learning_rate_actor),
        #     'critic': torch.optim.Adam(self.policy.critic_parameters, self.config.learning_rate_critic)}
        # self.scheduler = {
        #     'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
        #                                                start_factor=1.0,
        #                                                end_factor=self.end_factor_lr_decay,
        #                                                total_iters=self.config.running_steps),
        #     'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
        #                                                 start_factor=1.0,
        #                                                 end_factor=self.end_factor_lr_decay,
        #                                                 total_iters=self.config.running_steps)}
        # 定义优化器
        self.optimizer = {
            'actor': Adam(
                learning_rate=self.config.learning_rate_actor,
                parameters=self.policy.actor_parameters
            ),
            'critic': Adam(
                learning_rate=self.config.learning_rate_critic,
                parameters=self.policy.critic_parameters
            )
        }

        # 定义学习率调度器
        self.scheduler = {
            'actor': LinearWarmup(
                learning_rate=self.config.learning_rate_actor,
                warmup_steps=self.config.running_steps,
                start_lr=self.config.learning_rate_actor * 1.0,
                end_lr=self.config.learning_rate_actor * self.end_factor_lr_decay,
                verbose=False
            ),
            'critic': LinearWarmup(
                learning_rate=self.config.learning_rate_critic,
                warmup_steps=self.config.running_steps,
                start_lr=self.config.learning_rate_critic * 1.0,
                end_lr=self.config.learning_rate_critic * self.end_factor_lr_decay,
                verbose=False
            )
        }

        self.mse_loss = nn.MSELoss()
        self.tau = config.tau
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.use_automatic_entropy_tuning = config.use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(policy.action_space.shape).item()
            # self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
            # 定义 log_alpha 参数
            self.log_alpha = paddle.create_parameter(
                shape=[1],
                default_initializer=nn.initializer.Constant(value=0.0),
                dtype='float32'
            )
            self.log_alpha.stop_gradient = False  # 确保参数可训练

            self.alpha = self.log_alpha.exp()
            # self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate_actor)
            # 定义优化器
            self.alpha_optimizer = paddle.optimizer.Adam(
                learning_rate=self.config.learning_rate_actor,
                parameters=[self.log_alpha]
            )

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0):
        if isinstance(parameters, paddle.Tensor):
            parameters = [parameters]

        norm_type = float(norm_type)
        total_norm = 0.0

        for p in parameters:
            if p.grad is not None:
                param_norm = paddle.norm(p.grad, norm_type).item()
                total_norm += param_norm ** norm_type

        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.scale_(clip_coef)

        return total_norm

    def update(self, **samples):
        info = {}
        self.iterations += 1
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        act_batch = paddle.to_tensor(samples['actions']).to(self.device)
        next_batch = paddle.to_tensor(samples['obs_next']).to(self.device)
        rew_batch = paddle.to_tensor(samples['rewards']).to(self.device)
        ter_batch = paddle.to_tensor(samples['terminals'], dtype=paddle.float32).to(self.device)

        # actor update
        log_pi, policy_q_1, policy_q_2 = self.policy.Qpolicy(obs_batch)
        policy_q = paddle.min(policy_q_1, policy_q_2).reshape([-1])
        p_loss = (self.alpha * log_pi.reshape([-1]) - policy_q).mean()
        self.optimizer['actor'].clear_grad()
        p_loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
            self.clip_grad_norm_(self.policy.actor_parameters(), self.grad_clip_norm)

        self.optimizer['actor'].step()

        # critic update
        action_q_1, action_q_2 = self.policy.Qaction(obs_batch, act_batch)
        log_pi_next, target_q = self.policy.Qtarget(next_batch)
        target_value = target_q - self.alpha * log_pi_next.reshape([-1])
        backup = rew_batch + (1 - ter_batch) * self.gamma * target_value
        q_loss = self.mse_loss(action_q_1, backup.detach()) + self.mse_loss(action_q_2, backup.detach())
        self.optimizer['critic'].clear_grad()
        q_loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
            self.clip_grad_norm_(self.policy.critic_parameters(), self.grad_clip_norm)

        self.optimizer['critic'].step()

        # automatic entropy tuning
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.clear_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = paddle.zeros([])

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
