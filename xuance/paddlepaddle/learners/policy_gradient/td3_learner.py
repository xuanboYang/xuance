"""
Twin Delayed Deep Deterministic Policy Gradient (TD3)
Paper link: http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup

class TD3_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(TD3_Learner, self).__init__(config, policy)
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

        self.tau = config.tau
        self.gamma = config.gamma
        self.actor_update_delay = config.actor_update_delay
        self.mse_loss = nn.MSELoss()

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
        self.iterations += 1
        info = {}
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        act_batch = paddle.to_tensor(samples['actions']).to(self.device)
        next_batch = paddle.to_tensor(samples['obs_next']).to(self.device)
        rew_batch = paddle.to_tensor(samples['rewards']).to(self.device)
        ter_batch = paddle.to_tensor(samples['terminals'], dtype=paddle.float32).to(self.device)

        # critic update
        action_q_A, action_q_B = self.policy.Qaction(obs_batch, act_batch)
        action_q_A = action_q_A.reshape([-1])
        action_q_B = action_q_B.reshape([-1])
        next_q = self.policy.Qtarget(next_batch).reshape([-1])
        target_q = rew_batch + self.gamma * (1 - ter_batch) * next_q
        q_loss = self.mse_loss(action_q_A, target_q.detach()) + self.mse_loss(action_q_B, target_q.detach())
        self.optimizer['critic'].clear_grad()
        q_loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters, self.grad_clip_norm)
            self.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer['critic'].step()
        if self.scheduler is not None:
            self.scheduler['critic'].step()

        # actor update
        if self.iterations % self.actor_update_delay == 0:
            policy_q = self.policy.Qpolicy(obs_batch)
            p_loss = -policy_q.mean()
            self.optimizer['actor'].clear_grad()
            p_loss.backward()
            if self.use_grad_clip:
                # torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters, self.grad_clip_norm)
                self.clip_grad_norm_(self.policy.actor_parameters(), self.grad_clip_norm)
            self.optimizer['actor'].step()
            if self.scheduler is not None:
                self.scheduler['actor'].step()
            self.policy.soft_update(self.tau)
            info.update({"Ploss": p_loss.item()})

        actor_lr = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        critic_lr = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info.update({
                f"Qloss/rank_{self.rank}": q_loss.item(),
                f"QvalueA/rank_{self.rank}": action_q_A.mean().item(),
                f"QvalueB/rank_{self.rank}": action_q_B.mean().item(),
                f"actor_lr/rank_{self.rank}": actor_lr,
                f"critic_lr/rank_{self.rank}": critic_lr
            })
        else:
            info.update({
                "Qloss": q_loss.item(),
                "QvalueA": action_q_A.mean().item(),
                "QvalueB": action_q_B.mean().item(),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr
            })

        return info
