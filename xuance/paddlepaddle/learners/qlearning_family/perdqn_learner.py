"""
DQN with Prioritized Experience Replay (PER-DQN)
Paper link: https://arxiv.org/pdf/1511.05952.pdf
Implementation: Pytorch
"""
import os
import paddle
import numpy as np
from paddle import nn
from xuance.common import Optional
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup

class PerDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(PerDQN_Learner, self).__init__(config, policy)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
        #                                                    start_factor=1.0,
        #                                                    end_factor=self.end_factor_lr_decay,
        #                                                    total_iters=self.config.running_steps)
        # 定义优化器
        self.optimizer = Adam(
            learning_rate=self.config.learning_rate,
            epsilon=1e-5,  # 对应 PyTorch 的 eps 参数
            parameters=self.policy.parameters()
        )

        # 定义学习率调度器
        self.scheduler = LinearWarmup(
            learning_rate=self.config.learning_rate,  # 初始学习率
            warmup_steps=self.config.running_steps,  # 总步数（对应 PyTorch 的 total_iters）
            start_lr=self.config.learning_rate * 1.0,  # 起始学习率（start_factor=1.0）
            end_lr=self.config.learning_rate * self.end_factor_lr_decay,  # 结束学习率（end_factor）
            verbose=False
        )

        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim

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
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        act_batch = paddle.to_tensor(samples['actions']).to(self.device)
        next_batch = paddle.to_tensor(samples['obs_next']).to(self.device)
        rew_batch = paddle.to_tensor(samples['rewards']).to(self.device)
        ter_batch = paddle.to_tensor(samples['terminals'], dtype=paddle.float16).to(self.device)

        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(axis=-1)
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * self.one_hot(act_batch.long(), evalQ.shape[1])).sum(axis=-1)

        td_error = targetQ - predictQ
        loss = self.mse_loss(predictQ, targetQ)
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info = {
                f"Qloss/rank_{self.rank}": loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"predictQ/rank_{self.rank}": predictQ.mean().item()
            }
        else:
            info = {
                "Qloss": loss.item(),
                "learning_rate": lr,
                "predictQ": predictQ.mean().item()
            }
        
        return np.abs(td_error.cpu().detach().numpy()), info
