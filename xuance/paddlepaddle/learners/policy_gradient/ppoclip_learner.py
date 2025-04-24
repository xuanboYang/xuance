"""
Proximal Policy Optimization with clip trick (PPO_CLIP)
Paper link: https://arxiv.org/pdf/1707.06347.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup


class PPOCLIP_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(PPOCLIP_Learner, self).__init__(config, policy)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
        #                                                    start_factor=1.0,
        #                                                    end_factor=self.end_factor_lr_decay,
        #                                                    total_iters=self.config.running_steps)
        # 定义 Adam 优化器
        self.optimizer = Adam(
            learning_rate=self.config.learning_rate,
            epsilon=1e-5,
            parameters=self.policy.parameters()
        )
        # 定义线性学习率调度器（LinearWarmup）
        self.scheduler = LinearWarmup(
            learning_rate=self.config.learning_rate,  # 初始学习率
            warmup_steps=self.config.running_steps,  # 总步数（等价于 total_iters）
            start_lr=self.config.learning_rate * 1.0,  # 起始学习率（等价于 start_factor * base_lr）
            end_lr=self.config.learning_rate * self.end_factor_lr_decay,  # 最终学习率（等价于 end_factor * base_lr）
            verbose=False  # 是否打印学习率信息
        )

        self.mse_loss = nn.MSELoss()
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range

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
        ret_batch = paddle.to_tensor(samples['returns']).to(self.device)
        adv_batch = paddle.to_tensor(samples['advantages']).to(self.device)
        old_logp_batch = paddle.to_tensor(samples['aux_batch']['old_logp']).to(self.device)

        outputs, a_dist, v_pred = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -paddle.minimum(surrogate1, surrogate2).mean()

        c_loss = self.mse_loss(v_pred, ret_batch.detach())

        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss + self.vf_coef * c_loss
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            self.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]

        if self.distributed_training:
            info = {
                f"actor_loss/rank_{self.rank}": a_loss.item(),
                f"critic_loss/rank_{self.rank}": c_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"predict_value/rank_{self.rank}": v_pred.mean().item(),
                f"clip_ratio/rank_{self.rank}": cr
            }
        else:
            info = {
                "actor_loss": a_loss.item(),
                "critic_loss": c_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "predict_value": v_pred.mean().item(),
                "clip_ratio": cr
            }

        return info
