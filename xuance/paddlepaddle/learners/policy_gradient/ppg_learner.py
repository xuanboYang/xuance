"""
Phasic Policy Gradient (PPG)
Paper link: http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from xuance.paddlepaddle.utils.operations import merge_distributions
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup

class PPG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(PPG_Learner, self).__init__(config, policy)
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
        self.ent_coef = config.ent_coef
        self.clip_range = config.clip_range
        self.kl_beta = config.kl_beta
        self.policy_iterations = 0
        self.value_iterations = 0

    def update_policy(self, **samples):
        self.policy_iterations += 1
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        act_batch = paddle.to_tensor(samples['actions']).to(self.device)
        adv_batch = paddle.to_tensor(samples['advantages']).to(self.device)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])
        old_logp_batch = old_dist.log_prob(act_batch).detach()

        outputs, a_dist, _, _ = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)
        # ppo-clip core implementations 
        ratio = (log_prob - old_logp_batch).exp().float()
        surrogate1 = ratio.clamp(1.0 - self.clip_range, 1.0 + self.clip_range) * adv_batch
        surrogate2 = adv_batch * ratio
        a_loss = -paddle.minimum(surrogate1, surrogate2).mean()
        e_loss = a_dist.entropy().mean()
        loss = a_loss - self.ent_coef * e_loss
        # self.optimizer.zero_grad()
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            # 手动实现梯度裁剪函数
            def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
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

            clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        cr = ((ratio < 1 - self.clip_range).sum() + (ratio > 1 + self.clip_range).sum()) / ratio.shape[0]

        if self.distributed_training:
            info = {
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                f"clip_ratio/rank_{self.rank}": cr,
            }
        else:
            info = {
                "actor-loss": a_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr,
                "clip_ratio": cr,
            }

        return info

    def update_critic(self, **samples):
        self.value_iterations += 1
        obs_batch = paddle.to_tensor(samples['obs']).to(device=self.device)
        ret_batch = paddle.to_tensor(samples['returns']).to(device=self.device)

        _, _, v_pred, _ = self.policy(obs_batch)
        loss = self.mse_loss(v_pred, ret_batch)
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            # 手动实现梯度裁剪函数
            def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
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

            clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        if self.distributed_training:
            info = {f"critic-loss/rank_{self.rank}": loss.item()}
        else:
            info = {"critic-loss": loss.item()}
        return info

    def update_auxiliary(self, **samples):
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        ret_batch = paddle.to_tensor(samples['returns']).to(self.device)
        old_dist = merge_distributions(samples['aux_batch']['old_dist'])

        outputs, a_dist, v, aux_v = self.policy(obs_batch)
        aux_loss = self.mse_loss(v.detach(), aux_v)
        kl_loss = a_dist.kl_divergence(old_dist).mean()
        value_loss = self.mse_loss(v, ret_batch)
        loss = aux_loss + self.kl_beta * kl_loss + value_loss
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            # 手动实现梯度裁剪函数
            def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
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
            clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.distributed_training:
            info = {f"kl-loss/rank_{self.rank}": loss.item()}
        else:
            info = {"kl-loss": loss.item()}
        return info

    def update(self, *args):
        return
