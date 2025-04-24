"""
Policy Gradient (PG)
Paper link: https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import Learner
from argparse import Namespace
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup

class PG_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer):
        super(PG_Learner, self).__init__(config, policy)
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

        # 定义线性学习率调度器
        self.scheduler = LinearWarmup(
            learning_rate=self.config.learning_rate,  # 初始学习率
            warmup_steps=self.config.running_steps,  # 总步数（等价于 total_iters）
            start_lr=self.config.learning_rate,  # 起始学习率（等价于 start_factor * base_lr）
            end_lr=self.config.learning_rate * self.end_factor_lr_decay,  # 最终学习率（等价于 end_factor * base_lr）
            verbose=False  # 是否打印学习率信息
        )

        self.ent_coef = config.ent_coef

    def update(self, **samples):
        self.iterations += 1
        obs_batch = paddle.to_tensor(samples['obs']).to(self.device)
        act_batch = paddle.to_tensor(samples['actions']).to(self.device)
        ret_batch = paddle.to_tensor(samples['returns']).to(self.device)

        _, a_dist, _ = self.policy(obs_batch)
        log_prob = a_dist.log_prob(act_batch)

        a_loss = -(ret_batch * log_prob).mean()
        e_loss = a_dist.entropy().mean()

        loss = a_loss - self.ent_coef * e_loss
        # self.optimizer.zero_grad()
        self.optimizer.clear_grad()

        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
            def clip_grad_norm_(parameters, max_norm):
                """
                手动实现梯度裁剪（等价于 PyTorch 的 clip_grad_norm_）
                """
                if isinstance(parameters, paddle.Tensor):
                    parameters = [parameters]

                total_norm = 0.0
                for p in parameters:
                    if p.grad is not None:
                        param_norm = paddle.norm(p.grad).item()  # 计算当前参数的梯度范数
                        total_norm += param_norm ** 2
                total_norm = total_norm ** 0.5  # 计算总梯度范数

                clip_coef = max_norm / (total_norm + 1e-6)  # 避免除以零
                if clip_coef < 1:
                    for p in parameters:
                        if p.grad is not None:
                            p.grad.scale_(clip_coef)  # 按比例缩放梯度

                return total_norm

            # 调用
            clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info = {
                f"actor-loss/rank_{self.rank}": a_loss.item(),
                f"entropy/rank_{self.rank}": e_loss.item(),
                f"learning_rate/rank_{self.rank}": lr
            }
        else:
            info = {
                "actor-loss": a_loss.item(),
                "entropy": e_loss.item(),
                "learning_rate": lr
            }

        return info
