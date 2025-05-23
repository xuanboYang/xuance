"""
DQN with Dueling network (Dueling DQN)
Paper link: http://proceedings.mlr.press/v48/wangf16.pdf
Implementation: oneflow
"""
import oneflow as flow
from oneflow import nn
from xuance.oneflow.learners.learner import Learner
from argparse import Namespace


class DuelDQN_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(DuelDQN_Learner, self).__init__(config, policy)
        self.optimizer = flow.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = flow.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim

    def update(self, **samples):
        self.iterations += 1
        obs_batch = flow.as_tensor(samples['obs'], device=self.device)
        act_batch = flow.as_tensor(samples['actions'], device=self.device)
        next_batch = flow.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = flow.as_tensor(samples['rewards'], device=self.device)
        ter_batch = flow.as_tensor(samples['terminals'], dtype=flow.float, device=self.device)

        _, _, evalQ = self.policy(obs_batch)
        _, _, targetQ = self.policy.target(next_batch)
        targetQ = targetQ.max(dim=-1).values
        targetQ = rew_batch + self.gamma * (1 - ter_batch) * targetQ
        predictQ = (evalQ * self.one_hot(act_batch.long(), evalQ.shape[1])).sum(dim=-1)

        loss = self.mse_loss(predictQ, targetQ)
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            flow.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
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

        return info
