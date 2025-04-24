"""
MFQ: Mean Field Q-Learning
Paper link:
http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import LearnerMAS
from xuance.common import Optional, Union
from argparse import Namespace


class MFQ_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Layer,
                 optimizer: paddle.optimizer.Optimizer,
                 scheduler: Optional[paddle.optimizer.lr.LRScheduler] = None,
                 device: Optional[Union[int, str]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.temperature = config.temperature
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        self.softmax = paddle.nn.Softmax(axis=-1)
        super(MFQ_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def get_boltzmann_policy(self, q):
        return self.softmax(q / self.temperature)

    def update(self, sample):
        self.iterations += 1
        obs = paddle.Tensor(sample['obs']).to(self.device)
        actions = paddle.Tensor(sample['actions']).to(self.device)
        obs_next = paddle.Tensor(sample['obs_next']).to(self.device)
        act_mean = paddle.Tensor(sample['act_mean']).to(self.device)
        act_mean_next = paddle.Tensor(sample['act_mean_next']).to(self.device)
        rewards = paddle.Tensor(sample['rewards']).to(self.device)
        terminals = paddle.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = paddle.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = paddle.eye(self.n_agents).unsqueeze(0).expand((self.args.batch_size, -1, -1)).to(self.device)

        act_mean = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])
        act_mean_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])
        _, _, q_eval = self.policy(obs, act_mean, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
        q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
        shape = q_next.shape
        pi = self.get_boltzmann_policy(q_next)
        v_mf = paddle.bmm(q_next.reshape(-1, 1, shape[-1]), pi.unsqueeze(-1).reshape(-1, shape[-1], 1))
        v_mf = v_mf.reshape(*(list(shape[0:-1]) + [1]))
        q_target = rewards + (1 - terminals) * self.args.gamma * v_mf

        # calculate the loss function
        td_error = (q_eval_a - q_target.detach()) * agent_mask
        loss = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_eval_a.mean().item()
        }

        return info
