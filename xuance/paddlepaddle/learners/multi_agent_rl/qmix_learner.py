import paddle
from paddle import nn
from xuance.paddlepaddle.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
from operator import itemgetter


class QMIX_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Layer):
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        super(QMIX_Learner, self).__init__(config, model_keys, agent_keys, policy)

        # 定义优化器和学习率调度器
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.policy.parameters_model,
            learning_rate=config.learning_rate,
            epsilon=1e-5
        )
        self.scheduler = paddle.optimizer.lr.LinearLR(
            learning_rate=config.learning_rate,
            start_factor=1.0,
            end_factor=self.end_factor_lr_decay,
            total_steps=self.config.running_steps,
            verbose=False
        )
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # 准备训练数据
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(axis=1).reshape([batch_size, 1])
            terminals_tot = paddle.all(terminals[key], axis=1, keepdim=False).astype('float32').reshape([batch_size, 1])
        else:
            bs = batch_size
            rewards_list = itemgetter(*self.agent_keys)(rewards)
            rewards_tot = paddle.stack(rewards_list, axis=1).mean(axis=-1, keepdim=True)
            terminals_list = itemgetter(*self.agent_keys)(terminals)
            terminals_tot = paddle.stack(terminals_list, axis=1).all(axis=1, keepdim=True).astype('float32')

        _, _, q_eval = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions)
        _, q_next = self.policy.Qtarget(observation=obs_next, agent_ids=IDs)

        q_eval_a, q_next_a = {}, {}
        for key in self.model_keys:
            # actions_index = actions[key].astype('int64').unsqueeze(-1)
            # q_eval_a[key] = paddle.gather_nd(q_eval[key], actions_index).reshape([bs])

            actions_index = actions[key].astype('int64').unsqueeze(-1)
            q_eval_a[key] = paddle.take_along_axis(q_eval[key], actions_index, axis=-1).reshape([bs])



            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                             avail_actions=avail_actions, agent_key=key)
                actions_index = act_next[key].astype('int64').unsqueeze(-1)
                # q_next_a[key] = paddle.gather_nd(q_next[key], actions_index).reshape([bs])
                q_next_a[key] = paddle.take_along_axis(q_next[key], actions_index, axis=-1).reshape([bs])
            else:
                q_next_a[key] = paddle.max(q_next[key], axis=-1, keepdim=True).reshape([bs])

            q_eval_a[key] *= agent_mask[key].astype('float32')
            q_next_a[key] *= agent_mask[key].astype('float32')

        q_tot_eval = self.policy.Q_tot(q_eval_a, state)
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state_next)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # 计算损失函数
        loss = paddle.mean(paddle.square(q_tot_eval - q_tot_target.detach()))
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.ClipGradByNorm(clip_norm=self.grad_clip_norm)(self.policy.parameters_model)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.scheduler.last_lr

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # 准备训练数据
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(axis=1).reshape([-1, 1])
            terminals_tot = paddle.all(terminals[key], axis=1, keepdim=False).astype('float32').reshape([-1, 1])
        else:
            bs_rnn = batch_size
            rewards_list = itemgetter(*self.agent_keys)(rewards)
            rewards_tot = paddle.stack(rewards_list, axis=1).mean(axis=1).reshape([-1, 1])
            terminals_list = itemgetter(*self.agent_keys)(terminals)
            terminals_tot = paddle.stack(terminals_list, axis=1).all(axis=1).reshape([-1, 1]).astype('float32')

        # 计算个体 Q 值
        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions, rnn_hidden=rnn_hidden)

        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

        q_eval_a, q_next, q_next_a = {}, {}, {}
        for key in self.model_keys:
            actions_index = actions[key].astype('int64').unsqueeze(-1)
            q_eval_a[key] = paddle.gather_nd(q_eval[key][:, :-1], actions_index).reshape([bs_rnn, seq_len])
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = {k: actions_greedy[k].unsqueeze(-1)[:, 1:] for k in self.model_keys}
                q_next_a[key] = paddle.gather_nd(q_next[key], act_next[key].astype('int64').detach()).reshape(
                    [bs_rnn, seq_len])
            else:
                q_next_a[key] = paddle.max(q_next[key], axis=-1, keepdim=True).reshape([bs_rnn, seq_len])

            q_eval_a[key] = q_eval_a[key] * agent_mask[key].astype('float32')
            q_next_a[key] = q_next_a[key] * agent_mask[key].astype('float32')

            if self.use_parameter_sharing:
                q_eval_a[key] = q_eval_a[key].reshape([batch_size, self.n_agents, seq_len]).transpose(
                    [0, 2, 1]).reshape([-1, self.n_agents])
                q_next_a[key] = q_next_a[key].reshape([batch_size, self.n_agents, seq_len]).transpose(
                    [0, 2, 1]).reshape([-1, self.n_agents])
            else:
                q_eval_a[key] = q_eval_a[key].reshape([-1, 1])
                q_next_a[key] = q_next_a[key].reshape([-1, 1])

        # 计算总 Q 值
        q_tot_eval = self.policy.Q_tot(q_eval_a, state[:, :-1].reshape([batch_size * seq_len, -1]))
        q_tot_next = self.policy.Qtarget_tot(q_next_a, state[:, 1:].reshape([batch_size * seq_len, -1]))
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # 计算损失函数
        td_errors = (q_tot_eval - q_tot_target.detach()) * filled
        loss = paddle.sum(paddle.square(td_errors)) / paddle.sum(filled)
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.ClipGradByNorm(clip_norm=self.grad_clip_norm)(self.policy.parameters_model)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.scheduler.last_lr

        info.update({
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        })

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info