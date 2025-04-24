"""
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
Paper link:
http://proceedings.mlr.press/v97/son19a/son19a.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from xuance.paddlepaddle.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
from operator import itemgetter
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearWarmup


class QTRAN_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Layer):
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        super(QTRAN_Learner, self).__init__(config, model_keys, agent_keys, policy)
        # self.optimizer = torch.optim.Adam(self.policy.parameters_model, config.learning_rate, eps=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
        #                                                    start_factor=1.0,
        #                                                    end_factor=self.end_factor_lr_decay,
        #                                                    total_iters=self.config.running_steps)
        # 定义学习率调度器
        # scheduler = LinearWarmup(
        #     learning_rate=config.learning_rate,  # 初始学习率
        #     warmup_steps=self.config.running_steps,  # 总步数（对应 PyTorch 的 total_iters）
        #     start_lr=config.learning_rate * 1.0,  # 起始学习率（start_factor=1.0）
        #     end_lr=config.learning_rate * self.end_factor_lr_decay,  # 结束学习率（end_factor）
        #     verbose=False
        # )
        # # 定义优化器
        # self.optimizer = Adam(
        #     learning_rate=scheduler,  # 使用调度器作为学习率
        #     epsilon=1e-5,  # 对应 PyTorch 的 eps 参数
        #     parameters=self.policy.parameters_model
        # )
        self.scheduler = paddle.optimizer.lr.LinearLR(
            learning_rate=config.learning_rate,
            start_factor=config.learning_rate * 1.0,  # 起始学习率（start_factor=1.0）
            end_factor=config.learning_rate * self.end_factor_lr_decay,  # 结束学习率（end_factor）
            total_steps=self.config.running_steps,
            verbose=False
        )
        # 定义优化器和学习率调度器
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.policy.parameters_model,
            learning_rate=self.scheduler,
            epsilon=1e-5
        )

        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
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

        # print("sample_Tensor keys:", sample_Tensor.keys())
        # print("model_keys:", self.model_keys)
        print("agent_mask keys:", agent_mask)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(axis=1).reshape((batch_size, 1))
            terminals_tot = paddle.all(terminals[key], axis=1, keepdim=True).astype('float32').reshape((batch_size, 1))
        else:
            bs = batch_size
            rewards_tot = paddle.stack([rewards[k] for k in self.agent_keys], axis=1).mean(axis=-1, keepdim=True)
            terminals_tot = paddle.stack([terminals[k] for k in self.agent_keys], axis=1).all(axis=1,
                                                                                              keepdim=True).astype(
                'float32')

        _, hidden_state, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions)
        _, hidden_state_next, q_next = self.policy.Qtarget(obs_next, agent_ids=IDs)

        q_eval_a, q_eval_greedy_a, q_next_a = {}, {}, {}
        actions_next_greedy = {}
        for key in self.model_keys:
            # q_eval_a[key] = q_eval[key].gather(-1, actions[key].astype('int64').unsqueeze(-1)).reshape((bs,))
            print("q_eval[key] shape:", q_eval[key].shape)  # [32, 5]
            print("actions[key] shape:", actions[key].shape)
            print("actions[key].unsqueeze(-1) shape:", actions[key].unsqueeze(-1).shape)

            # q_eval[key] 形状[32, 5]
            # actions[key] 形状 [32]
            indices = actions[key].astype('int64').unsqueeze(-1)  # 形状变为 [32, 1]
            q_eval_a[key] = paddle.take_along_axis(q_eval[key], indices=indices, axis=-1).reshape((bs,)).astype(
                'float32')

            # q_eval_greedy_a[key] = q_eval[key].gather(paddle.to_tensor([-1], dtype='int64'),
            #                                           actions_greedy[key].astype('int64').unsqueeze(-1)).reshape(
            #     (bs,))
            q_eval_greedy_a[key] = paddle.take_along_axis(q_eval[key], 
                                                          indices=actions_greedy[key].astype('int64').unsqueeze(-1), axis=-1).reshape((bs,)).astype('float32')

            if self.use_actions_mask:
                q_next[key][avail_actions_next[key] == 0] = -1e10

            if self.config.double_q:
                _, _, act_next, _ = self.policy(observation=obs_next, agent_ids=IDs,
                                                avail_actions=avail_actions, agent_key=key)
                actions_next_greedy[key] = act_next[key]
                # q_next_a[key] = q_next[key].gather(-1, act_next[key].astype('int64').unsqueeze(-1)).reshape((bs,))

                print("q_next[key] shape:", q_next[key].shape)  # [32, 5]
                print("act_next[key] shape:", act_next[key].shape)
                print("act_next[key].unsqueeze(-1) shape:", act_next[key].unsqueeze(-1).shape)

                q_next_a[key] = paddle.take_along_axis(q_next[key], indices=act_next[key].astype('int64').unsqueeze(-1),
                                                       axis=-1).reshape((bs,)).astype('float32')

            else:
                actions_next_greedy[key] = paddle.argmax(q_next[key], axis=-1, keepdim=False)
                q_next_a[key] = paddle.max(q_next[key], axis=-1, keepdim=True).values.reshape((bs,)).astype('float32')

            q_eval_a[key] *= (agent_mask[key].astype('float32'))
            q_eval_greedy_a[key] *= (agent_mask[key].astype('float32'))
            q_next_a[key] *= (agent_mask[key].astype('float32'))

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
            q_joint_next, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy, agent_mask)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            loss_td = paddle.nn.functional.mse_loss(q_joint, y_dqn.detach())  # TD loss

            # -- Opt Loss --
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            error_opt = q_tot_greedy - q_joint_greedy_hat.detach() + v_joint
            loss_opt = paddle.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_tot = self.policy.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = paddle.clip(error_nopt, max=0)
            loss_nopt = paddle.mean(error_nopt ** 2)  # NOPT loss

            info["Q_joint"] = paddle.mean(q_joint).item()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.policy.Q_tran(state, hidden_state, actions, agent_mask)
            actions_choosen = paddle.concat([actions[k] for k in self.model_keys], axis=0).reshape(
                (-1, self.n_agents, 1))
            q_joint_choosen = q_count.gather(-1, actions_choosen.astype('int64')).reshape((-1, self.n_agents))
            q_next_count, _ = self.policy.Q_tran_target(state_next, hidden_state_next, actions_next_greedy, agent_mask)
            actions_next_choosen = paddle.concat([actions_next_greedy[k] for k in self.model_keys], axis=0).reshape(
                (-1, self.n_agents, 1))
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.astype('int64')).reshape(
                (-1, self.n_agents))

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            loss_td = paddle.nn.functional.mse_loss(q_joint_choosen, y_dqn.detach())  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state, hidden_state, actions_greedy, agent_mask)
            actions_greedy_current = paddle.concat([actions_greedy[k] for k in self.model_keys], axis=0).reshape(
                (-1, self.n_agents, 1))
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(-1, actions_greedy_current.astype('int64')).reshape(
                (-1, self.n_agents))
            error_opt = q_tot_greedy - q_joint_greedy_hat_all.detach() + v_joint
            loss_opt = paddle.mean(error_opt ** 2)  # Opt loss

            # -- Nopt Loss --
            q_eval_count = paddle.concat([q_eval[k] for k in self.model_keys], axis=0).reshape(
                (batch_size * self.n_agents, -1))
            q_sums = paddle.concat([q_eval_a[k] for k in self.model_keys], axis=0).reshape((-1, self.n_agents))
            q_sums_repeat = q_sums.unsqueeze(axis=1).repeat(1, self.n_agents, 1)
            agent_mask_diag = (1 - paddle.eye(self.n_agents, dtype='float32')).unsqueeze(0).repeat(batch_size, 1, 1)
            q_sum_mask = paddle.sum(q_sums_repeat * agent_mask_diag, axis=-1)
            q_count_for_nopt = q_count.reshape((batch_size * self.n_agents, -1))
            v_joint_repeated = v_joint.repeat_interleave(self.n_agents, axis=0).unsqueeze(-1)
            error_nopt = q_eval_count + q_sum_mask.flatten().unsqueeze(
                -1) - q_count_for_nopt.detach() + v_joint_repeated
            error_nopt_min = paddle.min(error_nopt, axis=-1).values
            loss_nopt = paddle.mean(error_nopt_min ** 2)  # NOPT loss

            info["Q_joint"] = paddle.mean(q_joint_choosen).item()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.get_lr()

        info.update({
            "learning_rate": lr,
            "loss_td": loss_td.item(),
            "loss_opt": loss_opt.item(),
            "loss_nopt": loss_nopt.item(),
            "loss": loss.item()
        })

        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
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
        filled_n = filled.repeat(1, self.n_agents)
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape([-1, 1])
        else:
            bs_rnn = batch_size
            rewards_tot = paddle.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(axis=1).reshape((-1, 1))
            terminals_tot = paddle.cast(
                paddle.stack(itemgetter(*self.agent_keys)(terminals), axis=1).all(1).reshape([-1, 1]), dtype="float32")

        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, hidden_state, actions_greedy, q_eval = self.policy(obs, agent_ids=IDs, avail_actions=avail_actions,
                                                              rnn_hidden=rnn_hidden)
        target_rnn_hidden = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, hidden_state_next, q_next_seq = self.policy.Qtarget(obs, agent_ids=IDs, rnn_hidden=target_rnn_hidden)

        q_eval_a, q_eval_greedy_a, q_next, q_next_a = {}, {}, {}, {}
        actions_greedy_eval, actions_next_greedy = {}, {}
        for key in self.model_keys:
            hidden_state[key] = hidden_state[key][:, :-1]
            hidden_state_next[key] = hidden_state_next[key][:, :-1]
            actions_greedy_eval[key] = actions_greedy[key][:, :-1]
            q_eval_a[key] = q_eval[key][:, :-1].gather(-1, actions[key].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_eval_greedy_a[key] = q_eval[key][:, :-1].gather(
                -1, actions_greedy[key][:, :-1].long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
            q_next[key] = q_next_seq[key][:, 1:]

            if self.use_actions_mask:
                q_next[key][avail_actions[key][:, 1:] == 0] = -1e10

            if self.config.double_q:
                act_next = actions_greedy[key][:, 1:]
                q_next_a[key] = q_next[key].gather(-1, act_next.long().unsqueeze(-1)).reshape(bs_rnn, seq_len)
                actions_next_greedy[key] = act_next
            else:
                actions_next_greedy[key] = q_next[key].argmax(axis=-1, keepdim=False)
                q_next_a[key] = q_next[key].max(axis=-1, keepdim=True).values.reshape(bs_rnn, seq_len)

            q_eval_a[key] *= agent_mask[key]
            q_eval_greedy_a[key] *= agent_mask[key]
            q_next_a[key] *= agent_mask[key]

        if self.config.agent == "QTRAN_base":
            # -- TD Loss --
            q_joint, v_joint = self.policy.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
            q_joint_next, _ = self.policy.Q_tran_target(state[:, 1:], hidden_state_next,
                                                        actions_next_greedy, agent_mask)
            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next
            td_error = (q_joint - y_dqn.detach()) * filled
            loss_td = (td_error ** 2).sum() / filled.sum()  # TD loss

            # -- Opt Loss --
            # Argmax across the current agents' actions
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state[:, :-1], hidden_state, actions_greedy_eval, agent_mask)
            error_opt = (q_tot_greedy - q_joint_greedy_hat.detach() + v_joint) * filled
            loss_opt = (error_opt ** 2).sum() / filled.sum()  # Opt loss

            # -- Nopt Loss --
            q_tot = self.policy.Q_tot(q_eval_a)
            q_joint_hat = q_joint
            error_nopt = q_tot - q_joint_hat.detach() + v_joint
            error_nopt = error_nopt.clamp(max=0) * filled
            loss_nopt = (error_nopt ** 2).sum() / filled.sum()  # NOPT loss

            info["Q_joint"] = q_joint.mean().item()

        elif self.config.agent == "QTRAN_alt":
            # -- TD Loss -- (Computed for all agents)
            q_count, v_joint = self.policy.Q_tran(state[:, :-1], hidden_state, actions, agent_mask)
            actions_choosen = itemgetter(*self.model_keys)(actions)
            actions_choosen = actions_choosen.reshape(-1, self.n_agents, 1)
            q_joint_choosen = q_count.gather(-1, actions_choosen.long()).reshape(-1, self.n_agents)
            q_next_count, _ = self.policy.Q_tran_target(state[:, 1:], hidden_state_next, actions_next_greedy,
                                                        agent_mask)
            actions_next_choosen = itemgetter(*self.model_keys)(actions_next_greedy)
            actions_next_choosen = actions_next_choosen.reshape(-1, self.n_agents, 1)
            q_joint_next_choosen = q_next_count.gather(-1, actions_next_choosen.long()).reshape(-1, self.n_agents)

            y_dqn = rewards_tot + (1 - terminals_tot) * self.gamma * q_joint_next_choosen
            td_errors = (q_joint_choosen - y_dqn.detach()) * filled_n
            loss_td = (td_errors ** 2).sum() / filled_n.sum()  # TD loss

            # -- Opt Loss -- (Computed for all agents)
            q_tot_greedy = self.policy.Q_tot(q_eval_greedy_a)
            q_joint_greedy_hat, _ = self.policy.Q_tran(state[:, :-1], hidden_state, actions_greedy_eval, agent_mask)
            actions_greedy_current = itemgetter(*self.model_keys)(actions_greedy_eval)
            actions_greedy_current = actions_greedy_current.reshape(-1, self.n_agents, 1)
            q_joint_greedy_hat_all = q_joint_greedy_hat.gather(
                -1, actions_greedy_current.long()).reshape(-1, self.n_agents)
            error_opt = (q_tot_greedy - q_joint_greedy_hat_all.detach() + v_joint) * filled_n
            loss_opt = (error_opt ** 2).sum() / filled_n.sum()  # Opt loss

            # -- Nopt Loss --
            q_eval_count = itemgetter(*self.model_keys)(q_eval)[:, :-1].reshape(batch_size, self.n_agents, seq_len, -1)
            q_eval_count = q_eval_count.transpose(1, 2).reshape(batch_size * seq_len * self.n_agents, -1)
            q_sums = itemgetter(*self.model_keys)(q_eval_a).reshape(batch_size, self.n_agents, seq_len)
            q_sums = q_sums.transpose(1, 2).reshape(batch_size * seq_len, self.n_agents)
            q_sums_repeat = q_sums.unsqueeze(axis=1).repeat(1, self.n_agents, 1)
            agent_mask_diag = (1 - paddle.eye(self.n_agents, dtype=paddle.float32).to(self.device)).unsqueeze(0).repeat(
                batch_size * seq_len, 1, 1)
            q_sum_mask = (q_sums_repeat * agent_mask_diag).sum(axis=-1)
            q_count_for_nopt = q_count.view(batch_size * seq_len * self.n_agents, -1)
            v_joint_repeated = v_joint.repeat(1, self.n_agents).view(-1, 1)
            error_nopt = q_eval_count + q_sum_mask.view(-1, 1) - q_count_for_nopt.detach() + v_joint_repeated
            error_nopt_min = paddle.min(error_nopt, axis=-1).values * filled_n.reshape(-1)
            loss_nopt = (error_nopt_min ** 2).sum() / filled_n.sum()  # NOPT loss

            info["Q_joint"] = q_joint_choosen.mean().item()

        else:
            raise ValueError("Mixer {} not recognised.".format(self.config.agent))

        # calculate the loss function
        loss = loss_td + self.config.lambda_opt * loss_opt + self.config.lambda_nopt * loss_nopt
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "loss_td": loss_td.item(),
            "loss_opt": loss_opt.item(),
            "loss_nopt": loss_nopt.item(),
            "loss": loss.item()
        })

        return info
