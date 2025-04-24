"""
DCG: Deep coordination graphs
Paper link: http://proceedings.mlr.press/v119/boehmer20a/boehmer20a.pdf
Implementation: Pytorch
"""
import paddle
from paddle import nn
from operator import itemgetter
from xuance.paddlepaddle.learners import LearnerMAS
from xuance.common import List
from argparse import Namespace
# try:
#     import paddle_scatter
# except ImportError:
#     print("The module torch_scatter is not installed.")
from paddle.optimizer.lr import LinearWarmup


class DCG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Layer):
        super(DCG_Learner, self).__init__(config, model_keys, agent_keys, policy)
        # self.optimizer = torch.optim.Adam(self.policy.parameters_model, self.learning_rate, eps=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
        #                                                    start_factor=1.0,
        #                                                    end_factor=self.end_factor_lr_decay,
        #                                                    total_iters=self.config.running_steps)
        # 定义 Adam 优化器
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate,
            parameters=self.policy.parameters_model,
            epsilon=1e-5
        )

        # 定义线性学习率调度器
        self.scheduler = LinearWarmup(
            learning_rate=self.learning_rate,
            warmup_steps=self.config.running_steps,
            start_lr=self.learning_rate,
            end_lr=self.learning_rate * self.end_factor_lr_decay,
            verbose=False
        )

        self.dim_hidden_state = policy.representation[self.model_keys[0]].output_shapes['state'][0]
        self.dim_act = max([self.policy.action_space[key].n for key in agent_keys])
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()

    def get_graph_values(self, hidden_states, use_target_net=False):
        if use_target_net:
            utilities = self.policy.target_utility(hidden_states)
            payoff = self.policy.target_payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        else:
            utilities = self.policy.utility(hidden_states)
            payoff = self.policy.payoffs(hidden_states, self.policy.graph.edges_from, self.policy.graph.edges_to)
        return utilities, payoff

    def act(self, hidden_states, avail_actions=None):
        """
        Calculate the actions via belief propagation.

        Args:
            hidden_states (torch.Tensor): The hidden states for the representation of all agents.
            avail_actions (torch.Tensor): The avail actions for the agents, default is None.

        Returns: The actions.
        """
        with paddle.no_grad():
            f_i, f_ij = self.get_graph_values(hidden_states)
        n_edges = self.policy.graph.n_edges
        n_vertexes = self.policy.graph.n_vertexes
        f_i_mean = f_i.double() / n_vertexes
        f_ij_mean = f_ij.double() / n_edges
        f_ji_mean = f_ij_mean.transpose(dim0=-1, dim1=-2).clone()
        batch_size = f_i.shape[0]

        msg_ij = paddle.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # i -> j (send)
        msg_ji = paddle.zeros(batch_size, n_edges, self.dim_act).to(self.device)  # j -> i (receive)
        #
        # msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
        #                                         dim_size=n_vertexes)
        # msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
        #                                          dim_size=n_vertexes)
        # 初始化输出张量
        n_vertexes = n_vertexes  # 图中的节点数
        msg_forward = paddle.zeros(shape=(msg_ij.shape[0], n_vertexes), dtype=msg_ij.dtype)
        msg_backward = paddle.zeros(shape=(msg_ji.shape[0], n_vertexes), dtype=msg_ji.dtype)

        # 使用 for 循环模拟 scatter_add 操作
        for i in range(msg_ij.shape[1]):  # 遍历边的索引
            msg_forward[:, self.policy.graph.edges_to[i]] += msg_ij[:, i]
            msg_backward[:, self.policy.graph.edges_from[i]] += msg_ji[:, i]

        utility = f_i_mean + msg_forward + msg_backward
        if len(self.policy.graph.edges) != 0:
            for i in range(self.config.n_msg_iterations):
                joint_forward = (utility[:, self.policy.graph.edges_from, :] - msg_ji).unsqueeze(axis=-1) + f_ij_mean
                joint_backward = (utility[:, self.policy.graph.edges_to, :] - msg_ij).unsqueeze(axis=-1) + f_ji_mean
                msg_ij = joint_forward.max(axis=-2)
                msg_ji = joint_backward.max(axis=-2)
                if self.config.msg_normalized:
                    msg_ij -= paddle.mean(msg_ij, axis=-1, keepdim=True)

                    msg_ji -= paddle.mean(msg_ji, axis=-1, keepdim=True)

                # msg_forward = torch_scatter.scatter_add(src=msg_ij, index=self.policy.graph.edges_to, dim=1,
                #                                         dim_size=n_vertexes)
                # msg_backward = torch_scatter.scatter_add(src=msg_ji, index=self.policy.graph.edges_from, dim=1,
                #                                          dim_size=n_vertexes)
                # 初始化 msg_forward 和 msg_backward
                n_vertexes = n_vertexes  # 图中的节点数

                # 沿指定维度进行 scatter_add 操作
                msg_forward = paddle.zeros(shape=(msg_ij.shape[0], n_vertexes), dtype=msg_ij.dtype)
                msg_backward = paddle.zeros(shape=(msg_ji.shape[0], n_vertexes), dtype=msg_ji.dtype)

                # 使用 for 循环模拟 scatter_add（针对每个 batch）
                for b in range(msg_ij.shape[0]):  # 遍历 batch 维度
                    # 更新 msg_forward
                    msg_forward[b] = paddle.bincount(
                        self.policy.graph.edges_to,
                        weights=msg_ij[b],
                        minlength=n_vertexes
                    )

                    # 更新 msg_backward
                    msg_backward[b] = paddle.bincount(
                        self.policy.graph.edges_from,
                        weights=msg_ji[b],
                        minlength=n_vertexes
                    )

                utility = f_i_mean + msg_forward + msg_backward
        if avail_actions is not None:
            avail_actions = paddle.Tensor(avail_actions)
            utility_detach = utility.clone().detach()
            utility_detach[avail_actions == 0] = -1e10
            actions_greedy = utility_detach.argmax(axis=-1)
        else:
            actions_greedy = utility.argmax(axis=-1)
        return actions_greedy

    def q_dcg(self, hidden_states, actions, states=None, use_target_net=False):
        f_i, f_ij = self.get_graph_values(hidden_states, use_target_net=use_target_net)
        f_i_mean = f_i.double() / self.policy.graph.n_vertexes
        f_ij_mean = f_ij.double() / self.policy.graph.n_edges
        utilities = f_i_mean.gather(-1, actions.unsqueeze(dim=-1).long()).sum(dim=1)
        if len(self.policy.graph.edges) == 0 or self.config.n_msg_iterations == 0:
            return utilities
        actions_ij = (actions[:, self.policy.graph.edges_from] * self.dim_act + \
                      actions[:, self.policy.graph.edges_to]).unsqueeze(-1)
        payoffs = f_ij_mean.reshape(list(f_ij_mean.shape[0:-2]) + [-1]).gather(-1, actions_ij.long()).sum(dim=1)
        if self.config.agent == "DCG_S":
            state_value = self.policy.bias(states)
            return utilities + payoffs + state_value
        else:
            return utilities + payoffs

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True if self.config.agent == "DCG_S" else False)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        state_next = sample_Tensor['state_next']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        avail_actions = sample_Tensor['avail_actions']
        avail_actions_next = sample_Tensor['avail_actions_next']

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            rewards_tot = paddle.mean(rewards[key], axis=1).reshape((batch_size, 1))
            # terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape(batch_size, 1)
            # 计算沿指定维度的所有元素是否都为 True
            all_result = paddle.all(terminals[key], axis=1, keepdim=False)
            # 将布尔值转换为浮点数 (True -> 1.0, False -> 0.0)
            float_result = paddle.cast(all_result, dtype='float32')
            # 调整形状为 (batch_size, 1)
            terminals_tot = float_result.reshape([batch_size, 1])

            actions = actions[key].reshape(batch_size, self.n_agents)
            if self.use_actions_mask:
                avail_actions_next = avail_actions_next[key].reshape(batch_size, self.n_agents, -1)
        else:
            rewards_tot = paddle.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(dim=-1, keepdim=True)
            # terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(dim=1, keepdim=True).float()
            # 提取指定键对应的值
            agent_keys = self.agent_keys  # 假设 self.agent_keys 是一个包含键名的列表
            terminals_selected = [terminals[key] for key in agent_keys]
            actions_selected = [actions[key] for key in agent_keys]
            # 使用 paddle.stack 沿指定维度堆叠张量
            terminals_stacked = paddle.stack(terminals_selected, axis=1)
            actions_stacked = paddle.stack(actions_selected, axis=-1)
            # 对 terminals_stacked 沿指定维度进行 all 操作，并转换为浮点数
            terminals_tot = paddle.all(terminals_stacked, axis=1, keepdim=True)
            terminals_tot = paddle.cast(terminals_tot, dtype='float32')

            actions = paddle.stack(itemgetter(*self.agent_keys)(actions), axis=-1)
            if self.use_actions_mask:
                avail_actions_next = paddle.stack(itemgetter(*self.agent_keys)(avail_actions_next), axis=-2)

        _, hidden_states = self.policy.get_hidden_states(batch_size, obs, use_target_net=False)
        q_tot_eval = self.q_dcg(hidden_states, actions, states=state, use_target_net=False)

        _, hidden_states_next = self.policy.get_hidden_states(batch_size, obs_next, use_target_net=False)
        action_next_greedy = paddle.Tensor(self.act(hidden_states_next, avail_actions_next)).to(self.device)
        _, hidden_states_target = self.policy.get_hidden_states(batch_size, obs_next, use_target_net=True)
        q_tot_next = self.q_dcg(hidden_states_target, action_next_greedy, states=state_next, use_target_net=True)

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        # calculate the loss function
        loss = self.mse_loss(q_tot_eval, q_tot_target.detach())
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
            paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_model,
                max_norm=self.grad_clip_norm
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True if self.config.agent == "DCG_S" else False)
        batch_size = sample_Tensor['batch_size']
        seq_len = sample['sequence_length']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        avail_actions = sample_Tensor['avail_actions']
        filled = sample_Tensor['filled'].reshape([-1, 1])

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs_rnn = batch_size * self.n_agents
            rewards_tot = rewards[key].mean(dim=1).reshape([-1, 1])
            terminals_tot = terminals[key].all(dim=1, keepdim=False).float().reshape([-1, 1])
            actions = actions[key].reshape(batch_size, self.n_agents, seq_len).transpose(1, 2)
            if self.use_actions_mask:
                avail_actions = avail_actions[key].reshape(batch_size, self.n_agents, seq_len + 1, -1).transpose(1, 2)
        else:
            bs_rnn = batch_size
            rewards_tot = paddle.stack(itemgetter(*self.agent_keys)(rewards), axis=1).mean(axis=1).reshape((-1, 1))
            # terminals_tot = torch.stack(itemgetter(*self.agent_keys)(terminals), dim=1).all(1).reshape([-1, 1]).float()
            # 提取指定键对应的值（等价于 itemgetter）
            selected_terminals = [terminals[key] for key in self.agent_keys]

            # 沿指定维度堆叠张量（等价于 torch.stack）
            stacked_terminals = paddle.stack(selected_terminals, axis=1)

            # 计算沿指定维度的所有元素是否都为 True（等价于 .all(1)）
            all_result = paddle.all(stacked_terminals, axis=1)

            # 调整形状为 [-1, 1]（等价于 .reshape([-1, 1])）
            reshaped_result = all_result.reshape([-1, 1])

            # 将布尔值转换为浮点数（等价于 .float()）
            terminals_tot = paddle.cast(reshaped_result, dtype='float32')

            actions = paddle.stack(itemgetter(*self.agent_keys)(actions), axis=-1)
            if self.use_actions_mask:
                avail_actions = paddle.stack(itemgetter(*self.agent_keys)(avail_actions), axis=-2)

        rnn_hidden = {k: self.policy.representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, hidden_states = self.policy.get_hidden_states(batch_size, obs, rnn_hidden, use_target_net=False)
        state_current = state[:, :-1] if self.config.agent == "DCG_S" else None
        state_next = state[:, 1:] if self.config.agent == "DCG_S" else None
        q_tot_eval = self.q_dcg(hidden_states[:, :-1].reshape(batch_size * seq_len, self.n_agents, -1),
                                actions.reshape(batch_size * seq_len, self.n_agents),
                                states=state_current, use_target_net=False)

        if self.use_actions_mask:
            avail_a_next = avail_actions[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1)
        else:
            avail_a_next = None
        hidden_states_next = hidden_states[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1)
        action_next_greedy = paddle.Tensor(self.act(hidden_states_next, avail_actions=avail_a_next)).to(self.device)
        rnn_hidden_target = {k: self.policy.target_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        _, hidden_states_tar = self.policy.get_hidden_states(batch_size, obs, rnn_hidden_target, use_target_net=True)
        q_tot_next = self.q_dcg(hidden_states_tar[:, 1:].reshape(batch_size * seq_len, self.n_agents, -1),
                                action_next_greedy, states=state_next, use_target_net=True)

        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next
        td_error = (q_tot_eval - q_tot_target.detach()) * filled

        # calculate the loss function
        loss = (td_error ** 2).sum() / filled.sum()
        self.optimizer.clear_grad()
        loss.backward()
        if self.use_grad_clip:
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
            paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_model,
                max_norm=self.grad_clip_norm
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_tot_eval.mean().item()
        }

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        return info
