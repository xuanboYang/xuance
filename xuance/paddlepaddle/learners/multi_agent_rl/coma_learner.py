"""
COMA: Counterfactual Multi-Agent Policy Gradients
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/11794
Implementation: Pytorch
"""
import paddle
from paddle import nn
from paddle.nn.functional import one_hot
from xuance.common import List
from argparse import Namespace
from xuance.paddlepaddle.learners.multi_agent_rl.iac_learner import IAC_Learner
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer import Adam
from paddle.optimizer.lr import LinearLR


class COMA_Learner(IAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Layer):
        config.use_value_clip, config.value_clip_range = False, None
        config.use_huber_loss, config.huber_delta = False, None
        config.use_value_norm = False
        config.vf_coef, config.ent_coef = None, None
        super(COMA_Learner, self).__init__(config, model_keys, agent_keys, policy)
        self.sync_frequency = config.sync_frequency
        self.n_actions = {k: self.policy.action_space[k].n for k in self.model_keys}
        self.mse_loss = nn.MSELoss()

    def build_optimizer(self):
        # self.optimizer = {
        #     'actor': torch.optim.Adam(self.policy.parameters_actor, self.config.learning_rate_actor, eps=1e-5),
        #     'critic': torch.optim.Adam(self.policy.parameters_critic, self.config.learning_rate_critic, eps=1e-5)
        # }
        # self.scheduler = {
        #     'actor': torch.optim.lr_scheduler.LinearLR(self.optimizer['actor'],
        #                                                start_factor=1.0,
        #                                                end_factor=self.end_factor_lr_decay,
        #                                                total_iters=self.config.running_steps),
        #     'critic': torch.optim.lr_scheduler.LinearLR(self.optimizer['critic'],
        #                                                 start_factor=1.0,
        #                                                 end_factor=self.end_factor_lr_decay,
        #                                                 total_iters=self.config.running_steps)
        # }
        # 定义优化器
        self.optimizer = {
            'actor': Adam(
                learning_rate=self.config.learning_rate_actor,
                parameters=self.policy.parameters_actor,
                epsilon=1e-5
            ),
            'critic': Adam(
                learning_rate=self.config.learning_rate_critic,
                parameters=self.policy.parameters_critic,
                epsilon=1e-5
            )
        }

        # 定义线性学习率调度器
        self.scheduler = {
            'actor': LinearLR(
                learning_rate=self.config.learning_rate_actor,
                total_steps=self.config.running_steps,
                start_factor=1.0,
                end_factor=self.end_factor_lr_decay,
                verbose=False
            ),
            'critic': LinearLR(
                learning_rate=self.config.learning_rate_critic,
                total_steps=self.config.running_steps,
                start_factor=1.0,
                end_factor=self.end_factor_lr_decay,
                verbose=False
            )
        }

    def update(self, sample, epsilon=0.0):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        agent_mask = sample_Tensor['agent_mask']
        avail_actions = sample_Tensor['avail_actions']
        returns = sample_Tensor['returns']
        IDs = sample_Tensor['agent_ids']

        bs = batch_size * self.n_agents if self.use_parameter_sharing else batch_size

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions, epsilon=epsilon)
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_onehot = {key: one_hot(actions[key].astype("int64"), self.n_actions[key])}
        else:
            # IDs = paddle.eye(self.n_agents).unsqueeze(0).repeat(batch_size, 1, 1).reshape((bs, -1)).to(self.device)
            # 生成单位矩阵并调整形状
            IDs = paddle.eye(self.n_agents).unsqueeze(0)  # 形状变为 (1, self.n_agents, self.n_agents)
            # 在第 0 维重复扩展
            IDs = paddle.tile(IDs, repeat_times=[batch_size, 1, 1])  # 形状变为 (batch_size, self.n_agents, self.n_agents)
            # 重塑张量形状
            IDs = IDs.reshape((bs, -1))  # 形状变为 (bs, self.n_agents * self.n_agents)
            # 将张量移动到指定设备
            if self.device is not None:
                IDs = IDs.to(self.device)

            actions_onehot = {k: one_hot(actions[k].astype("int64"), self.n_actions[k]) for k in self.agent_keys}

        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {k: values_pred.reshape((bs, -1)) for k in self.model_keys}
        else:
            values_pred_dict = {k: values_pred[:, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key]

            # 确保actions_onehot的数据类型正确
            # probs需要整型索引
            pi_probs = pi_dist_dict[key].probs(actions_onehot[key].astype("int64"))
            if self.use_actions_mask:
                pi_probs[avail_actions[key] == 0] = 0

           
            baseline = (pi_probs * values_pred_dict[key]).sum(-1).reshape((bs,))
            # pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).astype("int64"))
            # 确保 actions[key] 的形状正确
            actions_index = paddle.unsqueeze(actions[key].astype("int64"), axis=-1)  # 增加最后一维
            # 使用 paddle.gather 提取对应概率
            # print(f"Shape of pi_probs: {pi_probs.shape}")
            # print(f"Shape of actions[key]: {actions[key].shape}")
            # print(f"Shape of actions_index: {actions_index.shape}")
            # max_action = paddle.max(actions[key])
            # print(f"Maximum action value: {max_action.item()}")
            # print(f"pi_probs second dimension size: {pi_probs.shape[-1]}")
            # 确保 actions[key] 的数据类型为 int64
            # if actions[key].dtype not in [paddle.int32, paddle.int64]:
            #     print(f"Warning: actions[key] dtype is {actions[key].dtype}, converting to int64.")
            #     actions[key] = actions[key].astype("int64")

            # # 检查 pi_probs 是否包含无效行
            # valid_mask = paddle.any(pi_probs != 0, axis=-1)  # 检查每一行是否有非零值
            # pi_probs_valid = pi_probs[valid_mask]
            # actions_index_valid = actions_index[valid_mask]

            # 使用 paddle.gather 提取对应概率
            # pi_taken = paddle.gather(pi_probs_valid, index=actions_index_valid, axis=-1)
            pi_taken = paddle.take_along_axis(pi_probs, actions_index, axis=-1)
            # 输出结果
            # print(f"pi_taken: {pi_taken}, Shape: {pi_taken.shape}")

            # pi_taken = paddle.gather(pi_probs, index=actions_index, axis=-1)
            # # 输出结果
            # print(f"pi_taken shape: {pi_taken.shape}")

            # q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).astype("int64")).reshape((bs,))
            # 使用 paddle.gather 提取对应概率
            # q_taken = paddle.gather(values_pred_dict[key], index=actions_index, axis=-1)
            q_taken = paddle.take_along_axis(values_pred_dict[key], actions_index, axis=-1)

            log_pi_taken = paddle.log(pi_taken).reshape((bs, -1))
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].clear_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            # info["gradient_norm_actor"] = grad_norm.item()
            grad_norm = paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_critic,
                max_norm=self.grad_clip_norm
            )
            # 将梯度范数记录到 info 字典中
            info["gradient_norm_actor"] = float(grad_norm)

        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].clear_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            # info["gradient_norm_actor"] = grad_norm.item()
            grad_norm = paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_actor,
                max_norm=self.grad_clip_norm
            )
            # 将梯度范数记录到 info 字典中
            info["gradient_norm_actor"] = float(grad_norm)

        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].get_lr()
        learning_rate_critic = self.optimizer['critic'].get_lr()

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        }

        return info

    def update_rnn(self, sample, epsilon=0.0):
        self.iterations += 1
        info = {}

        sample_Tensor = self.build_training_data(sample=sample,
                                                 use_parameter_sharing=self.use_parameter_sharing,
                                                 use_actions_mask=self.use_actions_mask,
                                                 use_global_state=True)
        batch_size = sample_Tensor['batch_size']
        state = sample_Tensor['state']
        bs_rnn = batch_size * self.n_agents if self.use_parameter_sharing else batch_size
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        returns = sample_Tensor['returns']
        avail_actions = sample_Tensor['avail_actions']
        agent_mask = sample_Tensor['agent_mask']
        filled = sample_Tensor['filled']
        seq_len = filled.shape[1]
        IDs = sample_Tensor['agent_ids']

        if self.use_parameter_sharing:
            filled = filled.unsqueeze(1).expand(batch_size, self.n_agents, seq_len).reshape((bs_rnn, seq_len))
        else:
            # IDs = paddle.eye(self.n_agents).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1).to(self.device)
            # 生成单位矩阵并调整形状
            IDs = paddle.eye(self.n_agents).unsqueeze(0).unsqueeze(0)  # 形状变为 (1, 1, self.n_agents, self.n_agents)
            # 在前两个维度重复扩展
            IDs = paddle.tile(IDs, repeat_times=[batch_size, seq_len, 1,
                                                 1])  # 形状变为 (batch_size, seq_len, self.n_agents, self.n_agents)
            # 将张量移动到指定设备
            if self.device is not None:
                IDs = IDs.to(self.device)

        rnn_hidden_actor = {k: self.policy.actor_representation[k].init_hidden(bs_rnn) for k in self.model_keys}
        rnn_hidden_critic = {k: self.policy.critic_representation[k].init_hidden(bs_rnn) for k in self.model_keys}

        # feedforward
        _, pi_dist_dict = self.policy(observation=obs, agent_ids=IDs, avail_actions=avail_actions,
                                      rnn_hidden=rnn_hidden_actor, epsilon=epsilon)
        actions_onehot = {k: one_hot(actions[k].astype("int64"), self.n_actions[k]) for k in self.model_keys}
        _, values_pred = self.policy.get_values(state=state, observation=obs, actions=actions_onehot,
                                                agent_ids=IDs, rnn_hidden=rnn_hidden_critic, target=False)

        if self.use_parameter_sharing:
            values_pred_dict = {self.model_keys[0]: values_pred.transpose(1, 2).reshape((bs_rnn, seq_len, -1))}
        else:
            values_pred_dict = {k: values_pred[:, :, i] for i, k in enumerate(self.model_keys)}

        # calculate loss
        loss_a, loss_c = [], []
        for key in self.model_keys:
            mask_values = agent_mask[key] * filled

            pi_probs = pi_dist_dict[key].probs
            if self.use_actions_mask:
                pi_probs[avail_actions[key] == 0] = 0
            baseline = (pi_probs * values_pred_dict[key]).sum(-1).reshape((bs_rnn, seq_len))
            pi_taken = pi_probs.gather(-1, actions[key].unsqueeze(-1).astype("int64"))
            q_taken = values_pred_dict[key].gather(-1, actions[key].unsqueeze(-1).astype("int64")).reshape(
                (bs_rnn, seq_len))
            log_pi_taken = paddle.log(pi_taken).reshape((bs_rnn, seq_len))
            advantages = (q_taken - baseline).detach()
            loss_a.append(-(advantages * log_pi_taken * mask_values).sum() / mask_values.sum())

            td_error = (q_taken - returns[key].detach()) * mask_values
            loss_c.append((td_error ** 2).sum() / mask_values.sum())

        # update critic
        loss_critic = sum(loss_c)
        self.optimizer['critic'].clear_grad()
        loss_critic.backward()
        if self.use_grad_clip:
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.grad_clip_norm)
            # info["gradient_norm_actor"] = grad_norm.item()
            grad_norm = paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_critic,
                max_norm=self.grad_clip_norm
            )
            # 将梯度范数记录到 info 字典中
            info["gradient_norm_actor"] = float(grad_norm)

        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # update actor(s)
        loss_coma = sum(loss_a)
        self.optimizer['actor'].clear_grad()
        loss_coma.backward()
        if self.use_grad_clip:
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.grad_clip_norm)
            # info["gradient_norm_actor"] = grad_norm.item()
            grad_norm = paddle.nn.utils.clip_grad_norm_(
                parameters=self.policy.parameters_actor,
                max_norm=self.grad_clip_norm
            )
            # 将梯度范数记录到 info 字典中
            info["gradient_norm_actor"] = float(grad_norm)

        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Logger
        learning_rate_actor = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        learning_rate_critic = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": learning_rate_actor,
            "learning_rate_critic": learning_rate_critic,
            "actor_loss": loss_coma.item(),
            "critic_loss": loss_critic.item(),
            "advantage": advantages.mean().item(),
        }

        return info
