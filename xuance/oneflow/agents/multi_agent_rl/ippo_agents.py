import oneflow as flow
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from oneflow.nn import Module
from xuance.oneflow.utils import NormalizeFunctions, ActivationFunctions
from xuance.oneflow.policies import REGISTRY_Policy
from xuance.oneflow.agents import OnPolicyMARLAgents


class IPPO_Agents(OnPolicyMARLAgents):
    """The implementation of Independent PPO agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(IPPO_Agents, self).__init__(config, envs)

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self) -> Module:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (flow.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = flow.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device
        agent = self.config.agent
        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        C_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        # build policies
        if self.config.policy == "Categorical_MAAC_Policy":
            policy = REGISTRY_Policy["Categorical_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = False
        elif self.config.policy == "Gaussian_MAAC_Policy":
            policy = REGISTRY_Policy["Gaussian_MAAC_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                representation_actor=A_representation, representation_critic=C_representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                activation_action=ActivationFunctions[self.config.activation_action],
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
            self.continuous_control = True
        else:
            raise AttributeError(f"{agent} currently does not support the policy named {self.config.policy}.")
        return policy
