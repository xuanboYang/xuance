import oneflow as flow
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from oneflow.nn import Module
from xuance.oneflow.utils import NormalizeFunctions, ActivationFunctions
from xuance.oneflow.policies import REGISTRY_Policy, VDN_mixer
from xuance.oneflow.agents import OffPolicyMARLAgents


class VDN_Agents(OffPolicyMARLAgents):
    """The implementation of Value Decomposition Networks (VDN) agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(VDN_Agents, self).__init__(config, envs)

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

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

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        mixer = VDN_mixer()
        if self.config.policy == "Mixing_Q_network":
            policy = REGISTRY_Policy["Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"VDN currently does not support the policy named {self.config.policy}.")

        return policy
