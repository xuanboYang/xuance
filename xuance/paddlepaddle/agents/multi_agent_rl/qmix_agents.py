import paddle
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from xuance.paddlepaddle import Layer
from xuance.paddlepaddle.utils import NormalizeFunctions, ActivationFunctions
from xuance.paddlepaddle.policies import REGISTRY_Policy, QMIX_mixer
from xuance.paddlepaddle.agents import OffPolicyMARLAgents


class QMIX_Agents(OffPolicyMARLAgents):
    """The implementation of QMIX agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(QMIX_Agents, self).__init__(config, envs)
        self.state_space = envs.state_space
        self.use_global_state = True

        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)

        # build policy, optimizers, schedulers
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.model_keys, self.agent_keys, self.policy)

    def _build_policy(self) -> Layer:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = paddle.nn.initializer.Orthogonal()
        activation = ActivationFunctions[self.config.activation]

        # build representations
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents, self.device)
        if self.config.policy == "Mixing_Q_network":
            policy = REGISTRY_Policy["Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=self.device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"QMIX currently does not support the policy named {self.config.policy}.")

        return policy
