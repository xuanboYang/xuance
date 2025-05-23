import paddle
from argparse import Namespace
from xuance.environment import DummyVecEnv
from xuance.paddlepaddle import Layer
from xuance.paddlepaddle.utils import NormalizeFunctions, ActivationFunctions
from xuance.paddlepaddle.policies import REGISTRY_Policy
from xuance.paddlepaddle.agents.qlearning_family.dqn_agent import DQN_Agent


class DuelDQN_Agent(DQN_Agent):
    """The implementation of DuelDQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(DuelDQN_Agent, self).__init__(config, envs)

    def _build_policy(self) -> Layer:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = paddle.nn.initializer.Orthogonal()
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "Duel_Q_network":
            policy = REGISTRY_Policy["Duel_Q_network"](
                action_space=self.action_space, representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(f"{self.config.agent} currently does not support the policy named {self.config.policy}.")

        return policy
