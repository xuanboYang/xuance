import oneflow as flow
from argparse import Namespace
from xuance.environment import DummyVecEnv
from oneflow.nn import Module
from xuance.oneflow.utils import NormalizeFunctions, ActivationFunctions
from xuance.oneflow.policies import REGISTRY_Policy
from xuance.oneflow.agents.qlearning_family.dqn_agent import DQN_Agent


class C51_Agent(DQN_Agent):
    """The implementation of C51DQN agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(C51_Agent, self).__init__(config, envs)

    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = flow.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "C51_Q_network":
            policy = REGISTRY_Policy["C51_Q_network"](
                action_space=self.action_space,
                atom_num=self.config.atom_num, v_min=self.config.v_min, v_max=self.config.v_max,
                representation=representation, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training)
        else:
            raise AttributeError(f"C51 currently does not support the policy named {self.config.policy}.")

        return policy

