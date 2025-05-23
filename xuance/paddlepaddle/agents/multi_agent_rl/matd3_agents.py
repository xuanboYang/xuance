import paddle
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from xuance.paddlepaddle import Layer
from xuance.paddlepaddle.utils import NormalizeFunctions, ActivationFunctions
from xuance.paddlepaddle.policies import REGISTRY_Policy
from xuance.paddlepaddle.agents.multi_agent_rl.iddpg_agents import IDDPG_Agents


class MATD3_Agents(IDDPG_Agents):
    """The implementation of MATD3 agents.

    Args:
        config: The Namespace variable that provides hyper-parameters and other settings.
        envs: The vectorized environments.
    """

    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(MATD3_Agents, self).__init__(config, envs)

    def _build_policy(self) -> Layer:
        """
        Build representation(s) and policy(ies) for agent(s)

        Returns:
            policy (torch.nn.Module): A dict of policies.
        """
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = paddle.nn.initializer.Orthogonal()
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations
        A_representation = self._build_representation(self.config.representation, self.observation_space, self.config)
        critic_in = [sum(self.observation_space[k].shape) + sum(self.action_space[k].shape) for k in self.agent_keys]
        space_critic_in = {k: (sum(critic_in),) for k in self.agent_keys}
        C_representation = self._build_representation(self.config.representation, space_critic_in, self.config)

        # build policies
        if self.config.policy == "MATD3_Policy":
            policy = REGISTRY_Policy["MATD3_Policy"](
                action_space=self.action_space, n_agents=self.n_agents,
                actor_representation=A_representation, critic_representation=C_representation,
                actor_hidden_size=self.config.actor_hidden_size,
                critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device,
                use_distributed_training=self.distributed_training,
                activation_action=ActivationFunctions[self.config.activation_action],
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"MATD3 currently does not support the policy named {self.config.policy}.")

        return policy
