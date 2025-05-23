import paddle
from argparse import Namespace
from xuance.environment import DummyVecMultiAgentEnv
from xuance.paddlepaddle import Layer
from xuance.paddlepaddle.utils import NormalizeFunctions, ActivationFunctions
from xuance.paddlepaddle.policies import REGISTRY_Policy, QMIX_mixer, QMIX_FF_mixer
from xuance.paddlepaddle.agents.multi_agent_rl.qmix_agents import QMIX_Agents


class WQMIX_Agents(QMIX_Agents):
    """The implementation of Weighted QMIX agents.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecMultiAgentEnv):
        super(WQMIX_Agents, self).__init__(config, envs)

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
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policies
        dim_state = self.state_space.shape[-1]
        mixer = QMIX_mixer(dim_state, self.config.hidden_dim_mixing_net,
                           self.config.hidden_dim_hyper_net, self.n_agents, device)
        ff_mixer = QMIX_FF_mixer(dim_state, self.config.hidden_dim_ff_mix_net, self.n_agents, device)
        if self.config.policy == "Weighted_Mixing_Q_network":
            policy = REGISTRY_Policy["Weighted_Mixing_Q_network"](
                action_space=self.action_space, n_agents=self.n_agents, representation=representation,
                mixer=mixer, ff_mixer=ff_mixer, hidden_size=self.config.q_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation,
                device=device, use_distributed_training=self.distributed_training,
                use_parameter_sharing=self.use_parameter_sharing, model_keys=self.model_keys,
                use_rnn=self.use_rnn, rnn=self.config.rnn if self.use_rnn else None)
        else:
            raise AttributeError(f"WQMIX currently does not support the policy named {self.config.policy}.")

        return policy
