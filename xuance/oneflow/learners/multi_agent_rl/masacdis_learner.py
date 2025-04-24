"""
Multi-agent Soft Actor-critic (MASAC) with discrete action spaces.
Implementation: oneflow
"""
import oneflow as flow
from oneflow import nn
from xuance.common import List
from argparse import Namespace
from xuance.oneflow.learners.multi_agent_rl.isac_learner import ISAC_Learner
from operator import itemgetter


class MASACDIS_Learner(ISAC_Learner):
    def __init__(self,
                 config: Namespace,
                 model_keys: List[str],
                 agent_keys: List[str],
                 policy: nn.Module):
        super(MASACDIS_Learner, self).__init__(config, model_keys, agent_keys, policy)

    def update(self, sample):
        self.iterations += 1
        info = {}

        self.policy.soft_update(self.tau)
        return info

    def update_rnn(self, sample):
        self.iterations += 1
        info = {}

        return info
