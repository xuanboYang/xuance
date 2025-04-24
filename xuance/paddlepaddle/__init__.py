# from torch import Tensor
# 对应 torch.Tensor
from paddle import Tensor

# from torch.nn import Module, ModuleDict
# 对应 torch.nn.Module
from paddle.nn import Layer, LayerDict

# from torch.nn.parallel import DistributedDataParallel
# 对应 torch.nn.parallel.DistributedDataParallel
from paddle import DataParallel

from xuance.paddlepaddle.representations import REGISTRY_Representation
from xuance.paddlepaddle.policies import REGISTRY_Policy
from xuance.paddlepaddle.learners import REGISTRY_Learners
from xuance.paddlepaddle.agents import REGISTRY_Agents

__all__ = [
    "Tensor",
    "Layer",
    "LayerDict",
    "DataParallel",
    "REGISTRY_Representation", "REGISTRY_Policy", "REGISTRY_Learners", "REGISTRY_Agents"
]
