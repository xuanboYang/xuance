import numpy as np
import oneflow as flow
from xuance.common import Sequence, Optional, Union, Callable
from xuance.oneflow import Module, Tensor
from xuance.oneflow.utils import nn, mlp_block, ModuleType


# directly returns the original observation
class Basic_Identical(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[Union[str, int, flow.device]] = None,
                 **kwargs):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device

    def forward(self, observations: np.ndarray):
        if isinstance(observations, np.ndarray):
            observations = flow.from_numpy(observations).to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        state = observations.to(flow.float32)
        return {'state': state}


# process the input observations with stacks of MLP layers
class Basic_MLP(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 **kwargs):
        super(Basic_MLP, self).__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()
        self.model = self.model.to(self.device)

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            block, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         device=self.device)
            layers.extend(block)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        if isinstance(observations, np.ndarray):
            observations = flow.from_numpy(observations).to(self.device)
        elif observations.device != self.device:
            observations = observations.to(self.device)
        tensor_observation = observations.to(flow.float32)
        self.model = self.model.to(self.device)
        tensor_observation = tensor_observation.to(self.device)
        return {'state': self.model(tensor_observation)}
