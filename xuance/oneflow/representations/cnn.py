import numpy as np
import oneflow as flow
from xuance.common import Sequence, Optional, Union, Callable
from xuance.oneflow import Module, Tensor
from xuance.oneflow.utils import nn, cnn_block, mlp_block, ModuleType


# process the input observations with stacks of CNN layers
class Basic_CNN(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 **kwargs):
        super(Basic_CNN, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (filters[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for k, s, f in zip(self.kernels, self.strides, self.filters):
            block, input_shape = cnn_block(input_shape=input_shape,
                                          filter=f,
                                          kernel_size=k,
                                          stride=s,
                                          normalize=self.normalize,
                                          activation=self.activation,
                                          initialize=self.initialize,
                                          device=self.device)
            layers += block
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        if isinstance(observations, np.ndarray):
            observations = flow.from_numpy(observations).to(self.device)
        observations = observations.to(flow.float32)
        features = self.model(observations)
        return features.reshape(features.shape[0], -1)


class AC_CNN_Atari(Module):
    def __init__(self,
                 input_shape: Sequence[int],
                 kernels: Sequence[int],
                 strides: Sequence[int],
                 filters: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, flow.device]] = None,
                 fc_hidden_sizes: Sequence[int] = (),
                 **kwargs):
        super(AC_CNN_Atari, self).__init__()
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Channels x Height x Width
        self.kernels = kernels
        self.strides = strides
        self.filters = filters
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.fc_hidden_sizes = fc_hidden_sizes
        self.device = device
        self.H, self.W, self.C = input_shape
        self.output_shapes = {'state': (self.fc_hidden_sizes[-1],)}
        self.model = self._create_network()

    def _init_layer(self, layer, gain=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias)

    def _create_network(self):
        H, W, C = self.input_shape[1], self.input_shape[2], self.input_shape[0]  # x.shape: (N, C, H, W)
        layers = []
        channels = [C] + list(self.filters)
        in_channels = C
        for c, k, s in zip(channels[1:], self.kernels, self.strides):
            cnn_layer = nn.Conv2d(in_channels, c, kernel_size=k, stride=s)
            layers += [cnn_layer, self.activation()]
            in_channels = c
            H = int((H - k) / s + 1)
            W = int((W - k) / s + 1)

        cnn_output_size = H * W * self.filters[-1]
        layers += [nn.Flatten()]
        fc_layers, _ = mlp_block(input_dim=cnn_output_size,
                                  output_dim=self.fc_hidden_sizes[-1],
                                  hidden_sizes=self.fc_hidden_sizes[:-1],
                                  activation=self.activation,
                                  device=self.device)
        layers += fc_layers
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        if isinstance(observations, np.ndarray):
            observations = flow.from_numpy(observations).to(self.device)
        observations = observations.to(flow.float32)
        features = self.model(observations)
        return features
