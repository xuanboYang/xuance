import numpy as np
from xuance.common import Sequence, Optional, Union, Callable
from xuance.paddlepaddle import Layer, Tensor
from xuance.paddlepaddle.utils import paddle, nn, mlp_block, ModuleType


# directly returns the original observation
class Basic_Identical(Layer):
    def __init__(self,
                 input_shape: Sequence[int],
                 device: Optional[Union[str, int]] = None,
                 **kwargs):
        super(Basic_Identical, self).__init__()
        assert len(input_shape) == 1
        self.output_shapes = {'state': (input_shape[0],)}
        self.device = device

    def forward(self, observations: np.ndarray):
        state = paddle.to_tensor(observations, dtype=paddle.float32)
        return {'state': state}


# process the input observations with stacks of MLP layers
class Basic_MLP(Layer):
    def __init__(self,
                 input_shape: Sequence[int],
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int]] = None,
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

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.normalize, self.activation, self.initialize,
                                         device=self.device)
            layers.extend(mlp)
        return nn.Sequential(*layers)

    def forward(self, observations: np.ndarray):
        # tensor_observation = paddle.to_tensor(observations, dtype=paddle.float32)
        # 检查 observations 的类型
        if isinstance(observations, paddle.Tensor):
            # 如果是 Paddle 张量，使用 clone().detach()
            tensor_observation = observations.clone().detach().astype(paddle.float32)
        else:
            # 如果是 NumPy 数组或其他类型，使用 paddle.to_tensor 转换
            tensor_observation = paddle.to_tensor(observations, dtype=paddle.float32)

        return {'state': self.model(tensor_observation)}
