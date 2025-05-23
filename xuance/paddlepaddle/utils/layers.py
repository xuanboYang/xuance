# import paddle
import paddle

# import paddle.nn as nn
import paddle.nn as nn

from xuance.common import Optional, Sequence, Tuple, Type, Union, Callable

ModuleType = Type[nn.Layer]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1D, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[paddle.Tensor], paddle.Tensor]] = None,
              device: Optional[Union[str, int]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    linear = nn.Linear(input_dim, output_dim)  # device=device
    if initialize is not None:
        initialize(linear.weight)
        # nn.init.constant_(linear.bias, 0)
        # 使用 Constant 初始化器将偏置初始化为 0
        # linear.bias.set_value(nn.initializer.Constant(value=0)(linear.bias))
        nn.initializer.Constant(value=0)(linear.bias)  # 直接调用初始化器
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        block.append(normalize(output_dim))
    return block, (output_dim,)


def cnn_block(input_shape: Sequence[int],
              filter: int,
              kernel_size: int,
              stride: int,
              normalize: Optional[Union[nn.BatchNorm2D, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2D]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[paddle.Tensor], paddle.Tensor]] = None,
              device: Optional[Union[str, int]] = None
              ) -> Tuple[Sequence[ModuleType], Tuple]:
    assert len(input_shape) == 3  # CxHxW
    C, H, W = input_shape
    padding = int((kernel_size - stride) // 2)
    block = []
    cnn = nn.Conv2D(C, filter, kernel_size, stride, padding=padding)  # device=device
    if initialize is not None:
        initialize(cnn.weight)
        # nn.init.constant_(cnn.bias, 0)
        cnn.bias.set_value(nn.initializer.Constant(value=0)(cnn.bias.shape))
    block.append(cnn)
    C = filter
    H = int((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    W = int((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        if normalize == nn.GroupNorm:
            block.append(normalize(C // 2, C))
        elif normalize == nn.LayerNorm:
            block.append(normalize((C, H, W)))
        else:
            block.append(normalize(C))
    return block, (C, H, W)


def pooling_block(input_shape: Sequence[int],
                  scale: int,
                  pooling: Union[nn.AdaptiveMaxPool2D, nn.AdaptiveAvgPool2D],
                  device: Optional[Union[str, int]] = None) -> Sequence[ModuleType]:
    assert len(input_shape) == 3  # CxHxW
    block = []
    C, H, W = input_shape
    block.append(pooling(output_size=(H // scale, W // scale), device=device))
    return block


def gru_block(input_dim: int,
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[paddle.Tensor], paddle.Tensor]] = None,
              device: Optional[Union[str, int]] = None) -> Tuple[nn.Layer, int]:
    gru = nn.GRU(input_size=input_dim,
                 hidden_size=output_dim,
                 num_layers=num_layers,
                 # batch_first=True,
                 dropout=dropout)  # ,device=device
    if initialize is not None:
        for weight_list in gru.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
                else:
                    # nn.init.constant_(weight, 0)
                    weight.set_value(nn.initializer.Constant(value=0)(weight.shape))

    return gru, output_dim


def lstm_block(input_dim: int,
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[paddle.Tensor], paddle.Tensor]] = None,
               device: Optional[Union[str, int]] = None) -> Tuple[nn.Layer, int]:
    lstm = nn.LSTM(input_size=input_dim,
                   hidden_size=output_dim,
                   num_layers=num_layers,
                   # batch_first=True,
                   dropout=dropout)
    if initialize is not None:
        for weight_list in lstm.all_weights:
            for weight in weight_list:
                if len(weight.shape) > 1:
                    initialize(weight)
                else:
                    # nn.init.constant_(weight, 0)
                    weight.set_value(nn.initializer.Constant(value=0)(weight.shape))

    return lstm, output_dim
