import oneflow as flow
import oneflow.nn as nn
from xuance.common import Optional, Sequence, Tuple, Type, Union, Callable

ModuleType = Type[nn.Module]


def mlp_block(input_dim: int,
              output_dim: int,
              normalize: Optional[Union[nn.BatchNorm1d, nn.LayerNorm]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[flow.tensor], flow.tensor]] = None,
              device: Optional[Union[str, int]] = None) -> Tuple[Sequence[ModuleType], Tuple[int]]:
    block = []
    # 确保输入参数为Python原生int类型
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if device is not None:
        linear = nn.Linear(input_dim, output_dim, device=device)
    else:
        linear = nn.Linear(input_dim, output_dim)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    block.append(linear)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        # 检查normalize是否为LayerNorm类
        if normalize == nn.LayerNorm:
            # LayerNorm不接受device参数
            block.append(normalize(output_dim))
        elif normalize == nn.BatchNorm1d:
            # BatchNorm1d接受device参数
            if device is not None:
                block.append(normalize(output_dim, device=device))
            else:
                block.append(normalize(output_dim))
        else:
            # 其他情况，尝试传递device参数
            try:
                if device is not None:
                    block.append(normalize(output_dim, device=device))
                else:
                    block.append(normalize(output_dim))
            except TypeError:
                # 如果出现TypeError，说明不接受device参数
                block.append(normalize(output_dim))
    return block, (output_dim,)


def cnn_block(input_shape: Sequence[int],
              filter: int,
              kernel_size: int,
              stride: int,
              normalize: Optional[Union[nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d]] = None,
              activation: Optional[ModuleType] = None,
              initialize: Optional[Callable[[flow.tensor], flow.tensor]] = None,
              device: Optional[Union[str, int]] = None
              ) -> Tuple[Sequence[ModuleType], Tuple]:
    # 确保所有参数都是Python原生int类型
    input_shape = tuple(int(dim) for dim in input_shape)
    filter = int(filter)
    kernel_size = int(kernel_size)
    stride = int(stride)
    
    assert len(input_shape) == 3  # CxHxW
    C, H, W = input_shape
    padding = int((kernel_size - stride) // 2)
    block = []
    cnn = nn.Conv2d(C, filter, kernel_size, stride, padding=padding, device=device)
    if initialize is not None:
        initialize(cnn.weight)
        nn.init.constant_(cnn.bias, 0)
    block.append(cnn)
    C = filter
    H = int((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    W = int((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
    if activation is not None:
        block.append(activation())
    if normalize is not None:
        if normalize == nn.GroupNorm:
            block.append(normalize(C // 2, C, device=device))
        elif normalize == nn.LayerNorm:
            block.append(normalize((C, H, W), device=device))
        else:
            block.append(normalize(C, device=device))
    return block, (C, H, W)


def pooling_block(input_shape: Sequence[int],
                  scale: int,
                  pooling: Union[nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
                  device: Optional[Union[str, int]] = None) -> Sequence[ModuleType]:
    # 确保所有参数都是Python原生int类型
    input_shape = tuple(int(dim) for dim in input_shape)
    scale = int(scale)
    
    C, H, W = input_shape
    H, W = int(H / scale), int(W / scale)
    block = []
    pool = pooling((H, W))
    if device is not None:
        pool = pool.to(device)
    block.append(pool)
    return block, (C, H, W)


def gru_block(input_dim: int,
              output_dim: int,
              num_layers: int = 1,
              dropout: float = 0,
              initialize: Optional[Callable[[flow.tensor], flow.tensor]] = None,
              device: Optional[Union[str, int]] = None) -> Tuple[nn.Module, int]:
    # 确保所有参数都是Python原生int类型
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    num_layers = int(num_layers)

    gru = nn.GRU(
        input_size=input_dim,
        hidden_size=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True,
        device=device
    )
    for name, param in gru.named_parameters():
        if "weight" in name:
            if initialize is not None:
                initialize(param)
            else:
                nn.init.orthogonal_(param)
    return gru, output_dim


def lstm_block(input_dim: int,
               output_dim: int,
               num_layers: int = 1,
               dropout: float = 0,
               initialize: Optional[Callable[[flow.tensor], flow.tensor]] = None,
               device: Optional[Union[str, int]] = None) -> Tuple[nn.Module, int]:
    # 确保所有参数都是Python原生int类型
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    num_layers = int(num_layers)
    
    lstm = nn.LSTM(
        input_size=input_dim,
        hidden_size=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=True,
        device=device
    )
    for name, param in lstm.named_parameters():
        if "weight" in name:
            if initialize is not None:
                initialize(param)
            else:
                nn.init.orthogonal_(param)
    return lstm, output_dim
