from .layers import (
    flow, nn,
    ModuleType,
    mlp_block, cnn_block, pooling_block, gru_block, lstm_block
)
from .distributions import (
    Distribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    ActivatedDiagGaussianDistribution
)
from .operations import (init_distributed_mode, update_linear_decay, set_seed,
                         get_flat_grad, get_flat_params, assign_from_flat_grads,
                         assign_from_flat_params, split_distributions, merge_distributions)
from .value_norm import ValueNorm


class Softmax2d(nn.Module):
    def __init__(self):
        super(Softmax2d, self).__init__()
        self.softmax = nn.Softmax(dim=1)  # 沿通道维度 (dim=1) 应用 Softmax

    def forward(self, input_tensor):
        return self.softmax(input_tensor)
    
ActivationFunctions = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softmax2d": Softmax2d,
    "elu": nn.ELU,
}

NormalizeFunctions = {
    "LayerNorm": nn.LayerNorm,
    "GroupNorm": nn.GroupNorm,
    "BatchNorm": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "InstanceNorm2d": nn.InstanceNorm2d
}

def orthogonal_init(tensor, gain=1.0):
    """
    对张量进行正交初始化。
    
    参数：
        tensor (flow.Tensor): 需要初始化的张量。
        gain (float): 缩放因子，默认为 1.0。
    """
    import oneflow as flow
    import numpy as np
    if tensor.ndim < 2:
        raise ValueError("Only tensors with ndim >= 2 are supported.")

    rows = tensor.shape[0]
    cols = np.prod(tensor.shape[1:])  # 将剩余维度展平为一维

    # 创建一个随机矩阵
    flat_shape = (rows, cols)
    random_matrix = np.random.normal(0.0, 1.0, size=flat_shape)

    # 使用 QR 分解生成正交矩阵
    u, _, v = np.linalg.svd(random_matrix, full_matrices=False)
    orthogonal_matrix = u if u.shape == flat_shape else v

    # 调整形状并应用缩放因子
    orthogonal_matrix = orthogonal_matrix.reshape(tensor.shape)
    orthogonal_matrix = gain * orthogonal_matrix

    # 将结果赋值给张量
    with flow.no_grad():
        tensor.copy_(flow.tensor(orthogonal_matrix, dtype=tensor.dtype))

InitializeFunctions = {
    "orthogonal": orthogonal_init
}
