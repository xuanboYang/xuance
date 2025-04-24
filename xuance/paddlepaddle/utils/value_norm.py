import numpy as np
import paddle
import paddle.nn as nn


def paddle_dtype_to_numpy_dtype(dtype):
    """将 PaddlePaddle 数据类型转换为 NumPy 数据类型"""
    if dtype == paddle.float32:
        return np.float32
    elif dtype == paddle.float64:
        return np.float64
    elif dtype == paddle.float16:
        return np.float16
    elif dtype == paddle.int32:
        return np.int32
    elif dtype == paddle.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class ValueNorm(nn.Layer):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        # 创建不可训练的参数
        self.running_mean = self.create_parameter(
            shape=(input_shape,),
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.running_mean.stop_gradient = True

        self.running_mean_sq = self.create_parameter(
            shape=(input_shape,),
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.running_mean_sq.stop_gradient = True

        self.debiasing_term = self.create_parameter(
            shape=[1],  # 单个标量参数
            default_initializer=paddle.nn.initializer.Constant(0.0)
        )
        self.debiasing_term.stop_gradient = True

        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        self.running_mean.set_value(paddle.zeros_like(self.running_mean))
        self.running_mean_sq.set_value(paddle.zeros_like(self.running_mean_sq))
        self.debiasing_term.set_value(paddle.zeros_like(self.debiasing_term))

    def running_mean_var(self):
        """计算去偏后的均值和方差"""
        debiased_mean = self.running_mean / self.debiasing_term.clip(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clip(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clip(min=1e-2)
        return debiased_mean, debiased_var

    @paddle.no_grad()
    def update(self, input_vector):
        """更新 running_mean、running_mean_sq 和 debiasing_term"""
        if isinstance(input_vector, np.ndarray):
            input_vector = paddle.to_tensor(input_vector)

        input_vector = input_vector.astype(self.running_mean.dtype)  # 确保数据类型一致

        # 计算 batch 均值和平方均值
        batch_mean = input_vector.mean(axis=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(axis=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.shape[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        # 更新 running_mean 和 running_mean_sq
        new_running_mean = self.running_mean * weight + batch_mean * (1.0 - weight)
        self.running_mean.set_value(new_running_mean)

        new_running_mean_sq = self.running_mean_sq * weight + batch_sq_mean * (1.0 - weight)
        self.running_mean_sq.set_value(new_running_mean_sq)

        # 更新 debiasing_term
        new_debiasing_term = self.debiasing_term * weight + 1.0 * (1.0 - weight)
        self.debiasing_term.set_value(new_debiasing_term)

    def normalize(self, input_vector):
        """归一化输入向量"""
        if isinstance(input_vector, np.ndarray):
            input_vector = paddle.to_tensor(input_vector)

        input_vector = input_vector.astype(self.running_mean.dtype)  # 确保数据类型一致

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / paddle.sqrt(var)[(None,) * self.norm_axes]

        return out

    def denormalize(self, input_vector):
        """将归一化的数据还原为原始分布"""
        if isinstance(input_vector, np.ndarray):
            input_vector = paddle.to_tensor(input_vector)

        # 确保数据类型一致
        np_dtype = paddle_dtype_to_numpy_dtype(self.running_mean.dtype)
        input_vector = input_vector.astype(np_dtype)

        mean, var = self.running_mean_var()
        out = input_vector * paddle.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]

        return out.numpy()  # 转换为 NumPy 数组返回
