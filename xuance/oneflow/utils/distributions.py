import oneflow as flow
from oneflow import Tensor
# from oneflow.nn.functional import softplus
# from oneflow.distributions import Categorical
# from oneflow.distributions import Normal
from oneflow.nn.functional import softplus
from oneflow.distributions import Categorical
# from oneflow.distributions import Normal  # OneFlow 没有 Normal 分布类，需要自己实现


from oneflow.nn.functional import softplus
import oneflow.nn.functional as F
F.softplus
from oneflow.distributions import Categorical
# from oneflow.distributions import Normal  # OneFlow 没有 Normal 分布类，需要自己实现

from abc import ABC, abstractmethod

# kl_div = oneflow.distributions.kl_divergence
import oneflow.nn as nn


# 实现自己的 Normal 分布类
class Normal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.device = loc.device

    def sample(self):
        return self.loc + self.scale * flow.randn_like(self.scale)
    
    def rsample(self):
        return self.sample()  # OneFlow 可能没有区分 sample 和 rsample
    
    def log_prob(self, value):
        var = self.scale ** 2
        log_scale = flow.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - flow.log(flow.sqrt(2 * flow.tensor(3.14159265358979323846, device=self.device)))
    
    def entropy(self):
        return 0.5 + 0.5 * flow.log(2 * flow.tensor(3.14159265358979323846, device=self.device)) + flow.log(self.scale)


class Distribution(ABC):
    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def set_param(self, *args):
        raise NotImplementedError

    @abstractmethod
    def get_param(self):
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: flow.Tensor):
        raise NotImplementedError

    @abstractmethod
    def entropy(self):
        raise NotImplementedError

    @abstractmethod
    def stochastic_sample(self):
        raise NotImplementedError

    @abstractmethod
    def deterministic_sample(self):
        raise NotImplementedError


class CategoricalDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim
        self.probs = None

    def set_param(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs
            self.distribution = Categorical(probs=self.probs)
        else:
            self.probs = F.softmax(logits, dim=-1)
            self.distribution = Categorical(probs=self.probs)

    def get_param(self):
        return {"probs": self.probs}

    def log_prob(self, x):
        # 不使用 self.distribution.log_prob，因为 OneFlow 中的 Categorical 可能没有实现这个方法
        # 手动计算对数概率
        one_hot_x = F.one_hot(x.long(), self.action_dim)
        log_probs = flow.log(self.probs + 1e-8)  # 加一个小的值防止 log(0)
        return (one_hot_x * log_probs).sum(-1)

    def entropy(self):
        # 不使用 self.distribution.entropy，因为 OneFlow 中的 Categorical 可能没有实现这个方法
        # 手动计算熵
        log_probs = flow.log(self.probs + 1e-8)  # 加一个小的值防止 log(0)
        return -(self.probs * log_probs).sum(-1)

    def stochastic_sample(self):
        try:
            return self.distribution.sample()
        except NotImplementedError:
            # 手动实现采样
            sample_shape = self.probs.shape[:-1]
            # 创建一个与 probs 形状相同（除了最后一维）的均匀分布噪声
            uniform = flow.rand(sample_shape, device=self.probs.device)
            # 使用 Gumbel-Max 技巧进行采样
            gumbel = -flow.log(-flow.log(uniform + 1e-8) + 1e-8)
            # 返回具有最高 probs + gumbel 值的索引
            return (self.probs.log() + gumbel).argmax(dim=-1)

    def deterministic_sample(self):
        return flow.argmax(self.probs, dim=-1)


class DiagGaussianDistribution(Distribution):
    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.mean = None
        self.std = None

    def set_param(self, mean, std):
        self.mean = mean
        self.std = std
        self.distribution = Normal(self.mean, self.std)

    def get_param(self):
        return {"mean": self.mean, "std": self.std}

    def log_prob(self, x):
        return self.distribution.log_prob(x).sum(-1)

    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def stochastic_sample(self):
        return self.distribution.sample()

    def rsample(self):
        return self.distribution.rsample()

    def deterministic_sample(self):
        return self.mean

    def kl_divergence(self, other):
        # 自己实现KL散度计算
        return 0.5 * (
                (other.std / self.std).pow(2) + (self.mean - other.mean).pow(2) / self.std.pow(2) - 1 + 2 * (
                self.std.log() - other.std.log())
        ).sum(-1)


class ActivatedDiagGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, activation_fn=None, device=None):
        super(ActivatedDiagGaussianDistribution, self).__init__(action_dim)
        self.activation_fn = activation_fn
        self.device = device

    def activated_rsample(self):
        x = self.rsample()
        if self.activation_fn is not None:
            x = self.activation_fn()(x)
        return x

    def activated_rsample_and_logprob(self):
        x = self.rsample()
        log_prob = self.log_prob(x)
        if self.activation_fn is not None:
            x = self.activation_fn()(x)
        return x, log_prob
