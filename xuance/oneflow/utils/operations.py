import os
import random
import oneflow as flow
import numpy as np
import oneflow.nn as nn
from .distributions import CategoricalDistribution, DiagGaussianDistribution


def init_distributed_mode(master_port: str = None):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        master_port: The communication port of master device
    """
    rank = os.environ["LOCAL_RANK"]
    os.environ["MASTER_ADDR"] = "localhost"  # The IP address of the machine that is running the rank 0 process.
    os.environ["MASTER_PORT"] = "12355" if master_port is None else master_port
    flow.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if int(rank) == 0:
        print("The distributed process group is initialized.")


def update_linear_decay(optimizer, step, total_steps, initial_lr, end_factor):
    lr = initial_lr * (1 - step / float(total_steps))
    if lr < end_factor * initial_lr:
        lr = end_factor * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_seed(seed):
    flow.manual_seed(seed)
    # 检查CUDA是否可用，避免在CPU版本上调用CUDA函数
    if hasattr(flow, 'cuda') and hasattr(flow._oneflow_internal, 'GetCudaDeviceIndex'):
        flow.cuda.manual_seed(seed)
        flow.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_flat_grad(y: flow.Tensor, model: nn.Module) -> flow.Tensor:
    grads = flow.autograd.grad(y, model.parameters())
    return flow.cat([grad.reshape(-1) for grad in grads])


def get_flat_params(model: nn.Module) -> flow.Tensor:
    params = model.parameters()
    return flow.cat([param.reshape(-1) for param in params])


def assign_from_flat_grads(flat_grads: flow.Tensor, model: nn.Module) -> nn.Module:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = flat_grads[prev_ind:prev_ind + flat_size].view(param.size())
        prev_ind += flat_size
    return model


def assign_from_flat_params(flat_params: flow.Tensor, model: nn.Module) -> nn.Module:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data = flat_params[prev_ind:prev_ind + flat_size].view(param.size())
        prev_ind += flat_size
    return model


def split_distributions(distribution):
    """
    split a categorical distribution and a Gaussian distribution
    """
    if isinstance(distribution, CategoricalDistribution):
        reshaped = distribution.probs.reshape((-1, distribution.probs.shape[-1]))
        split_num = reshaped.shape[0]
        distributions = []
        for i in range(split_num):
            distributions.append(CategoricalDistribution(1, distribution.probs.shape[-1]))
            distributions[i].probs = reshaped[i].reshape((1, -1))
    elif isinstance(distribution, DiagGaussianDistribution):
        reshaped_mean = distribution.mean.reshape((-1, distribution.mean.shape[-1]))
        reshaped_std = distribution.std.reshape((-1, distribution.std.shape[-1]))
        split_num = reshaped_mean.shape[0]
        distributions = []
        for i in range(split_num):
            distributions.append(DiagGaussianDistribution(1, distribution.std.shape[-1]))
            distributions[i].mean = reshaped_mean[i].reshape((1, -1))
            distributions[i].std = reshaped_std[i].reshape((1, -1))
    else:
        raise TypeError("The input distribution is not a valid distribution!")
    return distributions


def merge_distributions(distribution_list):
    """
    merge a distribution list.
    The list contains categorical distributions or Gaussian distributions.
    """
    if isinstance(distribution_list[0], CategoricalDistribution):
        action_dim = distribution_list[0].action_dim
        probs = []
        for dist in distribution_list:
            probs.append(dist.probs)
        merged_probs = flow.cat(probs, dim=0)
        merged_dist = CategoricalDistribution(len(distribution_list), action_dim)
        merged_dist.probs = merged_probs
        return merged_dist
    elif isinstance(distribution_list[0], DiagGaussianDistribution):
        action_dim = distribution_list[0].action_dim
        means, stds = [], []
        for dist in distribution_list:
            means.append(dist.mean)
            stds.append(dist.std)
        merged_mean = flow.cat(means, dim=0)
        merged_std = flow.cat(stds, dim=0)
        merged_dist = DiagGaussianDistribution(len(distribution_list), action_dim)
        merged_dist.mean, merged_dist.std = merged_mean, merged_std
        return merged_dist
    else:
        raise TypeError("The input distribution is not a valid distribution!")
