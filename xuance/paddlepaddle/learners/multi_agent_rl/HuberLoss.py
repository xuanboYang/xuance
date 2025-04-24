import paddle


class HuberLoss(paddle.nn.Layer):
    def __init__(self, delta=1.0, reduction='none'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        # 计算误差
        diff = paddle.abs(input - target)

        # 根据 delta 值计算 Huber Loss
        loss = paddle.where(
            diff <= self.delta,
            0.5 * paddle.pow(diff, 2),  # 当误差小于等于 delta 时，使用 MSE
            self.delta * diff - 0.5 * self.delta ** 2  # 当误差大于 delta 时，使用 MAE
        )

        # 根据 reduction 参数进行规约
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
