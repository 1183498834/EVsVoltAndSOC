"""时间序列分解模块。

提供移动平均与序列分解两个基础模块，供 DLinear 与 DLSTM 网络复用。

分解思想来自论文 *Are Transformers Effective for Time Series Forecasting?*：
先对时间序列做趋势/残差分解，再分别建模。
"""
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """移动平均模块，用于提取时间序列的趋势成分。"""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 在时间序列两端做 padding，使输出长度与输入保持一致
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """序列分解模块：将序列分解为残差（季节性）与趋势两部分。"""

    def __init__(self, kernel_size):
        super().__init__()
        # 属性名沿用 moving_avg，以兼容已有模型权重的 state_dict key
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
