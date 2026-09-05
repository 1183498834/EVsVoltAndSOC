"""DLinear 模型：用于预测电动汽车充电过程中的 SOC（荷电状态）。

参考论文 *Are Transformers Effective for Time Series Forecasting?* 中的
Decomposition-Linear 结构：先做序列分解，再对残差（季节性）与趋势
分别用线性层做预测。
"""
import torch.nn as nn

from decomposition import SeriesDecomp


class Model(nn.Module):
    """Decomposition-Linear 网络。

    输入 : [Batch, seq_len, enc_in]
    输出 : [Batch, pred_len, enc_in]
    """

    def __init__(self, seq_len=30, pred_len=6, enc_in=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        kernel_size = 25
        self.decomp = SeriesDecomp(kernel_size)

        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

        # 如需可视化权重，可取消下面两行注释
        # self.linear_seasonal.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))
        # self.linear_trend.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]
