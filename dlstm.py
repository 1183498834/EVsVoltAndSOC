"""DLSTM 模型：用于预测电动汽车充电过程中的单体电池电压。

在 DLinear 的分解结构基础上，将残差（季节性）与趋势分支中的线性层
替换为 LSTM 层。
"""
import torch
import torch.nn as nn

from decomposition import SeriesDecomp


class Model(nn.Module):
    """Decomposition-LSTM 网络。

    输入 : [Batch, seq_len, enc_in]
    输出 : [Batch, pred_len, enc_in]
    """

    def __init__(self, seq_len=30, pred_len=6, enc_in=4, hidden_size=10, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.channels = enc_in

        kernel_size = 25
        self.decomp = SeriesDecomp(kernel_size)

        self.lstm_seasonal = nn.LSTM(seq_len, hidden_size, num_layers, batch_first=True)
        self.linear_seasonal = nn.Linear(hidden_size, pred_len)

        self.lstm_trend = nn.LSTM(seq_len, hidden_size, num_layers, batch_first=True)
        self.linear_trend = nn.Linear(hidden_size, pred_len)

        # 如需可视化权重，可取消下面两行注释
        # self.linear_seasonal.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))
        # self.linear_trend.weight = nn.Parameter((1 / seq_len) * torch.ones([pred_len, seq_len]))

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        # 随机初始化隐藏状态；如需可复现，可将 randn 替换为 zeros
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)

        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        seasonal_lstm, _ = self.lstm_seasonal(seasonal_init, (h_0, c_0))
        seasonal_output = self.linear_seasonal(seasonal_lstm)
        trend_lstm, _ = self.lstm_trend(trend_init, (h_0, c_0))
        trend_output = self.linear_trend(trend_lstm)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]
