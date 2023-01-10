# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:23:32 2022

@author: Flashy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len=30, pred_len=6, enc_in=4, hidden_size=10, num_layers=2):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = enc_in
        
        self.lstm_seasonal = nn.LSTM(self.seq_len, self.hidden_size, self.num_layers, batch_first=True)
        self.Linear_Seasonal = nn.Linear(self.hidden_size,self.pred_len)
        
        self.lstm_trend = nn.LSTM(self.seq_len, self.hidden_size, self.num_layers, batch_first=True)
        self.Linear_Trend = nn.Linear(self.hidden_size,self.pred_len)
            
        # Use this two lines if you want to visualize the weights
        # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_lstm, _ = self.lstm_seasonal(seasonal_init, (h_0, c_0))
        seasonal_output = self.Linear_Seasonal(seasonal_lstm)
        trend_lstm, _ = self.lstm_trend(trend_init, (h_0, c_0))
        trend_output = self.Linear_Trend(trend_lstm)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]

