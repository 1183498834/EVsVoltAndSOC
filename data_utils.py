"""数据加载与预处理工具。

供训练脚本（run_SOC.py / run_U_DLSTM.py）与预测脚本
（soc_predict.py / volt_predict.py）复用。
"""
import numpy as np
import torch


def make_windows(needed, seq_len, label_len, pred_len, step=1):
    """按滑动窗口切分序列。

    参数
    ----
    needed : np.ndarray
        形状为 (循环数, 时间步, 通道数) 的特征序列。
    seq_len : int
        输入序列长度。
    label_len : int
        标签中与输入重叠的前缀长度。
    pred_len : int
        需要预测的长度。
    step : int
        滑动步长，默认 1。

    返回
    ----
    data_x : np.ndarray
        输入窗口，形状 (窗口数, seq_len, 通道数)。
    data_y : np.ndarray
        标签窗口，形状 (窗口数, label_len + pred_len, 通道数)。
    """
    data_x, data_y = [], []
    for times in range(needed.shape[0]):
        start_x = 0
        while True:
            end_x = start_x + seq_len
            start_y = end_x - label_len
            end_y = start_y + label_len + pred_len
            data_x.append(needed[times, start_x:end_x, :])
            data_y.append(needed[times, start_y:end_y, :])
            start_x += step
            if start_x + seq_len + pred_len > needed.shape[1]:
                break
    return np.array(data_x), np.array(data_y)


def split_train_test(data_x, data_y, ratio=0.7):
    """按比例切分训练集与测试集（不 shuffle，保持时序顺序）。"""
    split = int(len(data_x) * ratio)
    return data_x[:split], data_x[split:], data_y[:split], data_y[split:]


def standardize_per_cycle(needed, scalers):
    """逐循环（逐充电次数）对每一列做标准化。

    参数
    ----
    needed : np.ndarray
        形状 (循环数, 时间步, 通道数)，原地修改。
    scalers : list[StandardScaler]
        每个通道一个 scaler，长度需等于通道数。
    """
    for times in range(needed.shape[0]):
        for channel, scaler in enumerate(scalers):
            needed[times, :, channel] = scaler.fit_transform(
                needed[times, :, channel].reshape(-1, 1)
            ).reshape(-1)


def load_state_dict(model, path, device="cpu"):
    """加载模型权重，并兼容 DataParallel 保存时带有的 ``module.`` 前缀。"""
    state_dict = torch.load(path, map_location=torch.device(device))
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model
