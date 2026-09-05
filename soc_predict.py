"""使用训练好的 DLinear 模型对 SOC 进行预测。

对 ``./np_data/*/charge_new_feature.npy`` 逐车预测，输出对比图与
MSE / MAE / RMSE / R2 评估指标。
"""
import os
from glob import iglob
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from data_utils import load_state_dict, make_windows
from dlinear import Model

SEQ_LEN = 60
LABEL_LEN = 42
PRED_LEN = 18

MODEL_PATH = "./SOC_unstand/model/soc_model.pt"
DATA_ROOT = "./np_data/*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = Model(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=3)
    model = load_state_dict(model, MODEL_PATH)
    model.to(device)
    model.eval()

    for car_dir in iglob(DATA_ROOT):
        data_path = os.path.join(car_dir, "charge_new_feature.npy")
        print(f"Running {data_path}!")
        data = np.load(data_path)

        battery_soc = data[:, :, 2]
        charge_features = data[:, :, :2]

        # NOTE: 训练脚本 run_SOC.py 对总电流通道做过 (x/0.1 + 2000) * 0.1 的
        # 缩放，此处未做该变换。如需与训练保持一致，请补充同样的处理。
        needed = np.dstack((charge_features, battery_soc))

        data_x, data_y = make_windows(needed, SEQ_LEN, LABEL_LEN, PRED_LEN, step=1)
        data_x_tensor = torch.FloatTensor(data_x).to(device)

        mse_list, mae_list, rmse_list, r2_list = [], [], [], []
        soc_series = []

        for batch in tqdm(range(data_x_tensor.shape[0])):
            input_x = data_x_tensor[batch:batch + 1]  # [1, seq_len, channels]
            true_y = data_y[batch, -PRED_LEN:, :]

            output = model(input_x)
            output = output[:, -PRED_LEN:, -1:]

            input_soc = input_x[0, :, -1].detach().cpu().numpy()
            pred_soc = output[0, :, -1].detach().cpu().numpy()

            gtsoc = np.concatenate((input_soc, true_y[:, -1]), axis=0)
            pdsoc = np.concatenate((input_soc, pred_soc), axis=0)
            soc_series.extend([gtsoc, pdsoc])

            # 保存对比图
            _save_picture(car_dir, batch, gtsoc, pdsoc)

            # 保存误差
            mse_list.append(mean_squared_error(true_y[:, -1], pred_soc))
            mae_list.append(mean_absolute_error(true_y[:, -1], pred_soc))
            rmse_list.append(sqrt(mean_squared_error(true_y[:, -1], pred_soc)))
            r2_list.append(r2_score(true_y[:, -1], pred_soc))

        _save_metrics(car_dir, soc_series, mse_list, mae_list, rmse_list, r2_list)


def _save_picture(car_dir, batch, gtsoc, pdsoc):
    picture_dir = os.path.join(car_dir, "pictures_soc")
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{batch}.png")

    plt.figure()
    plt.plot(gtsoc, label="GroundTruth SOC", linewidth=2)
    plt.plot(pdsoc, label="Prediction SOC", linewidth=2)
    plt.legend()
    plt.savefig(picture_path, bbox_inches="tight")
    plt.close()


def _save_metrics(car_dir, soc_series, mse_list, mae_list, rmse_list, r2_list):
    data_dir = os.path.join(car_dir, "soc_data")
    os.makedirs(data_dir, exist_ok=True)
    soc_data = pd.DataFrame(data=list(map(list, zip(*soc_series))))
    soc_data.to_csv(os.path.join(data_dir, "soc.csv"), encoding="gbk")

    for name, values in (
        ("soc_mse", mse_list),
        ("soc_mae", mae_list),
        ("soc_rmse", rmse_list),
        ("soc_r2", r2_list),
    ):
        metric_dir = os.path.join(car_dir, name)
        os.makedirs(metric_dir, exist_ok=True)
        np.save(os.path.join(metric_dir, "soc.npy"), values)


if __name__ == "__main__":
    main()
