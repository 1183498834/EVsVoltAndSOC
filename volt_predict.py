"""使用训练好的 DLSTM 模型对单体电池电压进行预测。

遍历 ``./result/model/*.pt`` 中每个电池的模型，对 ``./np_data/*/``
逐车预测，输出对比图与 MSE / MAE / RMSE / R2 评估指标。
"""
import os
import re
from glob import iglob
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_utils import load_state_dict, make_windows, standardize_per_cycle
from dlstm import Model

SEQ_LEN = 60
LABEL_LEN = 42
PRED_LEN = 18

MODEL_GLOB = "./result/model/*.pt"
DATA_ROOT = "./np_data/*"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    for model_path in iglob(MODEL_GLOB):
        cell_num = int(re.split(r"cell|_", os.path.basename(model_path))[1]) - 1
        print(f"Running the {cell_num} cell.")

        model = Model(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=4)
        model = load_state_dict(model, model_path)
        model.to(device)
        model.eval()

        for car_dir in iglob(DATA_ROOT):
            data_path = os.path.join(car_dir, "charge_new_feature.npy")
            print(f"Running {data_path}!")
            data = np.load(data_path)

            battery_volt = data[:, :, 3:]
            charge_features = data[:, :, :3]
            needed = np.dstack((charge_features, battery_volt[:, :, cell_num]))

            scalers = [StandardScaler() for _ in range(needed.shape[2])]
            standardize_per_cycle(needed, scalers)
            # NOTE: scalers 在逐循环拟合后最终保留的是最后一个循环的统计量，
            # 因此这里 inverse_transform 用的是同一组统计量还原所有 batch。
            cell_scaler = scalers[-1]

            data_x, data_y = make_windows(needed, SEQ_LEN, LABEL_LEN, PRED_LEN, step=1)
            data_x_tensor = torch.FloatTensor(data_x).to(device)

            mse_list, mae_list, rmse_list, r2_list = [], [], [], []
            volt_series = []

            for batch in tqdm(range(data_x_tensor.shape[0])):
                input_x = data_x_tensor[batch:batch + 1]  # [1, seq_len, channels]
                true_y = data_y[batch, -PRED_LEN:, :]

                output = model(input_x)
                output = output[:, -PRED_LEN:, -2:]

                input_volt = input_x[0, :, -1].detach().cpu().numpy()
                pred_volt = output[0, :, -1].detach().cpu().numpy()

                gd_volt = cell_scaler.inverse_transform(input_volt.reshape(-1, 1)).reshape(-1)
                inverse_volt = cell_scaler.inverse_transform(pred_volt.reshape(-1, 1)).reshape(-1)
                true_volt = cell_scaler.inverse_transform(true_y[:, -1].reshape(-1, 1)).reshape(-1)

                gtvolt = np.concatenate((gd_volt, true_volt), axis=0)
                pdvolt = np.concatenate((gd_volt, inverse_volt), axis=0)
                volt_series.extend([gtvolt, pdvolt])

                _save_picture(car_dir, cell_num, batch, gtvolt, pdvolt)

                mse_list.append(mean_squared_error(true_volt, inverse_volt))
                mae_list.append(mean_absolute_error(true_volt, inverse_volt))
                rmse_list.append(sqrt(mean_squared_error(true_volt, inverse_volt)))
                r2_list.append(r2_score(true_volt, inverse_volt))

            _save_metrics(car_dir, cell_num, volt_series, mse_list, mae_list, rmse_list, r2_list)


def _save_picture(car_dir, cell_num, batch, gtvolt, pdvolt):
    picture_dir = os.path.join(car_dir, "pictures_volt", str(cell_num))
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{batch}.png")

    plt.figure()
    plt.plot(gtvolt, label="GroundTruth Volt", linewidth=2)
    plt.plot(pdvolt, label="Prediction Volt", linewidth=2)
    plt.legend()
    plt.savefig(picture_path, bbox_inches="tight")
    plt.close()


def _save_metrics(car_dir, cell_num, volt_series, mse_list, mae_list, rmse_list, r2_list):
    data_dir = os.path.join(car_dir, "volt_data")
    os.makedirs(data_dir, exist_ok=True)
    volt_data = pd.DataFrame(data=list(map(list, zip(*volt_series))))
    volt_data.to_csv(os.path.join(data_dir, f"cell_{cell_num}.csv"), encoding="gbk")

    for name, values in (
        ("volt_mse", mse_list),
        ("volt_mae", mae_list),
        ("volt_rmse", rmse_list),
        ("volt_r2", r2_list),
    ):
        metric_dir = os.path.join(car_dir, name)
        os.makedirs(metric_dir, exist_ok=True)
        np.save(os.path.join(metric_dir, f"cell_{cell_num}.npy"), values)


if __name__ == "__main__":
    main()
