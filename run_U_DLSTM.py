"""使用 DLSTM 网络训练单体电池电压预测模型。

对每个单体电池分别训练一个模型：取前 3 个通道（总电压、总电流、
SOC）加上该单体电池的电压通道，逐充电循环标准化后，以滑动窗口
方式构造输入与标签。
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import make_windows, split_train_test, standardize_per_cycle
from dlstm import Model

# 超参数：seq_len = label_len + pred_len
SEQ_LEN = 60
LABEL_LEN = 42
PRED_LEN = 18
BATCH_SIZE = 128
EPOCHS = 100
LR = 1

DATA_PATH = "./charge_new_feature.npy"
MODEL_DIR = "./cell_scale/model"
PICTURE_DIR = "./cell_scale/pictures"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    original = np.load(DATA_PATH)

    # 单体电池电压取原始通道 3 起，并放大 1000 倍
    battery_volt = original[:, :, 3:] * 1000
    battery_num = battery_volt.shape[2]
    charge_features = original[:, :, :3]

    for n_battery in range(battery_num):
        needed = np.dstack((charge_features, battery_volt[:, :, n_battery]))

        # 每个特征一个 scaler，逐充电循环标准化
        scalers = [StandardScaler() for _ in range(needed.shape[2])]
        standardize_per_cycle(needed, scalers)
        cell_scaler = scalers[-1]

        # 滑动窗口切分，再按 7:3 划分训练/测试集
        data_x, data_y = make_windows(needed, SEQ_LEN, LABEL_LEN, PRED_LEN)
        train_x, test_x, train_y, test_y = split_train_test(data_x, data_y)

        trainset = np.stack([train_x, train_y], axis=-1)
        testset = np.stack([test_x, test_y], axis=-1)
        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        print(f"train battery {n_battery + 1}/{battery_num}!")

        model = Model(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=needed.shape[2])
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3,
            threshold=1e-4, threshold_mode="rel", min_lr=0, eps=1e-8,
        )

        best_train_loss = float("inf")

        for epoch in range(EPOCHS):
            model.train()
            iter_count = 0
            total_loss = 0.0
            pbar = tqdm(train_loader)
            for i, data in enumerate(pbar):
                batch_x = data[:, :, :, 0].float().to(device)
                batch_y = data[:, :, :, 1].float().to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                # 只取预测段、最后一个通道（单体电压）
                outputs = outputs[:, -PRED_LEN:, -1:]
                batch_y = batch_y[:, -PRED_LEN:, -1:]

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                iter_count += 1
                total_loss += loss.item()
                if i % 2 == 0 and i > 0:
                    pbar.set_description(f"{n_battery + 1}/{battery_num} : {epoch + 1}/{EPOCHS}")
                    pbar.set_postfix(loss=total_loss / iter_count)

            model.eval()
            test_loss = 0.0
            test_count = 0
            with torch.no_grad():
                for idx, test_data in enumerate(test_loader):
                    batch_test_x = test_data[:, :, :, 0].float().to(device)
                    batch_test_y = test_data[:, :, :, 1].float().to(device)

                    outputs = model(batch_test_x)
                    outputs = outputs[:, -PRED_LEN:, -1:]
                    batch_test_y = batch_test_y[:, -PRED_LEN:, -1:]

                    loss = criterion(outputs, batch_test_y)
                    test_loss += loss.item()
                    test_count += 1

                    # 每隔 2 个 batch 画一张图
                    if idx % 2 == 0:
                        _plot_volt(batch_test_x, batch_test_y, outputs, cell_scaler, n_battery, idx)

            avg_train_loss = total_loss / max(iter_count, 1)
            avg_test_loss = test_loss / max(test_count, 1)
            print(
                f"[Battery {n_battery + 1}/{battery_num} Epoch {epoch + 1}/{EPOCHS}] "
                f"train_loss={avg_train_loss:.6f} test_loss={avg_test_loss:.6f}"
            )

            model_path = os.path.join(MODEL_DIR, f"cell{n_battery + 1}_model.pt")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), model_path)

            scheduler.step(avg_train_loss)


def _plot_volt(batch_x, batch_y, outputs, cell_scaler, n_battery, idx):
    input_np = batch_x.detach().cpu().numpy()
    true_np = batch_y.detach().cpu().numpy()
    pred_np = outputs.detach().cpu().numpy()

    input_u = cell_scaler.inverse_transform(input_np[0, :, -1].reshape(-1, 1)).reshape(-1)
    true_u = cell_scaler.inverse_transform(true_np[0, :, -1].reshape(-1, 1)).reshape(-1)
    pred_u = cell_scaler.inverse_transform(pred_np[0, :, -1].reshape(-1, 1)).reshape(-1)

    gt = np.concatenate((input_u / 1000, true_u / 1000), axis=0)
    pd = np.concatenate((input_u / 1000, pred_u / 1000), axis=0)

    picture_dir = os.path.join(PICTURE_DIR, f"battery{n_battery + 1}")
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{idx}.png")

    plt.figure()
    plt.plot(gt, label="GroundTruth", linewidth=2)
    plt.plot(pd, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(picture_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
