"""使用 DLinear 网络训练 SOC（荷电状态）预测模型。

数据文件默认读取同目录下的 ``charge_new_feature.npy``，
形状为 (充电次数, 时间步, 通道数)。取前 3 个通道
（总电压、总电流、SOC），以滑动窗口方式构造输入与标签。
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import make_windows, split_train_test
from dlinear import Model

# 超参数：seq_len = label_len + pred_len
SEQ_LEN = 60
LABEL_LEN = 42
PRED_LEN = 18
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001

DATA_PATH = "./charge_new_feature.npy"
MODEL_PATH = "./SOC_unstand/model/soc_model.pt"
PICTURE_DIR = "./SOC_unstand/pictures"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # [充电次数, 时间步, 通道数]
    original = np.load(DATA_PATH)

    # 取前 3 个通道（总电压、总电流、SOC）；第 1 个通道做量纲缩放
    charge_features = original[:, :, :3].copy()
    charge_features[:, :, 1] = (charge_features[:, :, 1] / 0.1 + 2000) * 0.1

    # 滑动窗口切分，再按 7:3 划分训练/测试集
    data_x, data_y = make_windows(charge_features, SEQ_LEN, LABEL_LEN, PRED_LEN)
    train_x, test_x, train_y, test_y = split_train_test(data_x, data_y)

    trainset = np.stack([train_x, train_y], axis=-1)
    testset = np.stack([test_x, test_y], axis=-1)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Model(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=charge_features.shape[2])
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5,
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
            # 只取预测段、最后一个通道（SOC）
            outputs = outputs[:, -PRED_LEN:, -1]
            batch_y = batch_y[:, -PRED_LEN:, -1]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            iter_count += 1
            total_loss += loss.item()
            if i % 2 == 0 and i > 0:
                pbar.set_description(f"{epoch + 1}/{EPOCHS}")
                pbar.set_postfix(loss=total_loss / iter_count)

        model.eval()
        test_loss = 0.0
        test_count = 0
        with torch.no_grad():
            for idx, test_data in enumerate(test_loader):
                batch_test_x = test_data[:, :, :, 0].float().to(device)
                batch_test_y = test_data[:, :, :, 1].float().to(device)

                outputs = model(batch_test_x)
                outputs = outputs[:, -PRED_LEN:, -2:]
                batch_test_y = batch_test_y[:, -PRED_LEN:, -2:]

                loss = criterion(outputs, batch_test_y)
                test_loss += loss.item()
                test_count += 1

                # 每隔 20 个 batch 画一张图
                if idx % 20 == 0:
                    pred = outputs.detach().cpu().numpy()
                    true = batch_test_y.detach().cpu().numpy()
                    _plot_soc(batch_test_x, true, pred, epoch, idx)

        avg_train_loss = total_loss / max(iter_count, 1)
        avg_test_loss = test_loss / max(test_count, 1)
        print(
            f"[Epoch {epoch + 1}/{EPOCHS}] "
            f"train_loss={avg_train_loss:.6f} test_loss={avg_test_loss:.6f}"
        )

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), MODEL_PATH)

        # 与原版保持一致：调度器监控训练损失
        scheduler.step(avg_train_loss)


def _plot_soc(batch_x, true, pred, epoch, idx):
    input_np = batch_x.detach().cpu().numpy()
    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)

    picture_dir = os.path.join(PICTURE_DIR, f"soc_epoch{epoch}")
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{idx}.png")

    plt.figure()
    plt.plot(gt, label="GroundTruth SOC", linewidth=2)
    plt.plot(pd, label="Prediction SOC", linewidth=2)
    plt.legend()
    plt.savefig(picture_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
