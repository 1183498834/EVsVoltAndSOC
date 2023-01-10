# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:30:43 2022

@author: Flashy
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from DLinear import Model

# seq_len = label_len + pred_len
seq_len = 60
label_len = 42
pred_len = 18
batch = 128
epochs = 50
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = './charge_new_feature.npy'
original = np.load(path)
    

# [Batch, Seq len, Features]
charge_features = original[:, :, :3]
charge_features[:, :, 1] = (charge_features[:, :, 1]/0.1 + 2000) *0.1
needed = charge_features

    
# 滑动窗口弄出需要的seq len，然后按充电次数叠在一起，放在batch那一列，所以batch会扩充。
data_x = []
data_y = []
for times in range(needed.shape[0]):
    start_x = 0
    while True:
        end_x = start_x + seq_len
        start_y = end_x - label_len
        end_y = start_y + label_len + pred_len
        data_x.append(needed[times, start_x:end_x, :])
        data_y.append(needed[times, start_y:end_y, :])
        start_x += 1 
        if start_x + seq_len + pred_len > needed.shape[1]: 
            break
data_x = np.array(data_x)
data_y = np.array(data_y)


train_x = data_x[:int(len(data_x)*0.7), :, :]
test_x = data_x[int(len(data_x)*0.7):, :, :]
train_y = data_y[:int(len(data_y)*0.7), :, :]
test_y = data_y[int(len(data_y)*0.7):, :, :]
trainset = np.array([train_x, train_y]).transpose(1, 2, 3, 0)
testset = np.array([test_x, test_y]).transpose(1, 2, 3, 0)
train_loader = DataLoader(dataset=trainset, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=testset, batch_size=batch, shuffle=False, num_workers=0, drop_last=False)


print("train!")
model = Model(seq_len=seq_len, pred_len=pred_len)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
train_losses = []
test_losses = []
best_test_loss = float("inf")
best_train_loss = float("inf")
best_model = None

for epoch in range(epochs):
    iter_count = 0
    total_loss = 0
    
    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        batch_x = data[:, :, :, 0]
        batch_y = data[:, :, :, 1]
        iter_count += 1
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        
        outputs = model(batch_x)
        outputs = outputs[:, -pred_len:, -1]
        batch_y = batch_y[:, -pred_len:, -1].to(device)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % 2 == 0 and i > 0:
            pbar.set_description(f"{epoch+1}/{epochs}")
            pbar.set_postfix(loss=total_loss/iter_count)
    train_losses.append(total_loss/iter_count)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_count = 0
        for idx, test_data in enumerate(test_loader):
            batch_test_x = test_data[:, :, :, 0]
            batch_test_y = test_data[:, :, :, 1]
            test_count += 1
            batch_test_x = batch_test_x.float().to(device)
            batch_test_y = batch_test_y.float().to(device)
            
            outputs = model(batch_test_x)
            outputs = outputs[:, -pred_len:, -2:]
            batch_test_y = batch_test_y[:, -pred_len:, -2:].to(device)           
            outputs = outputs.detach().cpu().numpy()
            batch_test_y = batch_test_y.detach().cpu().numpy()
            
            # 画图
            pred = outputs
            true = batch_test_y
            
            picture_soc_path = './SOC_unstand/pictures/soc_epoch{}'.format(epoch)
            if not os.path.exists(picture_soc_path):
                os.makedirs(picture_soc_path)
            picture_soc_path = picture_soc_path + '/{}.png'.format(idx)
            
            if idx % 20 == 0:
                input = batch_test_x.detach().cpu().numpy()
                gtsoc = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pdsoc = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                plt.figure()
                plt.plot(gtsoc, label='GroundTruth SOC', linewidth=2)
                plt.plot(pdsoc, label='Prediction SOC', linewidth=2)
                plt.legend()
                plt.savefig(picture_soc_path, bbox_inches='tight')
                plt.close()
                
            
        test_losses.append(test_loss/test_count)
    
    model_path = './SOC_unstand/model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = model_path + '/soc_model.pt'
    if total_loss < best_train_loss:
        best_train_loss = total_loss
        best_model = model
        torch.save(best_model.state_dict(), model_path)
        
    scheduler.step(total_loss)

