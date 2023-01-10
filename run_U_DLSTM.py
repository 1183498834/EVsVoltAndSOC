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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from DLSTM import Model

# seq_len = label_len + pred_len
seq_len = 60
label_len = 42
pred_len = 18
batch = 128
epochs = 100
lr = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

volt_scaler = StandardScaler()
current_scaler = StandardScaler()
soc_scaler = StandardScaler()
cell_scaler = StandardScaler()


path = './charge_new_feature.npy'
original = np.load(path)
    

# [Batch, Seq len, Features]
battery_volt = original[:, :, 3:]*1000
 
battery_num = battery_volt.shape[2]
charge_features = original[:, :, :3]

for n_battery in range(battery_num):
    needed = np.dstack((charge_features, battery_volt[:, :, n_battery]))

     
    # 数据标准化
    for times in range(needed.shape[0]):
        needed[times, :, 0] = volt_scaler.fit_transform(needed[times, :, 0].reshape(-1, 1)).reshape(-1)
        needed[times, :, 1] = current_scaler.fit_transform(needed[times, :, 1].reshape(-1, 1)).reshape(-1)
        needed[times, :, 2] = soc_scaler.fit_transform(needed[times, :, 2].reshape(-1, 1)).reshape(-1)
        needed[times, :, 3] = cell_scaler.fit_transform(needed[times, :, 3].reshape(-1, 1)).reshape(-1)
    
    
    # 滑动窗口弄出需要的seq len，然后按充电次数叠在一起，放在batch那一列，所以batch会从现在的150扩充。
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

    
    train_loader = DataLoader(dataset=trainset, batch_size=batch, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(dataset=testset, batch_size=batch, shuffle=False, num_workers=4, drop_last=False)
    
    
    
    print("train!")
    model = Model(seq_len=seq_len, pred_len=pred_len)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
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
            outputs = outputs[:, -pred_len:, -1:]
            batch_y = batch_y[:, -pred_len:, -1:].to(device)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 2 == 0 and i > 0:
                pbar.set_description(f"{n_battery+1}/{battery_num} : {epoch+1}/{epochs}")
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
                outputs = outputs[:, -pred_len:, -1:]
                batch_test_y = batch_test_y[:, -pred_len:, -1:].to(device)           
                outputs = outputs.detach().cpu().numpy()
                batch_test_y = batch_test_y.detach().cpu().numpy()
                
                # 画图
                pred = outputs
                true = batch_test_y
                
                picture_path = './cell_scale/pictures/battery{}'.format(n_battery+1)
                if not os.path.exists(picture_path):
                    os.makedirs(picture_path)
                picture_path = picture_path + '/{}.png'.format(idx)
                
                if idx % 2 == 0:
                    input = batch_test_x.detach().cpu().numpy()
                    input_u = cell_scaler.inverse_transform(input[0, :, -1].reshape(-1, 1)).reshape(-1)
                    true_u = cell_scaler.inverse_transform(true[0, :, -1].reshape(-1, 1)).reshape(-1)
                    pred_u = cell_scaler.inverse_transform(pred[0, :, -1].reshape(-1, 1)).reshape(-1)
                    gt = np.concatenate((input_u/1000, true_u/1000), axis=0)
                    pd = np.concatenate((input_u/1000, pred_u/1000), axis=0)
                    plt.figure()
                    plt.plot(gt, label='GroundTruth', linewidth=2)
                    plt.plot(pd, label='Prediction', linewidth=2)
                    plt.legend()
                    plt.savefig(picture_path, bbox_inches='tight')
                    plt.close()
                
            test_losses.append(test_loss/test_count)
        
        model_path = './cell_scale/model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path + '/cell{}_model.pt'.format(n_battery+1)
        if total_loss < best_train_loss:
            best_train_loss = total_loss
            best_model = model
            torch.save(best_model.state_dict(), model_path)
            
        scheduler.step(total_loss)
        
