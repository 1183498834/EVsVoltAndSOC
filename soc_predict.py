# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:12:40 2022

@author: Flashy
"""
import os
import re
from glob import iglob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from DLinear import Model

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


seq_len = 60
label_len = 42
pred_len = 18
batch = 128
epochs = 100
lr = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = os.path.join('./SOC_unstand/model/soc_model.pt')

m_state_dict = torch.load(path, map_location=torch.device('cpu'))
value = m_state_dict.values()
keys=[]
for index, idx in enumerate(m_state_dict):
    if len(idx.split(".")) >= 3:
        idx = (".").join(idx.split(".")[1:])
        keys.append(idx)
    else:
        idx = (".").join(idx.split("."))
        keys.append(idx)
new_state_dict = dict(zip(keys, value))
model = Model(seq_len=seq_len, pred_len=pred_len)
model.load_state_dict(new_state_dict)

# 读取数据
data_root_path = './np_data/*'
root = iglob(data_root_path)
car_len = 0
for r in root:
    
    data_path = r+'/charge_new_feature.npy'
    print("Running {}!".format(data_path))
    data = np.load(data_path)
    battery_soc = data[:, :, 2]
    charge_features = data[:, :, :2]
    
    # 一个电池的模型应该对应一个电池
    needed = np.dstack((charge_features, battery_soc))
    # print(needed)
    
    
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
            start_x += end_y 
            if start_x + seq_len + pred_len > needed.shape[1]: 
                break
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    batch_len = data_x.shape[0]
    # print("x:", data_x.shape)
    # print("y:", data_y.shape)
    data_x = torch.FloatTensor(data_x)
    soc=[]
    soc_mse = []
    soc_mae = []
    soc_rmse = []
    soc_r2 = []
    for batch in tqdm(range(batch_len)):
        input_x = data_x[batch, :, :]
        true_y = data_y[batch, -pred_len:, :]
        # print("y:", true_y.shape)
        input_x = torch.unsqueeze(input_x, dim=0)
        
        output = model(input_x)
        output = output[:, -pred_len:, -1:]
        
        input_soc = input_x[0, :, -1].detach().cpu().numpy()
        pred_soc = output[0, :, -1].detach().cpu().numpy()

        
        gtsoc = np.concatenate((input_soc, true_y[:, -1]), axis=0)
        soc.append(gtsoc)
        pdsoc = np.concatenate((input_soc, pred_soc), axis=0)
        soc.append(pdsoc)
        
        
        # 画图保存
        picture_path = r + '\\pictures_soc'
        if not os.path.exists(picture_path):
            os.makedirs(picture_path)
        picture_path = picture_path + '\\{}.png'.format(batch)
        plt.figure()
        plt.plot(gtsoc, label='GroundTruth SOC', linewidth=2)
        plt.plot(pdsoc, label='Prediction SOC', linewidth=2)
        plt.legend()
        plt.savefig(picture_path, bbox_inches='tight')
        plt.close()
        
        
        # 保存误差
        soc_mse.append(mean_squared_error(true_y[:, -1], pred_soc))
        soc_mae.append(mean_absolute_error(true_y[:, -1], pred_soc))
        soc_rmse.append(sqrt(mean_squared_error(true_y[:, -1], pred_soc)))
        soc_r2.append(r2_score(true_y[:, -1], pred_soc))
        
        
    soc_data_path = r + '/soc_data'
    if not os.path.exists(soc_data_path):
        os.makedirs(soc_data_path)
    soc_data_path = soc_data_path + '/soc.csv'
    soc_data = pd.DataFrame(data=list(map(list, zip(*soc))))
    soc_data.to_csv(soc_data_path, encoding='gbk')
    
        
    soc_mse_path = r + '\\soc_mse'
    if not os.path.exists(soc_mse_path):
        os.makedirs(soc_mse_path)
    soc_mse_path = soc_mse_path + '\\soc.npy'
    np.save(soc_mse_path, soc_mse)
    
    soc_mae_path = r + '\\soc_mae'
    if not os.path.exists(soc_mae_path):
        os.makedirs(soc_mae_path)
    soc_mae_path = soc_mae_path + '\\soc.npy'
    np.save(soc_mae_path, soc_mae)
    
    soc_rmse_path = r + '\\soc_rmse'
    if not os.path.exists(soc_rmse_path):
        os.makedirs(soc_rmse_path)
    soc_rmse_path = soc_rmse_path + '\\soc.npy'
    np.save(soc_rmse_path, soc_rmse)
    
    soc_r2_path = r + '\\soc_r2'
    if not os.path.exists(soc_r2_path):
        os.makedirs(soc_r2_path)
    soc_r2_path = soc_r2_path + '\\soc.npy'
    np.save(soc_r2_path, soc_r2)
    
    