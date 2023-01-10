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
from DLSTM import Model

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

path = os.path.join('./result/model/*.pt')
files = iglob(path)
for i in files:
    cell_num = int(re.split('cell|\_', i)[1]) - 1
    print("Running the {} cell.".format(cell_num))
    """"""
    m_state_dict = torch.load(i, map_location=torch.device('cpu'))
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
        battery_volt = data[:, :, 3:]
        charge_features = data[:, :, :3]
        
        # 一个电池的模型应该对应一个电池
        needed = np.dstack((charge_features, battery_volt[:, :, cell_num]))
        # print(needed)
        
        volt_scaler = StandardScaler()
        current_scaler = StandardScaler()
        soc_scaler = StandardScaler()
        cell_scaler = StandardScaler()
        
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
                start_x += end_y 
                if start_x + seq_len + pred_len > needed.shape[1]: 
                    break
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        batch_len = data_x.shape[0]
        # print("x:", data_x.shape)
        # print("y:", data_y.shape)
        data_x = torch.FloatTensor(data_x)
        volt=[]
        volt_mse = []
        volt_mae = []
        volt_rmse = []
        volt_r2 = []
        for batch in tqdm(range(batch_len)):
            input_x = data_x[batch, :, :]
            true_y = data_y[batch, -pred_len:, :]
            # print("y:", true_y.shape)
            input_x = torch.unsqueeze(input_x, dim=0)
            
            output = model(input_x)
            output = output[:, -pred_len:, -2:]
            
            input_volt = input_x[0, :, -1].detach().cpu().numpy()
            pred_volt = output[0, :, -1].detach().cpu().numpy()

            
            gd_volt = cell_scaler.inverse_transform(input_volt.reshape(-1, 1)).reshape(-1)
            
            inverse_volt = cell_scaler.inverse_transform(pred_volt.reshape(-1, 1)).reshape(-1)
            
            true_volt = cell_scaler.inverse_transform(true_y[:, -1].reshape(-1, 1)).reshape(-1)
            
            gtvolt = np.concatenate((gd_volt, true_volt), axis=0)
            volt.append(gtvolt)
            pdvolt = np.concatenate((gd_volt, inverse_volt), axis=0)
            volt.append(pdvolt)
            
            
            # 画图保存
            picture_path = r + '\\pictures_volt\\{}'.format(cell_num)
            if not os.path.exists(picture_path):
                os.makedirs(picture_path)
            picture_path = picture_path + '\\{}.png'.format(batch)
            plt.figure()
            plt.plot(gtvolt, label='GroundTruth Volt', linewidth=2)
            plt.plot(pdvolt, label='Prediction Volt', linewidth=2)
            plt.legend()
            plt.savefig(picture_path, bbox_inches='tight')
            plt.close()
            
            
            # 保存误差
            volt_mse.append(mean_squared_error(true_volt, inverse_volt))
            volt_mae.append(mean_absolute_error(true_volt, inverse_volt))
            volt_rmse.append(sqrt(mean_squared_error(true_volt, inverse_volt)))
            volt_r2.append(r2_score(true_volt, inverse_volt))
            
            
        volt_data_path = r + '/volt_data'
        if not os.path.exists(volt_data_path):
            os.makedirs(volt_data_path)
        volt_data_path = volt_data_path + '/cell_{}.csv'.format(cell_num)
        volt_data = pd.DataFrame(data=list(map(list, zip(*volt))))
        volt_data.to_csv(volt_data_path, encoding='gbk')
        
            
        volt_mse_path = r + '\\volt_mse'
        if not os.path.exists(volt_mse_path):
            os.makedirs(volt_mse_path)
        volt_mse_path = volt_mse_path + '\\cell_{}.npy'.format(cell_num)
        np.save(volt_mse_path, volt_mse)
        
        volt_mae_path = r + '\\volt_mae'
        if not os.path.exists(volt_mae_path):
            os.makedirs(volt_mae_path)
        volt_mae_path = volt_mae_path + '\\cell_{}.npy'.format(cell_num)
        np.save(volt_mae_path, volt_mae)
        
        volt_rmse_path = r + '\\volt_rmse'
        if not os.path.exists(volt_rmse_path):
            os.makedirs(volt_rmse_path)
        volt_rmse_path = volt_rmse_path + '\\cell_{}.npy'.format(cell_num)
        np.save(volt_rmse_path, volt_rmse)
        
        volt_r2_path = r + '\\volt_r2'
        if not os.path.exists(volt_r2_path):
            os.makedirs(volt_r2_path)
        volt_r2_path = volt_r2_path + '\\cell_{}.npy'.format(cell_num)
        np.save(volt_r2_path, volt_r2)
        
        