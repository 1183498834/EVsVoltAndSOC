# EVsVoltAndSOC

2022 年数字汽车大赛创新组赛题二：**新能源汽车动力电池安全风险评估与故障预警（对充电数据进行预测）**。

本项目基于国标 GB/T 32960-2016 的车辆数据，对电动车充电过程中的动力电池 **SOC（荷电状态）** 与 **单体电池电压** 进行预测，用于电池故障预警。

## 赛题描述

原题及数据来源：[数字汽车大赛官网](http://www.ncbdc.top/competition/innovate?coid=54)

> 基于国标 GB/T 32960-2016 的数据，对在线运行车辆进行动力电池故障预警。对原始数据进行数据预处理，从车辆的充电、行驶、静置等工况进行数据建模，通过对电池的运行状态进行分析，建立电池故障评估体系，完成预警模型训练，输出电池故障预警，提前预测电池故障，提高汽车安全性。

## 解决思路

根据国标对电动车的数据建立自定义报警标准，将警报分为**报警**与**预警**两类：对可以预测的数据进行预测，从而提前预警；对无法预测的数据进行即时报警。

本项目选择对电动车充电数据进行预测，即预测电池充电时的 SOC 与电压：

- **SOC 预测**：参考论文 [*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504)，采用其中的 **DLinear** 神经网络；
- **电压预测**：在 DLinear 的基础上改进，将网络中的线性层替换为 **LSTM** 层，得到 **DLSTM** 网络。

具体流程如下：

1. 对数据进行预处理，去除异常值；
2. 对数据进行划分，选取总电流、总电压、SOC 等数据作为参数，划分输入数据与标签数据；
3. 构建神经网络并进行训练；
4. 输入数据进行预测并查看结果。

## 项目结构

```
.
├── dlinear.py          # DLinear 模型（SOC 预测）
├── dlstm.py            # DLSTM 模型（单体电压预测）
├── decomposition.py    # 时间序列分解（MovingAvg / SeriesDecomp）
├── data_utils.py       # 数据加载、滑动窗口、切分、标准化等工具
├── run_SOC.py          # 训练 SOC 预测模型
├── run_U_DLSTM.py      # 训练单体电压预测模型（逐电池训练）
├── soc_predict.py      # 用训练好的模型预测 SOC 并评估
├── volt_predict.py     # 用训练好的模型预测单体电压并评估
├── setup.py            # 包信息
├── requirements.txt    # 依赖
└── charge_new_feature.npy  # 示例数据：形状 (150, 147, 98)
```

## 环境依赖

- Python ≥ 3.7
- PyTorch
- NumPy / pandas / scikit-learn / matplotlib / tqdm

安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练 SOC 预测模型（DLinear）：

   ```bash
   python run_SOC.py
   ```

2. 训练单体电压预测模型（DLSTM）：

   ```bash
   python run_U_DLSTM.py
   ```

3. 用训练好的模型做 SOC 预测：

   ```bash
   python soc_predict.py
   ```

4. 用训练好的模型做电压预测：

   ```bash
   python volt_predict.py
   ```

## 数据说明

`charge_new_feature.npy` 为示例充电数据，形状为 `(150, 147, 98)`，含义如下：

- 第 0 维：150 次充电循环；
- 第 1 维：147 个时间步；
- 第 2 维：98 个通道，依次为：
  - 通道 0：总电压；
  - 通道 1：总电流；
  - 通道 2：SOC；
  - 通道 3 起：各单体电池电压。

预测脚本默认从 `./np_data/*/charge_new_feature.npy` 读取待预测车辆数据，具体目录可按需调整。

## 结果

其中一段预测结果如下，可以看出预测效果较好：

![SOC 预测结果](0.png)
