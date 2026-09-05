# EVsVoltAndSOC

> New Energy Vehicle Power Battery Safety Risk Assessment and Fault Warning — predicting the power battery's **SOC** and **cell voltage** from GB/T 32960-2016 charging data.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c)](https://pytorch.org/)
[![Competition](https://img.shields.io/badge/2022DigitalVehicleCompetition-Innovation-green)](http://www.ncbdc.top/competition/innovate?coid=54)

**Language / 语言：** [中文](README.md) · English

## Table of Contents

- [Problem Description](#problem-description)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Data Description](#data-description)
- [Results](#results)

## Problem Description

Original problem and data source: [Digital Vehicle Competition official website](http://www.ncbdc.top/competition/innovate?coid=54)

> Based on the GB/T 32960-2016 standard data, perform power battery fault warning for online running vehicles. Preprocess the raw data, model the vehicle's charging, driving, and stationary conditions, analyze the battery's operating state to build a battery fault assessment system, complete the warning model training, and output battery fault warnings to predict battery faults in advance and improve vehicle safety.

## Approach

Based on the national standard, custom alarm criteria are established for EV data, with alarms divided into **immediate alarms** and **early warnings**: predictable data is forecast to provide early warnings, while unpredictable data triggers immediate alarms.

This project focuses on predicting EV charging data, i.e., the battery's SOC and voltage during charging:

| Target | Model | Description |
| ------ | ----- | ----------- |
| SOC | DLinear | Based on [*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504) |
| Cell voltage | DLSTM | Replaces DLinear's linear layers with LSTM layers |

The overall pipeline:

1. Preprocess the data and remove outliers;
2. Split the data, selecting total current, total voltage, SOC, etc. as features to form inputs and labels;
3. Build and train the neural network;
4. Feed in data for prediction and inspect the results.

## Project Structure

```
.
├── dlinear.py              # DLinear model (SOC prediction)
├── dlstm.py                # DLSTM model (cell voltage prediction)
├── decomposition.py        # Time-series decomposition (MovingAvg / SeriesDecomp)
├── data_utils.py           # Data loading, sliding window, splitting, normalization utilities
├── run_SOC.py              # Train the SOC prediction model
├── run_U_DLSTM.py          # Train the cell voltage prediction model (per-cell)
├── soc_predict.py          # Predict SOC with the trained model and evaluate
├── volt_predict.py         # Predict cell voltage with the trained model and evaluate
├── setup.py                # Package metadata
├── requirements.txt        # Dependencies
└── charge_new_feature.npy  # Sample data: shape (150, 147, 98)
```

## Dependencies

| Dependency | Purpose |
| ---------- | ------- |
| Python ≥ 3.7 | Runtime |
| PyTorch | Neural network framework |
| NumPy / pandas | Data processing |
| scikit-learn | Normalization and evaluation metrics |
| matplotlib | Visualization |
| tqdm | Training progress bar |

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train the SOC prediction model (DLinear):

   ```bash
   python run_SOC.py
   ```

2. Train the cell voltage prediction model (DLSTM):

   ```bash
   python run_U_DLSTM.py
   ```

3. Predict SOC 