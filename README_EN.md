# EVsVoltAndSOC

2022 Digital Vehicle Competition, Innovation Group, Topic 2: **New Energy Vehicle Power Battery Safety Risk Assessment and Fault Warning (prediction on charging data)**.

**Language / 语言:** [中文](README.md) · English

## Problem Description

Original problem and data source: [Digital Vehicle Competition official website](http://www.ncbdc.top/competition/innovate?coid=54)

> Based on the GB/T 32960-2016 standard data, perform power battery fault warning for online running vehicles. Preprocess the raw data, model the vehicle's charging, driving, and stationary conditions, analyze the battery's operating state to build a battery fault assessment system, complete the warning model training, and output battery fault warnings to predict battery faults in advance and improve vehicle safety.

## Approach

Based on the national standard, custom alarm criteria are established for EV data, with alarms divided into **immediate alarms** and **early warnings**: predictable data is forecast to provide early warnings, while unpredictable data triggers immediate alarms.

This project focuses on predicting EV charging data, i.e., the battery's SOC and voltage during charging:

- **SOC prediction**: refers to the paper [*Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504), using its **DLinear** neural network;
- **Voltage prediction**: improves upon DLinear by replacing its linear layers with **LSTM** layers, resulting in the **DLSTM** network.

The overall pipeline:

1. Preprocess the data and remove outliers;
2. Split the data, selecting total current, total voltage, SOC, etc. as features to form inputs and labels;
3. Build and train the neural network;
4. Feed in data for prediction and inspect the results.

## Project Structure

```
.
├── dlinear.py          # DLinear model (SOC prediction)
├── dlstm.py            # DLSTM model (cell voltage prediction)
├── decomposition.py    # Time-series decomposition (MovingAvg / SeriesDecomp)
├── data_utils.py       # Data loading, sliding window, splitting, normalization utilities
├── run_SOC.py          # Train the SOC prediction model
├── run_U_DLSTM.py      # Train the cell voltage prediction model (per-cell)
├── soc_predict.py      # Predict SOC with the trained model and evaluate
├── volt_predict.py     # Predict cell voltage with the trained model and evaluate
├── setup.py            # Package metadata
├── requirements.txt    # Dependencies
└── charge_new_feature.npy  # Sample data: shape (150, 147, 98)
```

## Dependencies

- Python ≥ 3.7
- PyTorch
- NumPy / pandas / scikit-learn / matplotlib / tqdm

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

3. Predict SOC with the trained model:

   ```bash
   python soc_predict.py
   ```

4. Predict voltage with the trained model:

   ```bash
   python volt_predict.py
   ```

## Data Description

`charge_new_feature.npy` is sample charging data with shape `(150, 147, 98)`:

- Dimension 0: 150 charging cycles;
- Dimension 1: 147 time steps;
- Dimension 2: 98 channels, in order:
  - Channel 0: total voltage;
  - Channel 1: total current;
  - Channel 2: SOC;
  - Channel 3 onwards: individual cell voltages.

The prediction scripts read vehicle data from `./np_data/*/charge_new_feature.npy` by default; adjust the directory as needed.

## Results

A segment of prediction results is shown below, demonstrating good prediction performance:

![SOC prediction result](0.png)
