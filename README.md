# README
## Project Overview
This repository contains the code and documentation necessary to reproduce the modeling procedure described in our paper:
"Study of the relationship between individual consumer behavior and macroeconomic trends."

The project focuses on analyzing banking transaction data to study the relationship between individual consumer behavior and macroeconomic indicators. The analysis involves preprocessing the data, constructing a consumer activity metric, fitting statistical models, and evaluating their quality.

## Authors:
Anton Kovantsev

ITMO University, Saint Petersburg, Russia

Email: ankovantcev@itmo.ru


Anna Cherkasskaya

ITMO University, Saint Petersburg, Russia

Email: anyacherkasskaya@mail.ru

## Repository Contents
The repository is structured as follows:

### Notebooks:
- modeling_1st_dataset.ipynb: Analysis of the first dataset with 10K clients.
- modeling_2nd_dataset.ipynb: Analysis of the second dataset with 25K clients.

### Python files:
- model.py: Contains all functions used for modeling, including:
  - Distribution functions (classical and modified Maxwell–Boltzmann).
  - Parameter estimation methods (MLE, MLS).
  - Model evaluation metrics (RMSE, Kolmogorov-Smirnov, Wasserstein distance).
- data.py: Provides functions for handling inflation data and other auxiliary datasets.

### Datasets:
- dataset_1: First dataset with 10K clients (included in the repository).
- dataset_2: Second dataset with 25K clients (raw version available at Google Drive ).
### Dependencies:
Python >= 3.11

Libraries: numpy, pandas, scipy, matplotlib, seaborn, statsmodels.

### Dataset Descriptions
1. **First Dataset (dataset_1)**
**Transactions:** 19,262,668 transactions from 10,000 bank clients.

**Observation Period:** January 1, 2018 – August 15, 2022.

**Columns:**

- client: Client identifier (discrete numeric).
- card: Payment card identifier (discrete numeric).
- date: Transaction date (continuous numeric, date format).
- amt: Amount spent (continuous numeric).
- mcc: Merchant category code (discrete numeric).
- group: Purchase group (categorical). Examples: 'food', 'travel', 'fun'.
- value: Basic value category (categorical):
  - survival: 'food', 'outfit', 'health', 'dwelling'.
  - socialization: 'travel', 'nonfood', 'telecom', 'misc', 'remote'.
  - self_realization: 'fun', 'kids', 'beauty', 'charity'.

2. **Second Dataset (dataset_2)**
**Transactions:** 6,938,421 transactions from 25,000 bank clients.

**Observation Period:** May 1, 2017 – December 31, 2018.

**Columns:**

Same as dataset_1.

**Raw Data:**
An unprepared version of the 2nd dataset can be found at [Google Drive](https://drive.google.com/drive/folders/1PKqYacxA3ZWsRbn8CySrPnXYor_aJFJJ).


### File Descriptions
1. **model.py**
Contains all functions used in the analysis:

**Distribution Functions:**
- clas_maxwell_boltzmann(x, B, beta): Classical Maxwell–Boltzmann distribution.
- mod_maxwell_boltzmann(x, B, beta, x0, alpha): Modified Maxwell–Boltzmann distribution.

**Parameter Estimation:**
- fit_methods(...): Fits models using MLE or MLS.
- neg_log_likelihood(...): Computes the negative log-likelihood for MLE.

**Model Evaluation:**
- compute_rmse(...): Computes RMSE.
- evaluate_fit(...): Evaluates model fit using statistical metrics.
- get_inflation_data(dataset_id): Returns inflation data for the specified dataset.

2. **data.py**

Provides inflation data.

3. **modeling_1st_dataset.ipynb**

Analysis of the first dataset with 10K clients. The modified and classical Maxwell-Boltzmann distribution is used as a continuous distribution.

4. **modeling_1st_dataset.ipynb**

Analysis of the first dataset with 25K clients. The modified Maxwell-Boltzmann distribution is used as a continuous distribution.