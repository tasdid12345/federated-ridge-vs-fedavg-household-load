# Federated Ridge vs FedAvg MLP for Household Load Forecasting

Smart-home energy forecasting is often proposed as a federated learning task, because household data stays on the device and only model updates are shared. The paper "Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes" motivates this setting. Neural networks trained with FedAvg are the common choice, but their weights are hard to interpret and their training needs many communication rounds. This experiment asks whether a simpler federated linear model can do as well.

**Question:** Can a federated ridge regression, which aggregates exact sufficient statistics and exposes readable coefficients, match or beat a FedAvg-trained MLP for next-hour household load forecasting?

**Dataset:** UCI Individual Household Electric Power Consumption, minute-level data resampled to hourly mean active power. It is not the paper's own dataset (`same_as_paper` is false). The series is split chronologically 80/20. The training part is divided into contiguous time blocks that act as simulated federated clients.

**Baseline:** a FedAvg-trained scikit-learn `MLPRegressor`. Clients run local `partial_fit` from the global weights, and the server averages the weights.

**Improved model:** federated ridge regression. Each client sends only X'X and X'y once. The server sums them and solves for the coefficients, which can be read directly in kW units. The penalty is chosen by leave-one-block-out cross-validation on the training data only.

**Features:** lags 1–24 and 168 hours of load, plus cyclic hour-of-day and day-of-week features of the target hour.

**Metric:** test mean absolute error (kW), with RMSE and R² as secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json`. A bar chart is saved to `figures/mae_comparison.png`.

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th. https://ieeexplore.ieee.org/abstract/document/10991402/

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
