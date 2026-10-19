# Federated ridge vs FedAvg MLP for next-hour household load

The script downloads the UCI household power data, resamples it to hourly means, and makes a chronological 80/20 split. The training part is split into five contiguous time blocks that act as federated clients.

- **Baseline:** a FedAvg-trained scikit-learn MLP (64-32, 20 rounds, 1 local epoch) on the original lag and calendar features, averaged over seeds 42, 43 and 44.
- **Improved:** federated ridge regression on an extended feature set. Each client sends only X'X and X'y once, and the server solves for the weights. The penalty is chosen by leave-one-block-out CV on the training blocks only. The test set is used once, for the final evaluation.

The metric is test MAE in kW (lower is better).

Changes in this revision: the improved model now uses richer features (rolling 24h/168h means, 24h std, weekly same-hour mean, hour-of-day and day-of-week indicators, and hour x lag interactions). The baseline inputs, rows, split, and FedAvg schedule are unchanged.

Run with `python analysis.py`. It writes `results.json` and `figures/mae_comparison.png`.

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th. https://ieeexplore.ieee.org/abstract/document/10991402/

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
