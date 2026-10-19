# Federated ridge vs FedAvg MLP for next-hour household load

The script downloads the UCI household power data, resamples it to hourly means, and makes a chronological 80/20 split. The training part is split into five contiguous time blocks that act as federated clients.

- **Baseline:** a FedAvg-trained scikit-learn MLP (64-32, 20 rounds, 1 local epoch) on the original lag and calendar features, averaged over seeds 42, 43 and 44.
- **Improved:** federated ridge regression on an extended feature set. Each client sends only X'X and X'y once, and the server solves for the weights. The penalty is chosen by leave-one-block-out CV on the training blocks only. The test set is used once, for the final evaluation.

The metric is test MAE in kW (lower is better).

Changes in this revision: the improved model now uses richer features (rolling 24h/168h means, 24h std, weekly same-hour mean, hour-of-day and day-of-week indicators, and hour x lag interactions). The baseline inputs, rows, split, and FedAvg schedule are unchanged.

Run with `python analysis.py`. It writes `results.json` and `figures/mae_comparison.png`.

## Results (latest run)

On the UCI household power dataset (hourly resampled, chronological test split), a federated ridge regression reached a test MAE of 0.419 kW versus 0.439 kW for a standard FedAvg-trained MLP forecaster, a small gain of about 0.019 kW.

| Metric | Value |
|---|---|
| baseline_score | 0.4386 |
| improved_score | 0.4194 |
| improvement | 0.01928 |
| baseline_score_std | 0.007865 |
| improved_score_std | 0 |
| baseline_rmse | 0.5915 |
| improved_rmse | 0.5732 |
| baseline_r2 | 0.3812 |
| improved_r2 | 0.4189 |
| ridge_lambda | 0.01 |
| n_seeds | 3 |
| n_baseline_features | 29 |
| n_improved_features | 112 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single chronological split of one household series; the clients are time blocks, so they are non-IID by construction. Both models use the same rows and the same test period; the baseline inputs are unchanged. The improvement comes only from the richer linear feature map for ridge, with lambda chosen by leave-one-block-out CV on training data. Only the MLP is randomized; ridge is deterministic, so its seed spread is zero. The MLP is untuned, so the comparison is not a fully tuned benchmark. The motivating paper's model and results were not available, so no paper numbers are reproduced.

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th. https://ieeexplore.ieee.org/abstract/document/10991402/

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
