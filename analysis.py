"""Federated ridge regression vs FedAvg-trained MLP for next-hour household load (kW)."""
import copy, io, json, os, sys, time, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

SEED = 42
SEEDS = [42, 43, 44]                       # seeds for the randomized MLP; ridge is run with the same seeds
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
N_CLIENTS = 5                              # contiguous time blocks of the training part = simulated clients
LAGS = list(range(1, 25)) + [168]          # previous 1-24 hours and the same hour one week ago
# Baseline features (unchanged): the MLP is trained on exactly these.
BASE_FEATS = [f"lag_{k}" for k in LAGS] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
# Improved-model features: baseline features plus rolling summaries, weekly same-hour mean,
# hour-of-day and day-of-week indicators, and hour x recent-load interactions.
EXTRA_FEATS = ["roll24_mean", "roll168_mean", "roll24_std", "weekly_same_hour_mean"]
HOUR_OH = [f"hour_oh_{j}" for j in range(24)]
DOW_OH = [f"dow_oh_{j}" for j in range(7)]
HX1 = [f"hour_oh_{j}_x_lag1" for j in range(24)]
HX24 = [f"hour_oh_{j}_x_lag24" for j in range(24)]
IMP_FEATS = BASE_FEATS + EXTRA_FEATS + HOUR_OH + DOW_OH + HX1 + HX24
ROUNDS, LOCAL_EPOCHS = 20, 1               # FedAvg schedule (unchanged for the baseline)
LAMBDAS = np.logspace(-2, 6, 9)            # ridge penalty grid, selected by leave-one-block-out CV on train
OUT = "results.json"


def download(url, retries=5):
    """GET with timeout=60; retry with exponential backoff on HTTP 429 and 5xx."""
    for attempt in range(retries):
        r = requests.get(url, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            wait = 2 ** attempt
            print(f"HTTP {r.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("download failed after retries")


def load_hourly():
    """Read the minute-level zip and resample Global_active_power to hourly means (kW)."""
    print("Downloading UCI household power data...")
    zf = zipfile.ZipFile(io.BytesIO(download(URL)))
    name = next(n for n in zf.namelist() if n.endswith(".txt"))
    df = pd.read_csv(zf.open(name), sep=";", na_values="?",
                     usecols=["Date", "Time", "Global_active_power"])
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.Series(df["Global_active_power"].to_numpy(float), index=ts).sort_index()
    h = s.resample("h").mean()
    print(f"Minute rows: {len(df)}, hourly rows: {len(h)}")
    # fill only short gaps (up to 3 h); longer gaps are removed by the lag dropna below
    return h.interpolate(limit=3, limit_area="inside")


def build_frame(h):
    """One table for both models. Features at hour t use load up to t; target is load at t+1.
    Rows are selected by the baseline features only, so both models see identical rows."""
    cols = {f"lag_{k}": h.shift(k) for k in LAGS}
    nxt = h.index + pd.Timedelta(hours=1)
    hr, dw = np.asarray(nxt.hour), np.asarray(nxt.dayofweek)
    cols["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    cols["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    cols["dow_sin"] = np.sin(2 * np.pi * dw / 7)
    cols["dow_cos"] = np.cos(2 * np.pi * dw / 7)
    F = pd.DataFrame(cols, index=h.index)
    F["y"] = h.shift(-1)
    F = F.dropna(subset=BASE_FEATS + ["y"])            # same rows as the original table
    idx = F.index

    # extra features (only used by the improved model), all using data up to hour t
    past = h.shift(1)
    E = pd.DataFrame(index=h.index)
    E["roll24_mean"] = past.rolling(24, min_periods=1).mean()
    E["roll168_mean"] = past.rolling(168, min_periods=1).mean()
    E["roll24_std"] = past.rolling(24, min_periods=2).std().fillna(0.0)
    E["weekly_same_hour_mean"] = pd.concat([h.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1)
    E = E.loc[idx]
    for c in EXTRA_FEATS:
        F[c] = E[c].to_numpy()
    hr_i = np.asarray(((idx + pd.Timedelta(hours=1)).hour))
    dw_i = np.asarray(((idx + pd.Timedelta(hours=1)).dayofweek))
    for j in range(24):
        F[f"hour_oh_{j}"] = (hr_i == j).astype(float)
        F[f"hour_oh_{j}_x_lag1"] = F[f"hour_oh_{j}"] * F["lag_1"]
        F[f"hour_oh_{j}_x_lag24"] = F[f"hour_oh_{j}"] * F["lag_24"]
    for j in range(7):
        F[f"dow_oh_{j}"] = (dw_i == j).astype(float)
    F[IMP_FEATS] = F[IMP_FEATS].fillna(0.0)
    return F


def global_stats(clients):
    """Round 1 (shared by both methods): clients send counts and sums; server derives mean and std."""
    n = sum(len(yk) for _, yk in clients)
    sx = sum(Xk.sum(0) for Xk, _ in clients)
    sxx = sum((Xk ** 2).sum(0) for Xk, _ in clients)
    sy = sum(yk.sum() for _, yk in clients)
    syy = sum((yk ** 2).sum() for _, yk in clients)
    mu, ymu = sx / n, sy / n
    return mu, np.sqrt(sxx / n - mu ** 2), ymu, np.sqrt(syy / n - ymu ** 2)


def fedavg_mlp(clients, mu, sd, ymu, ysd, seed):
    """FedAvg: each round, clients run local partial_fit from the global weights; server averages by sample count."""
    scaled = [((Xk - mu) / sd, (yk - ymu) / ysd) for Xk, yk in clients]
    g = MLPRegressor(hidden_layer_sizes=(64, 32), batch_size=256, learning_rate_init=1e-3, random_state=seed)
    g.partial_fit(scaled[0][0][:1], scaled[0][1][:1])  # one-row call only creates the weight shapes
    sizes = np.array([len(yk) for _, yk in scaled], float)
    wts = sizes / sizes.sum()
    for rnd in range(ROUNDS):
        local = []
        for Xk, yk in scaled:
            m = copy.deepcopy(g)
            for _e in range(LOCAL_EPOCHS):
                m.partial_fit(Xk, yk)
            local.append((m.coefs_, m.intercepts_))
        g.coefs_ = [sum(wt * c[i] for wt, (c, _b) in zip(wts, local)) for i in range(len(g.coefs_))]
        g.intercepts_ = [sum(wt * b[i] for wt, (_c, b) in zip(wts, local)) for i in range(len(g.intercepts_))]
        if rnd % 5 == 4:
            print(f"  FedAvg round {rnd + 1}/{ROUNDS} done")
    return g


def client_stats(Z, yk, ymu, ysd):
    """Ridge client message: X'X and X'y on standardized features (with bias column) and target. Nothing else leaves the client."""
    return Z.T @ Z, Z.T @ ((yk - ymu) / ysd)


def solve_ridge(A, b, lam):
    """Server: solve (A + lam*P) w = b; the intercept (index 0) is not penalised."""
    P = np.eye(len(b)) * lam
    P[0, 0] = 0.0
    return np.linalg.solve(A + P, b)


def select_lambda(Ak, bk, Zs, yks, ymu, ysd):
    """Leave-one-block-out CV on training blocks; each held-out client scores its own block (MAE in kW)."""
    A_tot, b_tot = sum(Ak), sum(bk)
    best_mae, best_lam = np.inf, None
    for lam in LAMBDAS:
        err, cnt = 0.0, 0
        for k in range(N_CLIENTS):
            w = solve_ridge(A_tot - Ak[k], b_tot - bk[k], lam)  # exact: total minus held-out client
            err += np.abs(Zs[k] @ w * ysd + ymu - yks[k]).sum()
            cnt += len(yks[k])
        if err / cnt < best_mae:
            best_mae, best_lam = err / cnt, lam
    print(f"Ridge CV: best lambda={best_lam:.3g}, CV MAE={best_mae:.4f} kW")
    return best_lam


def fit_federated_ridge(X, y, blocks):
    """Federated ridge on training rows only: standardisation, CV lambda, and summed client statistics."""
    clients = [(X[b], y[b]) for b in blocks]
    mu, sd, ymu, ysd = global_stats(clients)
    sd = np.maximum(sd, 1e-8)
    Zs = [np.hstack([np.ones((len(yk), 1)), (Xk - mu) / sd]) for Xk, yk in clients]
    stats = [client_stats(Z, yk, ymu, ysd) for Z, (_, yk) in zip(Zs, clients)]
    Ak, bk = [s[0] for s in stats], [s[1] for s in stats]
    lam = select_lambda(Ak, bk, Zs, [yk for _, yk in clients], ymu, ysd)  # CV on train only
    w = solve_ridge(sum(Ak), sum(bk), lam)                # server sums client statistics and solves
    return {"w": w, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, "lam": lam}


def predict_ridge(m, X):
    Z = np.hstack([np.ones((len(X), 1)), (X - m["mu"]) / m["sd"]])
    return Z @ m["w"] * m["ysd"] + m["ymu"]


def evaluate(y_true, y_pred):
    """Shared test metrics for both models (kW)."""
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def summary(runs, key):
    """Mean and std over seeds for one metric."""
    v = [r[key] for r in runs]
    return float(np.mean(v)), float(np.std(v))


def main():
    np.random.seed(SEED)
    t0 = time.time()
    F = build_frame(load_hourly())
    X_base = F[BASE_FEATS].to_numpy(float)
    X_imp = F[IMP_FEATS].to_numpy(float)
    y = F["y"].to_numpy(float)
    n = len(y)
    cut = int(0.8 * n)                                    # chronological 80/20 split, done once
    blocks = np.array_split(np.arange(cut), N_CLIENTS)   # contiguous time blocks = simulated clients
    print(f"Modelling rows={n}, train={cut}, test={n - cut}, clients={N_CLIENTS}, "
          f"baseline features={len(BASE_FEATS)}, improved features={len(IMP_FEATS)}")

    # baseline inputs (unchanged)
    Xtr_b, ytr = X_base[:cut], y[:cut]
    Xte_b, yte = X_base[cut:], y[cut:]
    clients = [(Xtr_b[b], ytr[b]) for b in blocks]
    mu, sd, ymu, ysd = global_stats(clients)
    sd = np.maximum(sd, 1e-8)
    Xte_s = (Xte_b - mu) / sd

    # improved model: federated ridge on extended features, lambda chosen on train blocks only
    ridge = fit_federated_ridge(X_imp[:cut], ytr, blocks)
    w, lam = ridge["w"], ridge["lam"]
    imp_pred = predict_ridge(ridge, X_imp[cut:])          # test set used only here

    coef = w[1:] / ridge["sd"] * ridge["ysd"]             # readable coefficients in kW units
    b0 = ridge["ysd"] * w[0] + ridge["ymu"] - coef @ ridge["mu"]
    print(f"Ridge intercept={b0:.4f}; largest coefficients:")
    for i in np.argsort(-np.abs(coef))[:8]:
        print(f"  {IMP_FEATS[i]}: {coef[i]:.4f}")

    base_runs, imp_runs = [], []
    for seed in SEEDS:
        print(f"Seed {seed}: FedAvg MLP baseline")
        mlp = fedavg_mlp(clients, mu, sd, ymu, ysd, seed)
        base_runs.append(evaluate(yte, mlp.predict(Xte_s) * ysd + ymu))
        imp_runs.append(evaluate(yte, imp_pred))

    bm, bs = summary(base_runs, "mae")
    im, ist = summary(imp_runs, "mae")
    metrics = {
        "baseline_score": bm, "improved_score": im, "improvement": bm - im,
        "baseline_score_std": bs, "improved_score_std": ist,
        "baseline_rmse": summary(base_runs, "rmse")[0], "improved_rmse": summary(imp_runs, "rmse")[0],
        "baseline_r2": summary(base_runs, "r2")[0], "improved_r2": summary(imp_runs, "r2")[0],
        "ridge_lambda": float(lam), "n_seeds": len(SEEDS),
        "n_baseline_features": len(BASE_FEATS), "n_improved_features": len(IMP_FEATS),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["FedAvg MLP", "Federated ridge"], [bm, im], yerr=[bs, ist], capsize=6,
            color=["grey", "steelblue"])
    plt.ylabel("Test MAE (kW), lower is better")
    plt.title("Next-hour household load")
    plt.tight_layout()
    plt.savefig("figures/mae_comparison.png", dpi=120)
    plt.close()

    out = {
        "question": "Can a federated ridge regression, which aggregates exact sufficient statistics and exposes "
                    "readable coefficients, match or beat a FedAvg-trained MLP for next-hour household load forecasting?",
        "method": "Hourly mean Global_active_power with lag and calendar features; chronological 80/20 split; "
                  "training part divided into contiguous time-block clients; both models trained federatedly and "
                  "evaluated on the same held-out test period.",
        "dataset": {
            "name": "UCI Individual Household Electric Power Consumption (resampled to hourly)",
            "source_url": URL, "n_rows": int(n), "n_train": int(cut), "n_test": int(n - cut),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "Test MAE (kW)",
        "higher_is_better": False,
        "baseline": "FedAvg-trained scikit-learn MLPRegressor (64-32 ReLU, Adam, local partial_fit, "
                    "20 rounds of 1 local epoch, sample-weighted averaging of weights across 5 clients), "
                    "on the original lag and calendar features.",
        "improved": "Federated ridge regression on extended features (original lags and calendar, rolling 24h/168h "
                    "means, 24h std, weekly same-hour mean, hour-of-day and day-of-week indicators, and hour x lag_1 / "
                    "hour x lag_24 interactions). Each client sends only X'X and X'y once; the server sums them and "
                    "solves; lambda chosen by leave-one-block-out CV on the training blocks.",
        "metrics": metrics,
        "notes": "Single chronological split of one household series; the clients are time blocks, so they are "
                 "non-IID by construction. Both models use the same rows and the same test period; the baseline "
                 "inputs are unchanged. The improvement comes only from the richer linear feature map for ridge, "
                 "with lambda chosen by leave-one-block-out CV on training data. Only the MLP is randomized; ridge "
                 "is deterministic, so its seed spread is zero. The MLP is untuned, so the comparison is not a "
                 "fully tuned benchmark. The motivating paper's model and results were not available, so no paper "
                 "numbers are reproduced.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline MAE={bm:.4f}+-{bs:.4f} kW, improved MAE={im:.4f}+-{ist:.4f} kW, "
          f"improvement={bm - im:.4f} kW. Wrote {OUT} in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
