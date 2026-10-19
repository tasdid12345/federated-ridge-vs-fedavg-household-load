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
FEATS = [f"lag_{k}" for k in LAGS] + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
ROUNDS, LOCAL_EPOCHS = 20, 1               # FedAvg schedule (shared budget guidance: small and fast)
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


def make_table(h):
    """Features at hour t use load up to t plus calendar of t+1; target is load at t+1."""
    X = pd.DataFrame({f"lag_{k}": h.shift(k) for k in LAGS}, index=h.index)
    nxt = h.index + pd.Timedelta(hours=1)
    hr, dw = np.asarray(nxt.hour), np.asarray(nxt.dayofweek)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    X["dow_sin"], X["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    t = X.assign(y=h.shift(-1)).dropna()
    return t[FEATS].to_numpy(float), t["y"].to_numpy(float)


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
    X, y = make_table(load_hourly())
    n = len(y)
    cut = int(0.8 * n)                                    # chronological 80/20 split, done once
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    blocks = np.array_split(np.arange(cut), N_CLIENTS)   # contiguous time blocks = simulated clients
    clients = [(Xtr[b], ytr[b]) for b in blocks]
    print(f"Modelling rows={n}, train={cut}, test={n - cut}, clients={N_CLIENTS}")

    mu, sd, ymu, ysd = global_stats(clients)
    sd = np.maximum(sd, 1e-8)
    Xte_s = (Xte - mu) / sd
    Zs = [np.hstack([np.ones((len(yk), 1)), (Xk - mu) / sd]) for Xk, yk in clients]
    stats = [client_stats(Z, yk, ymu, ysd) for Z, (_, yk) in zip(Zs, clients)]
    Ak, bk = [s[0] for s in stats], [s[1] for s in stats]
    lam = select_lambda(Ak, bk, Zs, [yk for _, yk in clients], ymu, ysd)  # CV on train only
    Zte = np.hstack([np.ones((len(yte), 1)), Xte_s])
    w = solve_ridge(sum(Ak), sum(bk), lam)                # server sums client statistics and solves

    coef = w[1:] / sd * ysd                               # readable coefficients in kW units
    b0 = ysd * w[0] + ymu - coef @ mu
    print(f"Ridge intercept={b0:.4f}; largest coefficients:")
    for i in np.argsort(-np.abs(coef))[:5]:
        print(f"  {FEATS[i]}: {coef[i]:.4f}")

    base_runs, imp_runs = [], []
    for seed in SEEDS:
        print(f"Seed {seed}: FedAvg MLP baseline")
        mlp = fedavg_mlp(clients, mu, sd, ymu, ysd, seed)
        base_runs.append(evaluate(yte, mlp.predict(Xte_s) * ysd + ymu))
        imp_runs.append(evaluate(yte, Zte @ w * ysd + ymu))  # test set used only here, once

    bm, bs = summary(base_runs, "mae")
    im, ist = summary(imp_runs, "mae")
    metrics = {
        "baseline_score": bm, "improved_score": im, "improvement": bm - im,
        "baseline_score_std": bs, "improved_score_std": ist,
        "baseline_rmse": summary(base_runs, "rmse")[0], "improved_rmse": summary(imp_runs, "rmse")[0],
        "baseline_r2": summary(base_runs, "r2")[0], "improved_r2": summary(imp_runs, "r2")[0],
        "ridge_lambda": float(lam), "n_seeds": len(SEEDS),
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
                    "20 rounds of 1 local epoch, sample-weighted averaging of weights across 5 clients).",
        "improved": "Federated ridge regression: each client sends only X'X and X'y once; the server sums them "
                    "and solves; lambda chosen by leave-one-block-out CV on the training blocks; coefficients "
                    "are readable in kW units.",
        "metrics": metrics,
        "notes": "Single chronological split of one household series; the clients are time blocks, so they are "
                 "non-IID by construction. Hourly aggregation (about two million minute rows reduced to tens of "
                 "thousands of hours) is well under the row cap, so no subsampling was needed. Only the MLP is "
                 "randomized; ridge is deterministic, so its seed spread is zero. The MLP is untuned and the ridge "
                 "is linear, so the comparison is not a fully tuned benchmark. The motivating paper's model and "
                 "results were not available, so no paper numbers are reproduced.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Baseline MAE={bm:.4f}+-{bs:.4f} kW, improved MAE={im:.4f}+-{ist:.4f} kW, "
          f"improvement={bm - im:.4f} kW. Wrote {OUT} in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
