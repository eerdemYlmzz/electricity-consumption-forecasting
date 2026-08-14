"""
Revised single-step forecasting experiment.

Task definition
---------------
Given the actual consumption measurements of the past hour, predict the
consumption of the next 10-minute step.

The series is sampled every 10 minutes, so one hour of history is 6 steps:

    SEQUENCE_LENGTH  = 6   (window length, not stride; the stride is 1)
    FORECAST_HORIZON = 1   (one step ahead = 10 minutes)

This is sliding-window one-step-ahead prediction. Every prediction is made from
six *observed* values. The model is never fed its own predictions, so the
results must not be read as a long-horizon forecast.

Protocol
--------
* The raw series is split chronologically into 70/10/20 partitions *before*
  windowing, and windows are generated independently inside each partition.
  Nothing is shuffled.
* A StandardScaler is fitted on the training partition only and applied to all
  three partitions. Predictions and targets are inverse-transformed before any
  metric is computed, so every reported number is in the original consumption
  unit.
* Three training-free reference models are evaluated on the same targets.
* The architecture mandated by the revision (hidden 32, one layer) is trained,
  and a larger configuration is trained under an otherwise identical protocol so
  that the choice of capacity can be argued from validation results.

Unit note
---------
The unit of the consumption column is not documented in the source dataset, so
no physical unit is printed or plotted. Axes and tables read "Consumption".
"""

import copy
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_PATH = Path("data/data_processed.csv")
PLOTS_DIR = Path("plots")
MODEL_DIR = Path("model")
LOG_FILE = Path("hyperparameter_logs.json")

SEQUENCE_LENGTH = 6  # window length: 6 x 10 minutes = 1 hour of history
FORECAST_HORIZON = 1  # predict the next 10-minute step
STEPS_PER_DAY = 144  # 24 h / 10 min, used by the seasonal naive reference

INPUT_SIZE = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 8
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
DPI = 300

# The datetime column in data_processed.csv was overwritten during preprocessing
# and no longer matches the source data. The true index is reconstructed here:
# 52416 rows at a 10-minute stride is exactly 364 days starting 2017-01-01.
SERIES_START = "2017-01-01 00:00"
SERIES_FREQ = "10min"

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_series():
    data = pd.read_csv(DATA_PATH)
    values = data["consumption"].values.astype(np.float32).reshape(-1, 1)
    index = pd.date_range(SERIES_START, periods=len(values), freq=SERIES_FREQ)
    return values, index


def split_bounds(n):
    """Chronological partition boundaries over the raw series."""
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }


def create_sequences(values, offset):
    """Window a single partition.

    Returns the inputs, the targets, and the index of each target in the full
    series. Windowing happens after the split, so no window can span a partition
    boundary.
    """
    xs, ys, target_index = [], [], []
    last_start = len(values) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
    for i in range(last_start):
        target = i + SEQUENCE_LENGTH + FORECAST_HORIZON - 1
        xs.append(values[i : i + SEQUENCE_LENGTH])
        ys.append(values[target])
        target_index.append(offset + target)
    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
        np.array(target_index),
    )


def compute_metrics(y_true, y_pred):
    """All metrics in the original consumption unit."""
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error / y_true)) * 100.0),
        "r2": 1.0 - ss_res / ss_tot,
    }


def reference_predictions(series, target_index):
    """Training-free reference models, evaluated on the same targets.

    Every lookup is strictly backwards in time, so none of these leak.

      persistence    y_hat[t] = y[t-1]                  last observed value
      seasonal naive y_hat[t] = y[t-144]                same time of day, previous day
      moving average y_hat[t] = mean(y[t-6 .. t-1])     mean of the input window
    """
    flat = series.reshape(-1)
    window = np.stack(
        [flat[target_index - k] for k in range(1, SEQUENCE_LENGTH + 1)], axis=1
    )
    return {
        "persistence": flat[target_index - 1].reshape(-1, 1),
        "seasonal_naive": flat[target_index - STEPS_PER_DAY].reshape(-1, 1),
        "moving_average": window.mean(axis=1).reshape(-1, 1),
    }


class LSTMForecaster(nn.Module):
    """Minimal single-step forecaster.

    Only the layers that are actually used in forward() are defined; the unused
    activation attributes of the earlier notebook implementation are gone.
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def evaluate(model, loader, criterion, scaler):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_X)
            total_loss += criterion(out, batch_y).item() * batch_X.size(0)
            preds.append(out.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    preds = scaler.inverse_transform(np.vstack(preds))
    targets = scaler.inverse_transform(np.vstack(targets))
    return loss, compute_metrics(targets, preds), preds, targets


def run_experiment(name, hidden_size, num_layers, partitions, series):
    set_seed(SEED)

    train_raw = series[slice(*partitions["train"])]
    scaler = StandardScaler()
    scaler.fit(train_raw)

    loaders, target_index = {}, {}
    for part, (start, stop) in partitions.items():
        scaled = scaler.transform(series[start:stop])
        X, y, idx = create_sequences(scaled, offset=start)
        target_index[part] = idx
        loaders[part] = DataLoader(
            TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False
        )

    model = LSTMForecaster(INPUT_SIZE, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] hidden={hidden_size} layers={num_layers} params={n_params}")
    print(
        f"[{name}] sequences: "
        + " ".join(f"{k}={len(v.dataset)}" for k, v in loaders.items())
    )

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    stale, stopped_early = 0, False

    for epoch in range(MAX_EPOCHS):
        model.train()
        running = 0.0
        for batch_X, batch_y in loaders["train"]:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_X.size(0)

        train_loss = running / len(loaders["train"].dataset)
        val_loss, val_metrics, _, _ = evaluate(
            model, loaders["val"], criterion, scaler
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_metrics["rmse"])

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
            marker = " *"
        else:
            stale += 1

        print(
            f"[{name}] epoch {epoch + 1:3d}/{MAX_EPOCHS} - "
            f"train_loss {train_loss:.6f} - val_loss {val_loss:.6f} - "
            f"val_rmse {val_metrics['rmse']:.4f}{marker}"
        )

        if stale >= PATIENCE:
            stopped_early = True
            print(
                f"[{name}] early stopping at epoch {epoch + 1}; "
                f"restoring weights from epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)

    _, val_metrics, _, _ = evaluate(model, loaders["val"], criterion, scaler)
    _, test_metrics, test_pred, test_true = evaluate(
        model, loaders["test"], criterion, scaler
    )

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / f"{name}.pth")

    record = {
        "experiment_name": name,
        "task": {
            "description": (
                "predict the next 10-minute consumption value from the "
                "previous hour of observed values"
            ),
            "sequence_length": SEQUENCE_LENGTH,
            "forecast_horizon": FORECAST_HORIZON,
            "stride": 1,
            "prediction_mode": "sliding window, one step ahead, teacher forced",
        },
        "hyperparameters": {
            "seed": SEED,
            "input_size": INPUT_SIZE,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "trainable_parameters": n_params,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "scaling": "StandardScaler fitted on the training partition only",
            "scaler_mean": float(scaler.mean_[0]),
            "scaler_scale": float(scaler.scale_[0]),
        },
        "results": {
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        "history": history,
    }

    return record, test_pred, test_true, target_index["test"]


def plot_loss_curves(records, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(records), figsize=(6 * len(records), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (label, record) in zip(axes, records):
        history = record["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], marker="o", markersize=3, label="Train")
        ax.plot(epochs, history["val_loss"], marker="s", markersize=3, label="Validation")
        ax.axvline(
            record["results"]["best_epoch"],
            color="green",
            linestyle=":",
            label=f"Best epoch ({record['results']['best_epoch']})",
        )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (standardized scale)")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _format_time_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_test_period(timestamps, y_true, y_pred, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(timestamps, y_true, linewidth=0.6, label="Actual")
    ax.plot(timestamps, y_pred, linewidth=0.6, alpha=0.8, label="Predicted")
    ax.set_title("Test period - actual vs predicted (one step ahead, 10 minutes)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    _format_time_axis(ax)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_week(timestamps, y_true, y_pred, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    span = 7 * STEPS_PER_DAY
    ts, actual, pred = timestamps[:span], y_true[:span], y_pred[:span]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ts, actual, linewidth=1.1, label="Actual")
    ax.plot(ts, pred, linewidth=1.1, alpha=0.85, label="Predicted")
    start, end = ts[0].strftime("%Y-%m-%d"), ts[-1].strftime("%Y-%m-%d")
    ax.set_title(f"First week of the test period - actual vs predicted ({start} to {end})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption")
    _format_time_axis(ax)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_errors(timestamps, y_true, y_pred, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    residual = (y_pred - y_true).reshape(-1)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2, 1]}
    )

    top.plot(timestamps, residual, linewidth=0.5)
    top.axhline(0, color="black", linewidth=0.8)
    top.set_title("Prediction error over the test period (predicted - actual)")
    top.set_xlabel("Date")
    top.set_ylabel("Error")
    _format_time_axis(top)
    top.grid(True, linestyle="--", alpha=0.5)

    bottom.hist(residual, bins=80)
    bottom.axvline(0, color="black", linewidth=0.8)
    bottom.set_title("Distribution of prediction errors")
    bottom.set_xlabel("Error")
    bottom.set_ylabel("Count")
    bottom.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(y_true, y_pred, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    actual = y_true.reshape(-1)
    pred = y_pred.reshape(-1)
    low = float(min(actual.min(), pred.min()))
    high = float(max(actual.max(), pred.max()))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(actual, pred, s=2, alpha=0.25, edgecolors="none")
    ax.plot([low, high], [low, high], color="red", linestyle="--", linewidth=1,
            label="y = x")
    ax.set_title("Test period - predicted vs actual")
    ax.set_xlabel("Actual consumption")
    ax.set_ylabel("Predicted consumption")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def append_records(records):
    existing = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass
    names = {r["experiment_name"] for r in records}
    existing = [e for e in existing if e.get("experiment_name") not in names]
    existing.extend(records)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def format_metrics(name, m):
    return (
        f"  {name:<34} MSE {m['mse']:.4f}  RMSE {m['rmse']:.4f}  "
        f"MAE {m['mae']:.4f}  MAPE {m['mape']:.2f}%  R2 {m['r2']:.4f}"
    )


def main():
    series, index = load_series()
    partitions = split_bounds(len(series))

    print("=" * 100)
    print("Task: predict the next 10-minute value from the previous hour (6 steps)")
    print(f"Series: {len(series)} rows, {index[0]} to {index[-1]}, stride {SERIES_FREQ}")
    for part, (start, stop) in partitions.items():
        n_seq = stop - start - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
        print(
            f"  {part:<6} raw [{start}, {stop})  length {stop - start:>6}  "
            f"sequences {n_seq:>6}  {index[start]} .. {index[stop - 1]}"
        )
    print("=" * 100)

    # Reference models, evaluated on exactly the targets the network is scored on.
    reference_rows = {}
    for part in ("val", "test"):
        start, stop = partitions[part]
        _, y, target_index = create_sequences(series[start:stop], offset=start)
        y_true = y.numpy()
        for ref_name, pred in reference_predictions(series, target_index).items():
            reference_rows.setdefault(ref_name, {})[part] = compute_metrics(y_true, pred)

    print("\nReference models (no training involved) - test partition")
    for ref_name, parts in reference_rows.items():
        print(format_metrics(ref_name, parts["test"]))

    configs = [
        ("lstm_h32_l1", 32, 1, "Revised model - hidden 32, 1 layer"),
        ("lstm_h64_l2", 64, 2, "Capacity check - hidden 64, 2 layers"),
    ]

    records, outputs = [], {}
    for name, hidden, layers, label in configs:
        record, pred, true, target_index = run_experiment(
            name, hidden, layers, partitions, series
        )
        records.append((label, record))
        outputs[name] = (pred, true, index[target_index])

    main_name = configs[0][0]
    pred, true, timestamps = outputs[main_name]

    plot_loss_curves(records, "revised_loss_curves.png")
    plot_test_period(timestamps, true, pred, "revised_test_actual_vs_predicted.png")
    plot_week(timestamps, true, pred, "revised_test_week_zoom.png")
    plot_errors(timestamps, true, pred, "revised_test_errors.png")
    plot_scatter(true, pred, "revised_test_scatter.png")

    reference_records = [
        {
            "experiment_name": f"reference_{ref_name}",
            "task": {
                "sequence_length": SEQUENCE_LENGTH,
                "forecast_horizon": FORECAST_HORIZON,
            },
            "results": {
                "val_metrics": parts["val"],
                "test_metrics": parts["test"],
            },
        }
        for ref_name, parts in reference_rows.items()
    ]
    append_records(reference_records + [r for _, r in records])

    print("\n" + "=" * 100)
    print("Test partition summary (original consumption unit)")
    print("=" * 100)
    for ref_name, parts in reference_rows.items():
        print(format_metrics(f"reference: {ref_name}", parts["test"]))
    for label, record in records:
        print(format_metrics(label, record["results"]["test_metrics"]))

    print("\nValidation partition summary (used for model selection)")
    print("=" * 100)
    for ref_name, parts in reference_rows.items():
        print(format_metrics(f"reference: {ref_name}", parts["val"]))
    for label, record in records:
        print(format_metrics(label, record["results"]["val_metrics"]))

    print("\n" + "=" * 100)
    for label, record in records:
        r = record["results"]
        h = record["hyperparameters"]
        print(
            f"  {label}: {h['trainable_parameters']} params, ran {r['epochs_run']} "
            f"epochs, best epoch {r['best_epoch']}, early stopped: {r['stopped_early']}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
