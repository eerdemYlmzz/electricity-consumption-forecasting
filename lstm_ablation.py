"""
Ablation study for the LSTM electricity consumption baseline.

Run A : leakage-free chronological 70/10/20 split, per-epoch validation loss,
        early stopping on validation loss, no scaling.
Run B : identical to Run A, with standardization fitted on the training
        partition only.

Both runs share the same seed, architecture, optimizer, epoch budget and early
stopping policy, so any difference between them is attributable to the scaling
step alone.

A persistence (naive) baseline is evaluated first and acts as the reference
floor: y_hat[t] = y[t-1].

All reported metrics are computed in the original kWh units. For the scaled run
the predictions and targets are inverse-transformed before the metrics are
computed, so Run A and Run B remain directly comparable.
"""

import copy
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DATA_PATH = Path("data/data_processed.csv")
PLOTS_DIR = Path("plots")
MODEL_DIR = Path("model")
LOG_FILE = Path("hyperparameter_logs.json")

SEQUENCE_LENGTH = 60
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_STACKED_LAYERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 8
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
DPI = 300

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_series():
    data = pd.read_csv(DATA_PATH)
    return data["consumption"].values.astype(np.float32).reshape(-1, 1)


def split_series(series):
    """Chronological split of the raw series, before any windowing.

    Splitting the raw series first and windowing each partition independently
    is what keeps the validation and test partitions leakage-free: no input
    window can span a partition boundary.
    """
    n = len(series)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return (
        series[:n_train],
        series[n_train : n_train + n_val],
        series[n_train + n_val :],
    )


def create_sequences(series, seq_length):
    xs, ys = [], []
    for i in range(len(series) - seq_length):
        xs.append(series[i : i + seq_length])
        ys.append(series[i + seq_length])
    return (
        torch.tensor(np.array(xs), dtype=torch.float32),
        torch.tensor(np.array(ys), dtype=torch.float32),
    )


def compute_metrics(y_true, y_pred):
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


def persistence_metrics(X, y):
    """Repeat the last observed value of the input window as the prediction."""
    return compute_metrics(y.numpy(), X[:, -1, :].numpy())


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_stacked_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_stacked_layers, batch_size, self.hidden_size, device=x.device
        )
        c0 = torch.zeros(
            self.num_stacked_layers, batch_size, self.hidden_size, device=x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def evaluate(model, loader, criterion, mean, scale):
    """Return (loss in training units, metric dict in original kWh units)."""
    model.eval()
    total_loss = 0.0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X)
            total_loss += criterion(preds, batch_y).item() * batch_X.size(0)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(batch_y.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    preds_original = np.vstack(preds_all) * scale + mean
    targets_original = np.vstack(targets_all) * scale + mean
    return loss, compute_metrics(targets_original, preds_original)


def run_experiment(experiment_name, apply_scaling):
    set_seed(SEED)

    series = load_series()
    train_raw, val_raw, test_raw = split_series(series)

    if apply_scaling:
        mean = float(train_raw.mean())
        scale = float(train_raw.std())
        train_raw = (train_raw - mean) / scale
        val_raw = (val_raw - mean) / scale
        test_raw = (test_raw - mean) / scale
    else:
        mean, scale = 0.0, 1.0

    X_train, y_train = create_sequences(train_raw, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_raw, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_raw, SEQUENCE_LENGTH)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_STACKED_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    print(f"\n[{experiment_name}] scaling={apply_scaling} device={device}")
    print(
        f"[{experiment_name}] sequences: "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, mean, scale)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_metrics["rmse"])

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1

        print(
            f"[{experiment_name}] epoch {epoch + 1:3d}/{MAX_EPOCHS} - "
            f"train_loss {train_loss:.6f} - val_loss {val_loss:.6f} - "
            f"val_rmse_kwh {val_metrics['rmse']:.4f}{marker}"
        )

        if epochs_without_improvement >= PATIENCE:
            stopped_early = True
            print(
                f"[{experiment_name}] early stopping at epoch {epoch + 1}; "
                f"no improvement for {PATIENCE} epochs. "
                f"Restoring weights from epoch {best_epoch}."
            )
            break

    model.load_state_dict(best_state)

    val_loss, val_metrics = evaluate(model, val_loader, criterion, mean, scale)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, mean, scale)

    record = {
        "experiment_name": experiment_name,
        "hyperparameters": {
            "seed": SEED,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_stacked_layers": NUM_STACKED_LAYERS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "scaling": "standardization_train_only" if apply_scaling else "none",
            "scaler_mean": mean,
            "scaler_std": scale,
        },
        "results": {
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": val_loss,
            "val_metrics": val_metrics,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        },
        "history": history,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / f"{experiment_name}.pth")

    return record


def plot_loss_curve(record, filename, title):
    history = record["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = record["results"]["best_epoch"]

    PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    plt.plot(
        epochs, history["val_loss"], label="Validation Loss", marker="s", markersize=4
    )
    plt.axvline(
        best_epoch,
        color="green",
        linestyle=":",
        label=f"Best epoch ({best_epoch})",
    )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_comparison(record_a, record_b, naive_val_rmse, filename):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    for record, label, marker in (
        (record_a, "Run A - no scaling", "o"),
        (record_b, "Run B - standardized", "s"),
    ):
        values = record["history"]["val_rmse"]
        plt.plot(
            range(1, len(values) + 1),
            values,
            label=label,
            marker=marker,
            markersize=4,
        )
    plt.axhline(
        naive_val_rmse,
        color="red",
        linestyle="--",
        label=f"Persistence baseline ({naive_val_rmse:.4f})",
    )
    plt.title("Validation RMSE per Epoch (original kWh units)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (kWh)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def append_records(records):
    existing = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass

    names = {record["experiment_name"] for record in records}
    existing = [e for e in existing if e.get("experiment_name") not in names]
    existing.extend(records)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)


def format_metrics(name, metrics):
    return (
        f"  {name:<28} RMSE {metrics['rmse']:.4f}  MAE {metrics['mae']:.4f}  "
        f"MAPE {metrics['mape']:.2f}%  R2 {metrics['r2']:.4f}"
    )


def main():
    series = load_series()
    _, val_raw, test_raw = split_series(series)

    X_val_raw, y_val_raw = create_sequences(val_raw, SEQUENCE_LENGTH)
    X_test_raw, y_test_raw = create_sequences(test_raw, SEQUENCE_LENGTH)

    naive_val = persistence_metrics(X_val_raw, y_val_raw)
    naive_test = persistence_metrics(X_test_raw, y_test_raw)

    print("=" * 88)
    print("Persistence (naive) baseline - no training involved")
    print(format_metrics("validation", naive_val))
    print(format_metrics("test", naive_test))
    print("=" * 88)

    record_a = run_experiment("lstm_v4_noscale_earlystop", apply_scaling=False)
    record_b = run_experiment("lstm_v5_scaled_earlystop", apply_scaling=True)

    plot_loss_curve(
        record_a,
        "run_a_loss_curve.png",
        "Run A - Train vs Validation Loss (no scaling, raw kWh)",
    )
    plot_loss_curve(
        record_b,
        "run_b_loss_curve.png",
        "Run B - Train vs Validation Loss (standardized)",
    )
    plot_comparison(record_a, record_b, naive_val["rmse"], "ablation_val_rmse.png")

    naive_record = {
        "experiment_name": "persistence_baseline",
        "hyperparameters": {
            "sequence_length": SEQUENCE_LENGTH,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
        },
        "results": {"val_metrics": naive_val, "test_metrics": naive_test},
    }
    append_records([naive_record, record_a, record_b])

    print("\n" + "=" * 88)
    print("Test set summary (original kWh units)")
    print("=" * 88)
    print(format_metrics("persistence baseline", naive_test))
    print(format_metrics("Run A (no scaling)", record_a["results"]["test_metrics"]))
    print(format_metrics("Run B (standardized)", record_b["results"]["test_metrics"]))
    print("=" * 88)
    for record in (record_a, record_b):
        results = record["results"]
        print(
            f"  {record['experiment_name']}: ran {results['epochs_run']} epochs, "
            f"best epoch {results['best_epoch']}, "
            f"early stopped: {results['stopped_early']}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
