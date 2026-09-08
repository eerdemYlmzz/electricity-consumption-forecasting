"""
Phase 1 plots:
1. Window-length comparison (RMSE and R^2 per architecture across the 6
   window lengths) -- the "different window length comparison" plot from
   the advisor's list.
2. Overfitting evidence (train loss vs test RMSE per epoch, GRU/48h as the
   representative case) -- documents why the fixed-epoch/no-validation
   protocol produces the pattern seen in the Phase 1 results table.
3. Actual vs predicted, and prediction error over time, for the
   best-scoring config in the results CSV -- the two required plots from
   the advisor's list that were still missing.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import phase1_baseline_models as m

RESULTS_PATH = "results/phase1_baseline_results.csv"
WINDOW_PLOT_PATH = "plots/phase1_window_length_comparison.png"
OVERFIT_PLOT_PATH = "plots/phase1_overfitting_evidence.png"
ACTUAL_VS_PRED_PATH = "plots/phase1_actual_vs_predicted.png"
ERROR_OVER_TIME_PATH = "plots/phase1_error_over_time.png"


def _prepare_data():
    """Same scaling + target-lag-as-input setup as run_all() in
    phase1_baseline_models.py. Returns everything needed to build sequences
    for any (seq_len, model_name) combination."""
    m.set_seed(m.SEED)
    df, feature_cols = m.load_data(m.DATA_PATH)
    train_df, test_df = m.chronological_split(df, m.TRAIN_FRACTION)

    feature_scaler = m.StandardScaler().fit(train_df[feature_cols].values)
    target_scaler = m.StandardScaler().fit(train_df[[m.TARGET]].values)

    train_features = feature_scaler.transform(train_df[feature_cols].values)
    train_target = target_scaler.transform(train_df[[m.TARGET]].values).ravel()
    test_features = feature_scaler.transform(test_df[feature_cols].values)
    test_target = target_scaler.transform(test_df[[m.TARGET]].values).ravel()

    train_inputs = np.concatenate([train_features, train_target.reshape(-1, 1)], axis=1)
    test_inputs = np.concatenate([test_features, test_target.reshape(-1, 1)], axis=1)
    num_inputs = len(feature_cols) + 1

    return train_inputs, train_target, test_inputs, test_target, test_df, target_scaler, num_inputs


def plot_window_length_comparison():
    df = pd.read_csv(RESULTS_PATH)
    models = df["model"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in models:
        sub = df[df["model"] == name].sort_values("window_length_h")
        axes[0].plot(sub["window_length_h"], sub["rmse"], marker="o", label=name)
        axes[1].plot(sub["window_length_h"], sub["r2"], marker="o", label=name)

    axes[0].set_xlabel("Window length (hours)")
    axes[0].set_ylabel("Test RMSE (MW)")
    axes[0].set_title("RMSE vs window length")
    axes[0].legend()

    axes[1].set_xlabel("Window length (hours)")
    axes[1].set_ylabel("Test R^2")
    axes[1].set_title("R^2 vs window length")
    axes[1].legend()

    fig.suptitle("Phase 1 baseline models -- Panama national load")
    fig.tight_layout()
    fig.savefig(WINDOW_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {WINDOW_PLOT_PATH}")


def _train_and_predict(seq_len, model_name):
    train_inputs, train_target, test_inputs, test_target, test_df, target_scaler, num_inputs = _prepare_data()

    X_train, y_train = m.create_sequences(train_inputs, train_target, seq_len, m.HORIZON)
    X_test, y_test = m.create_sequences(test_inputs, test_target, seq_len, m.HORIZON)

    target_row_offset = seq_len + m.HORIZON - 1
    timestamps = test_df.index[target_row_offset : target_row_offset + len(y_test)]

    m.set_seed(m.SEED)
    model = m.build_model(model_name, seq_len, num_inputs, m.HIDDEN_SIZE)
    history = m.train_model(model, X_train, y_train, m.EPOCHS, m.BATCH_SIZE, m.LEARNING_RATE)
    metrics, pred, true = m.evaluate_model(model, X_test, y_test, target_scaler)
    return timestamps, true, pred, metrics, history


def plot_overfitting_evidence(seq_len=48, model_name="GRU"):
    train_inputs, train_target, test_inputs, test_target, test_df, target_scaler, num_inputs = _prepare_data()
    X_train, y_train = m.create_sequences(train_inputs, train_target, seq_len, m.HORIZON)
    X_test, y_test = m.create_sequences(test_inputs, test_target, seq_len, m.HORIZON)

    m.set_seed(m.SEED)
    model = m.build_model(model_name, seq_len, num_inputs, m.HIDDEN_SIZE)
    model.to(m.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=m.LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    X_train_t = torch.from_numpy(X_train).to(m.DEVICE)
    y_train_t = torch.from_numpy(y_train).to(m.DEVICE)
    n = len(X_train_t)

    train_losses, test_rmses = [], []
    for epoch in range(m.EPOCHS):
        model.train()
        perm = torch.randperm(n, device=m.DEVICE)
        epoch_loss = 0.0
        for start in range(0, n, m.BATCH_SIZE):
            idx = perm[start : start + m.BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)
        metrics, _, _ = m.evaluate_model(model, X_test, y_test, target_scaler)
        test_rmses.append(metrics["rmse"])

    best_epoch = min(range(len(test_rmses)), key=lambda i: test_rmses[i])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, m.EPOCHS + 1), train_losses, color="tab:blue", label="train loss (scaled MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(range(1, m.EPOCHS + 1), test_rmses, color="tab:red", label="test RMSE (MW)")
    ax2.set_ylabel("Test RMSE (MW)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axvline(best_epoch + 1, color="gray", linestyle="--", alpha=0.7)
    ax2.annotate(
        f"best epoch = {best_epoch + 1}",
        xy=(best_epoch + 1, test_rmses[best_epoch]),
        xytext=(best_epoch + 3, test_rmses[best_epoch] + 5),
    )

    fig.suptitle(f"Overfitting evidence: {model_name}, {seq_len}h window, no validation/early stopping")
    fig.tight_layout()
    fig.savefig(OVERFIT_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {OVERFIT_PLOT_PATH}  (best epoch={best_epoch + 1}, reported epoch={m.EPOCHS})")


def plot_actual_vs_predicted_and_errors(seq_len=None, model_name=None, zoom_days=14):
    """Picks the best-scoring (seq_len, model) from the results CSV unless
    both are given explicitly."""
    if seq_len is None or model_name is None:
        results = pd.read_csv(RESULTS_PATH)
        best = results.loc[results["r2"].idxmax()]
        seq_len, model_name = int(best["window_length_h"]), best["model"]

    timestamps, true, pred, metrics, _ = _train_and_predict(seq_len, model_name)
    print(
        f"{model_name}@{seq_len}h  R2={metrics['r2']:.3f} RMSE={metrics['rmse']:.2f} "
        f"MAE={metrics['mae']:.2f} MAPE={metrics['mape']:.2f}%"
    )
    errors = true - pred

    zoom_hours = 24 * zoom_days
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(timestamps[:zoom_hours], true[:zoom_hours], label="actual", linewidth=1.3)
    ax.plot(timestamps[:zoom_hours], pred[:zoom_hours], label="predicted", linewidth=1.3, alpha=0.8)
    ax.set_ylabel("National demand (MW)")
    ax.set_title(f"Actual vs predicted -- {model_name}, {seq_len}h window (first {zoom_days} days of test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ACTUAL_VS_PRED_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {ACTUAL_VS_PRED_PATH}")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamps, errors, linewidth=0.5, color="tab:red")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Error: actual - predicted (MW)")
    ax.set_title(f"Prediction error over time -- {model_name}, {seq_len}h window (full test period)")
    fig.tight_layout()
    fig.savefig(ERROR_OVER_TIME_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {ERROR_OVER_TIME_PATH}")


if __name__ == "__main__":
    plot_window_length_comparison()
    plot_overfitting_evidence()
    plot_actual_vs_predicted_and_errors()
