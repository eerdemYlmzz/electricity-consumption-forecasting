"""
Phase 1 - plain baseline deep learning models, Panama national load.

Per the advisor's spec: predict the next hour from the past L hours, for
L in {48,72,96,120,144,168}, using all 14 features. No XAI-based feature
selection and no hyperparameter optimization at this stage -- that is
Phases 2-4. No persistent validation set: chronological train/test split
only, fixed epoch budget, no early stopping.

A first run at 30 epochs showed textbook overfitting on every architecture
(train loss falling monotonically while test RMSE got worse from epoch ~5
onward). Dropout + weight decay were tried and barely moved the epoch-30
numbers -- and tuning their strength by watching test RMSE is the same
class of leakage as tuning the epoch count itself. Reverted: plain
models, no regularization, fixed 30-epoch budget, whatever it produces.

Architectures: MLP, Simple RNN, LSTM, GRU, LSTM+Attention, BiGRU, 1D-CNN,
TCN (8) x 6 window lengths = 48 runs. Results are written to
results/phase1_baseline_results.csv.

MLP/CNN1D/TCN use GELU, not ReLU: same single-projection, pointwise,
zero-added-parameter activation, just a smoother curve at 0. SwiGLU was
considered and rejected for this phase -- it needs two linear projections
per block instead of one, which would roughly double those layers'
parameter count. Given the overfitting problem above, more capacity is
the wrong direction; a SwiGLU "capacity check" belongs later, once
Phase 4 has a real mechanism (real val split, HPO) to justify it against.

Metrics: MAE, RMSE, R^2 (advisor's minimum) plus MAPE -- safe to add here
since national_demand_mw never approaches zero (unlike e.g. solar
irradiance, which is exactly zero at night and blows MAPE up).
"""
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_PATH = "data/processed/panama_load.csv"
RESULTS_PATH = "results/phase1_baseline_results.csv"
TARGET = "national_demand_mw"
WINDOW_LENGTHS = [48, 72, 96, 120, 144, 168]
HORIZON = 1
TRAIN_FRACTION = 0.8
EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    feature_cols = [c for c in df.columns if c != TARGET]
    return df, feature_cols


def chronological_split(df, train_fraction):
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def create_sequences(features, target, seq_len, horizon):
    """Window i = features[i : i+seq_len] (seq_len past hours) predicts
    target[i + seq_len + horizon - 1] (the horizon-th hour after the window
    ends; horizon=1 -> the hour immediately following the window).
    Both partitions (train, test) are windowed independently -- a test
    window never reaches back into the train partition, so a handful of
    hours right after the split (up to seq_len-1) are not used as targets.
    This costs <2% of the test set at the longest window and keeps the
    boundary trivially auditable.
    """
    n = len(features)
    num_samples = n - seq_len - horizon + 1
    if num_samples <= 0:
        raise ValueError(f"seq_len={seq_len} + horizon={horizon} exceeds partition length {n}")
    num_features = features.shape[1]
    X = np.empty((num_samples, seq_len, num_features), dtype=np.float32)
    y = np.empty(num_samples, dtype=np.float32)
    for i in range(num_samples):
        X[i] = features[i : i + seq_len]
        y[i] = target[i + seq_len + horizon - 1]
    return X, y


def _test_create_sequences():
    """Explicit boundary check, per the advisor's instruction to verify
    window start/end indices -- catches the class of off-by-one bug found
    in the single-dataset project's '# 60 stride' comment.
    """
    features = np.arange(10).reshape(-1, 1).astype(np.float32)
    target = np.arange(10).astype(np.float32) * 10
    X, y = create_sequences(features, target, seq_len=3, horizon=1)
    assert X.shape == (7, 3, 1), X.shape
    assert y.shape == (7,), y.shape
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2])
    assert y[0] == 30  # target right after the window: index 3
    np.testing.assert_array_equal(X[-1].ravel(), [6, 7, 8])
    assert y[-1] == 90  # index 9, the last valid target

    X2, y2 = create_sequences(features, target, seq_len=3, horizon=2)
    assert X2.shape == (6, 3, 1), X2.shape
    np.testing.assert_array_equal(X2[0].ravel(), [0, 1, 2])
    assert y2[0] == 40  # horizon=2 -> index 3+2-1=4
    print("create_sequences boundary test passed")


class MLPForecaster(nn.Module):
    def __init__(self, seq_len, num_features, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


class SimpleRNNForecaster(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMAttentionForecaster(nn.Module):
    """LSTM over the window, then additive attention pooling over every
    timestep's hidden state (instead of using only the final one)."""

    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.attn_score = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        scores = self.attn_score(out)  # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        context = (weights * out).sum(dim=1)  # (batch, hidden)
        return self.fc(context).squeeze(-1)


class BiGRUForecaster(nn.Module):
    """Bidirectional GRU. No leakage concern: both directions only ever see
    the past window itself, never the target -- the backward pass just
    reads that same window right-to-left, it doesn't look past the window
    end. Final representation is the forward pass's last hidden state
    concatenated with the backward pass's last hidden state (h_n), not
    out[:, -1, :] -- the latter would pair the forward-final state with the
    backward direction's state after seeing only one input, which wastes
    the backward pass."""

    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: (2, batch, hidden_size)
        combined = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(combined).squeeze(-1)


class CNN1DForecaster(nn.Module):
    """Two Conv1d layers over the time axis + global average pooling."""

    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, num_features, seq_len)
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def _tcn_levels_for(seq_len, kernel_size):
    """Smallest number of dilated (2^i) causal conv levels whose combined
    receptive field covers the whole window. Without this, a fixed-depth
    TCN would only ever look at a fixed-size chunk of the window regardless
    of the nominal window length, which would silently break the 6-window
    comparison (TCN would not actually be using the longer windows)."""
    levels, receptive_field = 1, kernel_size
    while receptive_field < seq_len:
        levels += 1
        receptive_field = 1 + (kernel_size - 1) * (2**levels - 1)
    return levels


class TCNForecaster(nn.Module):
    """Stack of causal, dilated Conv1d layers. Causal = each layer's output
    at position t depends only on inputs at positions <= t (implemented by
    padding on the left side only, via padding=(k-1)*dilation then trimming
    the trailing (k-1)*dilation positions the conv would otherwise produce
    from the right-padded side)."""

    def __init__(self, num_features, hidden_size, seq_len, kernel_size=3):
        super().__init__()
        num_levels = _tcn_levels_for(seq_len, kernel_size)
        layers = []
        in_channels = num_features
        for i in range(num_levels):
            dilation = 2**i
            pad = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size, padding=pad, dilation=dilation))
            in_channels = hidden_size
        self.convs = nn.ModuleList(layers)
        self.pads = [(kernel_size - 1) * (2**i) for i in range(num_levels)]
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, num_features, seq_len)
        for conv, pad in zip(self.convs, self.pads):
            x = torch.nn.functional.gelu(conv(x)[:, :, :-pad])  # trim right-side padding to stay causal
        return self.fc(x[:, :, -1]).squeeze(-1)


def build_model(name, seq_len, num_features, hidden_size):
    if name == "MLP":
        return MLPForecaster(seq_len, num_features, hidden_size)
    if name == "SimpleRNN":
        return SimpleRNNForecaster(num_features, hidden_size)
    if name == "LSTM":
        return LSTMForecaster(num_features, hidden_size)
    if name == "GRU":
        return GRUForecaster(num_features, hidden_size)
    if name == "LSTM_Attention":
        return LSTMAttentionForecaster(num_features, hidden_size)
    if name == "BiGRU":
        return BiGRUForecaster(num_features, hidden_size)
    if name == "CNN1D":
        return CNN1DForecaster(num_features, hidden_size)
    if name == "TCN":
        return TCNForecaster(num_features, hidden_size, seq_len)
    raise ValueError(f"unknown model name: {name}")


def train_model(model, X_train, y_train, epochs, batch_size, lr):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_train_t = torch.from_numpy(X_train).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    n = len(X_train_t)

    history = []
    for epoch in range(epochs):
        model.train()
        # shuffling minibatch order is standard SGD practice and does not
        # violate the "never shuffle" rule -- that rule is about the
        # train/test split itself; each windowed sample here is already a
        # self-contained (past, future) pair, so reordering samples during
        # training leaks nothing.
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history.append(epoch_loss / n)
    return history


def evaluate_model(model, X_test, y_test, target_scaler):
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(DEVICE)
        pred_scaled = model(X_test_t).cpu().numpy()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred) ** 0.5
    r2 = r2_score(true, pred)
    # safe here: national_demand_mw never approaches zero (unlike e.g. solar
    # irradiance, which is exactly zero at night and would blow MAPE up)
    mape = mean_absolute_percentage_error(true, pred) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}, pred, true


def run_all():
    _test_create_sequences()
    set_seed(SEED)

    df, feature_cols = load_data(DATA_PATH)
    train_df, test_df = chronological_split(df, TRAIN_FRACTION)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")

    feature_scaler = StandardScaler().fit(train_df[feature_cols].values)
    target_scaler = StandardScaler().fit(train_df[[TARGET]].values)

    train_features = feature_scaler.transform(train_df[feature_cols].values)
    train_target = target_scaler.transform(train_df[[TARGET]].values).ravel()
    test_features = feature_scaler.transform(test_df[feature_cols].values)
    test_target = target_scaler.transform(test_df[[TARGET]].values).ravel()

    # The window's inputs include the target's own recent history (its lag)
    # alongside the 14 exogenous features -- not leakage: at prediction time
    # t, values at t-seq_len..t-1 have already happened, exactly like the
    # weather features for those same hours. This is standard autoregressive
    # input, same as the single-dataset project's univariate LSTM did with
    # nothing but the target's own window. Adds 1 input dim (14 -> 15).
    train_inputs = np.concatenate([train_features, train_target.reshape(-1, 1)], axis=1)
    test_inputs = np.concatenate([test_features, test_target.reshape(-1, 1)], axis=1)
    num_inputs = len(feature_cols) + 1

    model_names = ["MLP", "SimpleRNN", "LSTM", "GRU", "LSTM_Attention", "BiGRU", "CNN1D", "TCN"]
    records = []

    for seq_len in WINDOW_LENGTHS:
        X_train, y_train = create_sequences(train_inputs, train_target, seq_len, HORIZON)
        X_test, y_test = create_sequences(test_inputs, test_target, seq_len, HORIZON)

        for name in model_names:
            set_seed(SEED)  # identical init/shuffle across models and window lengths
            model = build_model(name, seq_len, num_inputs, HIDDEN_SIZE)
            n_params = sum(p.numel() for p in model.parameters())

            start = time.time()
            history = train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE)
            metrics, _, _ = evaluate_model(model, X_test, y_test, target_scaler)
            elapsed = time.time() - start

            print(
                f"window={seq_len:>3}h  model={name:<15} "
                f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.4f} "
                f"MAPE={metrics['mape']:.2f}% params={n_params} time={elapsed:.1f}s"
            )
            records.append({
                "window_length_h": seq_len,
                "model": name,
                "params": n_params,
                "epochs": EPOCHS,
                "final_train_loss": history[-1],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
                "train_seconds": elapsed,
            })

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nsaved: {RESULTS_PATH}  ({len(results)} rows)")
    return results


if __name__ == "__main__":
    run_all()
