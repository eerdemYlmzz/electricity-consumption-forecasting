"""
Phase 2 - SHAP feature ranking, BiGRU @ 72h (best Phase 1 config, R^2=0.988).

Only SHAP is used (LIME dropped -- the advisor's spec requires at least
one of SHAP/LIME, not both; LIME's tabular explainer perturbs "features"
independently, which is a poor fit for 72 highly autocorrelated lag
steps per feature and was judged not worth its far higher cost here).

Uses shap.GradientExplainer (gradient-based, works directly on the
trained PyTorch model without needing a differentiable surrogate or
thousands of perturbed forward passes per sample -- LIME's approach).
Explains a fixed, seeded sample of 200 test windows against a background
of 100 training windows (both from SEED=42 for reproducibility).

Critical requirement from the advisor's mail: report feature-level
importance, not per-lag importance. Every window has 15 input columns
(14 exogenous features + the target's own lag) x 72 timesteps -- SHAP
gives one value per (timestep, column) cell. We sum |SHAP value| over
the 72 timesteps for each column, then average over the 200 explained
windows, collapsing 72*15=1080 numbers down to 15 -- one score per
feature, no lag columns ever shown.

Reuses phase1_baseline_models.py's load_data/chronological_split/
create_sequences/build_model unchanged, so the preprocessing and input
column order are guaranteed identical to what the checkpoint was
trained on -- reordering or re-deriving any of this independently would
silently corrupt the loaded weights (wrong-but-no-error, not a crash).
"""
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import phase1_baseline_models as m

MODEL_NAME = "BiGRU"
SEQ_LEN = 72
N_EXPLAIN = 200
N_BACKGROUND = 100
CHECKPOINT_PATH = f"results/checkpoints/{MODEL_NAME}_{SEQ_LEN}h.pt"
RESULTS_PATH = "results/phase2_shap_feature_importance.csv"
PLOT_PATH = "plots/phase2_shap_feature_importance.png"


class _UnsqueezeOutput(nn.Module):
    """shap.GradientExplainer expects a (batch, num_outputs) model output;
    our forecasters return (batch,) via .squeeze(-1). Wrap rather than
    touch the trained architecture itself."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x).unsqueeze(-1)


def build_inputs():
    df, feature_cols = m.load_data(m.DATA_PATH)
    train_df, test_df = m.chronological_split(df, m.TRAIN_FRACTION)

    feature_scaler = m.StandardScaler().fit(train_df[feature_cols].values)
    target_scaler = m.StandardScaler().fit(train_df[[m.TARGET]].values)

    train_features = feature_scaler.transform(train_df[feature_cols].values)
    train_target = target_scaler.transform(train_df[[m.TARGET]].values).ravel()
    test_features = feature_scaler.transform(test_df[feature_cols].values)
    test_target = target_scaler.transform(test_df[[m.TARGET]].values).ravel()

    # identical construction to run_all() -- same column order, same dtype
    train_inputs = np.concatenate([train_features, train_target.reshape(-1, 1)], axis=1)
    test_inputs = np.concatenate([test_features, test_target.reshape(-1, 1)], axis=1)
    num_inputs = len(feature_cols) + 1
    feature_names = feature_cols + [f"{m.TARGET} (lag)"]

    return train_inputs, train_target, test_inputs, test_target, num_inputs, feature_names


def load_trained_model(num_inputs):
    model = m.build_model(MODEL_NAME, SEQ_LEN, num_inputs, m.HIDDEN_SIZE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_shap_values(model, background, to_explain):
    """Try GPU first; cuDNN's RNN backward is known to be finicky under
    SHAP's gradient-based explainers (double-backward / eval-mode
    mismatches), so fall back to CPU on any RuntimeError there."""
    wrapped = _UnsqueezeOutput(model)
    try:
        wrapped.to(m.DEVICE)
        explainer = shap.GradientExplainer(wrapped, background.to(m.DEVICE))
        shap_values = explainer.shap_values(to_explain.to(m.DEVICE))
    except RuntimeError as exc:
        print(f"GPU GradientExplainer failed ({exc!r}); falling back to CPU")
        wrapped.to("cpu")
        explainer = shap.GradientExplainer(wrapped, background.cpu())
        shap_values = explainer.shap_values(to_explain.cpu())

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:  # (n_explain, seq_len, num_inputs, 1) in some shap versions
        shap_values = shap_values[..., 0]
    return shap_values


def run():
    m.set_seed(m.SEED)
    train_inputs, train_target, test_inputs, test_target, num_inputs, feature_names = build_inputs()

    X_train, _ = m.create_sequences(train_inputs, train_target, SEQ_LEN, m.HORIZON)
    X_test, _ = m.create_sequences(test_inputs, test_target, SEQ_LEN, m.HORIZON)
    print(f"train windows: {len(X_train)}, test windows: {len(X_test)}")

    model = load_trained_model(num_inputs)

    rng = np.random.default_rng(m.SEED)
    background_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
    explain_idx = rng.choice(len(X_test), size=min(N_EXPLAIN, len(X_test)), replace=False)

    background = torch.from_numpy(X_train[background_idx])
    to_explain = torch.from_numpy(X_test[explain_idx])

    print(f"computing SHAP: {len(to_explain)} explained windows, {len(background)} background windows...")
    shap_values = compute_shap_values(model, background, to_explain)
    print(f"shap_values shape: {shap_values.shape}  (expected: ({len(to_explain)}, {SEQ_LEN}, {num_inputs}))")

    # Feature-level aggregation, per the advisor's requirement: sum |SHAP|
    # over the 72 lag positions for each column, then average over the
    # 200 explained windows -- 1080 numbers per window collapse to 15.
    per_window_importance = np.abs(shap_values).sum(axis=1)  # (n_explain, num_inputs)
    feature_importance = per_window_importance.mean(axis=0)  # (num_inputs,)

    result = pd.DataFrame({"feature": feature_names, "mean_abs_shap": feature_importance})
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result.to_csv(RESULTS_PATH, index=False)
    print(result.to_string(index=False))
    print(f"\nsaved: {RESULTS_PATH}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(result["feature"][::-1], result["mean_abs_shap"][::-1])
    ax.set_xlabel("Mean |SHAP value| (summed over 72 lags, averaged over 200 windows)")
    ax.set_title(f"Phase 2 -- SHAP feature importance, {MODEL_NAME}@{SEQ_LEN}h")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {PLOT_PATH}")

    return result


if __name__ == "__main__":
    run()
