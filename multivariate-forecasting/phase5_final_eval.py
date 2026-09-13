"""
Phase 5 - final evaluation. Retrains the Phase 4 winner (BiGRU, Phase 3's
k=4 feature set, Phase 4's HPO'd hyperparameters) on the FULL original
train partition (not the 85% HPO-only subset used to score trials) for
the full 30-epoch budget matching Phase 1/3's protocol, then evaluates
ONCE on the untouched, locked test set -- this is the study's headline
number, per the advisor's Phase 5 spec.

Winning hyperparameters are hardcoded from the Phase 4 run's best-overall
row (results/phase4_hpo_results.csv, Optuna trial 18): hidden_size=256,
learning_rate=0.000465, batch_size=32, num_layers=1.

Also writes the "toplu sonuc tablosu" the advisor asked for: Phase 1
baseline (all 15 features, default hyperparams) -> Phase 3 top-k (k=4,
default hyperparams) -> Phase 5 final (k=4, HPO'd hyperparams), showing
the pipeline's cumulative improvement.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import phase1_baseline_models as m

MODEL_NAME = "BiGRU"
SEQ_LEN = 72
TOP_K_FEATURES = ["national_demand_mw (lag)", "T2M_toc", "T2M_dav", "T2M_san"]

BEST_HIDDEN_SIZE = 256
BEST_LEARNING_RATE = 0.00046534881364761817
BEST_BATCH_SIZE = 32
BEST_NUM_LAYERS = 1

CHECKPOINT_PATH = "results/checkpoints/final_model.pt"
RESULTS_PATH = "results/phase5_final_results.csv"
SUMMARY_PATH = "results/phase5_pipeline_summary.csv"
ACTUAL_VS_PRED_PATH = "plots/phase5_actual_vs_predicted.png"
ERROR_OVER_TIME_PATH = "plots/phase5_error_over_time.png"


def build_full_inputs():
    df, _ = m.load_data(m.DATA_PATH)
    train_df, test_df = m.chronological_split(df, m.TRAIN_FRACTION)
    exogenous = [f for f in TOP_K_FEATURES if f != f"{m.TARGET} (lag)"]

    feature_scaler = m.StandardScaler().fit(train_df[exogenous].values)
    target_scaler = m.StandardScaler().fit(train_df[[m.TARGET]].values)

    def to_inputs(split_df):
        features = feature_scaler.transform(split_df[exogenous].values)
        target = target_scaler.transform(split_df[[m.TARGET]].values).ravel()
        inputs = np.concatenate([features, target.reshape(-1, 1)], axis=1)
        return inputs, target

    train_inputs, train_target = to_inputs(train_df)
    test_inputs, test_target = to_inputs(test_df)
    return train_inputs, train_target, test_inputs, test_target, target_scaler, test_df


def run():
    m.set_seed(m.SEED)
    train_inputs, train_target, test_inputs, test_target, target_scaler, test_df = build_full_inputs()

    X_train, y_train = m.create_sequences(train_inputs, train_target, SEQ_LEN, m.HORIZON)
    X_test, y_test = m.create_sequences(test_inputs, test_target, SEQ_LEN, m.HORIZON)
    print(f"train windows: {len(X_train)}, test windows: {len(X_test)}")

    m.set_seed(m.SEED)
    model = m.build_model(MODEL_NAME, SEQ_LEN, len(TOP_K_FEATURES), BEST_HIDDEN_SIZE, num_layers=BEST_NUM_LAYERS)
    n_params = sum(p.numel() for p in model.parameters())

    m.train_model(model, X_train, y_train, m.EPOCHS, BEST_BATCH_SIZE, BEST_LEARNING_RATE)
    metrics, pred, true = m.evaluate_model(model, X_test, y_test, target_scaler)

    print(
        f"FINAL  MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.4f} "
        f"MAPE={metrics['mape']:.2f}% params={n_params}"
    )

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved checkpoint: {CHECKPOINT_PATH}")

    final_row = {
        "stage": "Phase 5 final (k=4, HPO'd)",
        "model": MODEL_NAME, "window_length_h": SEQ_LEN, "k": len(TOP_K_FEATURES),
        "hidden_size": BEST_HIDDEN_SIZE, "learning_rate": BEST_LEARNING_RATE,
        "batch_size": BEST_BATCH_SIZE, "num_layers": BEST_NUM_LAYERS,
        "params": n_params, "mae": metrics["mae"], "rmse": metrics["rmse"],
        "r2": metrics["r2"], "mape": metrics["mape"],
    }
    pd.DataFrame([final_row]).to_csv(RESULTS_PATH, index=False)
    print(f"saved: {RESULTS_PATH}")

    phase1 = pd.read_csv("results/phase1_baseline_results.csv")
    phase1_row = phase1[(phase1["model"] == MODEL_NAME) & (phase1["window_length_h"] == SEQ_LEN)].iloc[0]
    phase3 = pd.read_csv("results/phase3_topk_results.csv")
    phase3_row = phase3[phase3["k"] == len(TOP_K_FEATURES)].iloc[0]

    summary = pd.DataFrame([
        {"stage": "Phase 1 baseline (all 15 features, default hyperparams)",
         "k": 15, "hidden_size": m.HIDDEN_SIZE, "mae": phase1_row["mae"], "rmse": phase1_row["rmse"],
         "r2": phase1_row["r2"], "mape": phase1_row["mape"]},
        {"stage": "Phase 3 top-k (k=4, default hyperparams)",
         "k": 4, "hidden_size": m.HIDDEN_SIZE, "mae": phase3_row["mae"], "rmse": phase3_row["rmse"],
         "r2": phase3_row["r2"], "mape": phase3_row["mape"]},
        {"stage": "Phase 5 final (k=4, HPO'd hyperparams)",
         "k": 4, "hidden_size": BEST_HIDDEN_SIZE, "mae": metrics["mae"], "rmse": metrics["rmse"],
         "r2": metrics["r2"], "mape": metrics["mape"]},
    ])
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"saved: {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    target_row_offset = SEQ_LEN + m.HORIZON - 1
    timestamps = test_df.index[target_row_offset: target_row_offset + len(true)]
    errors = true - pred
    zoom_hours = 24 * 14

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(timestamps[:zoom_hours], true[:zoom_hours], label="actual", linewidth=1.3)
    ax.plot(timestamps[:zoom_hours], pred[:zoom_hours], label="predicted", linewidth=1.3, alpha=0.8)
    ax.set_ylabel("National demand (MW)")
    ax.set_title("Phase 5 final model -- actual vs predicted (first 14 days of test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ACTUAL_VS_PRED_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {ACTUAL_VS_PRED_PATH}")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(timestamps, errors, linewidth=0.5, color="tab:red")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Error: actual - predicted (MW)")
    ax.set_title("Phase 5 final model -- prediction error over time (full test period)")
    fig.tight_layout()
    fig.savefig(ERROR_OVER_TIME_PATH, dpi=150)
    plt.close(fig)
    print(f"saved: {ERROR_OVER_TIME_PATH}")

    return summary


if __name__ == "__main__":
    run()
