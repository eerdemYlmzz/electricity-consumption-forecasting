"""
Phase 3 - top-k feature subset experiments, BiGRU @ 72h.

Uses the Phase 2 SHAP ranking (results/phase2_shap_feature_importance.csv)
to select the top-k features for k in {3,4,5,6,8,10}, per the advisor's
spec. Same architecture and window as Phase 2 (BiGRU@72h, the best
Phase 1 config) so this phase isolates the effect of feature selection
alone -- not confounded with an architecture or window-length change.

Same protocol as Phase 1: chronological split, no validation set, fixed
30-epoch budget, StandardScaler fit on train only (refit here since the
feature set is smaller -- fitting on all 15 columns and then slicing
would leak the dropped columns' distribution into the scaler's fitted
mean/std of the kept ones only if columns interacted, which they don't
for a per-column StandardScaler, but refitting per-subset is the
correct, unambiguous version of "fit only on train").

The all-features (k=15) BiGRU@72h result is pulled from the existing
Phase 1 run (results/phase1_baseline_results.csv) rather than retrained
-- identical setup, no need to redo the compute -- and included as the
row top-k should be judged against, not against zero.
"""
import time

import numpy as np
import pandas as pd

import phase1_baseline_models as m

MODEL_NAME = "BiGRU"
SEQ_LEN = 72
K_VALUES = [3, 4, 5, 6, 8, 10]
SHAP_RANKING_PATH = "results/phase2_shap_feature_importance.csv"
PHASE1_RESULTS_PATH = "results/phase1_baseline_results.csv"
RESULTS_PATH = "results/phase3_topk_results.csv"


def build_full_inputs():
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
    feature_names = feature_cols + [f"{m.TARGET} (lag)"]

    return train_inputs, train_target, test_inputs, test_target, feature_names, target_scaler


def load_ranking():
    ranking = pd.read_csv(SHAP_RANKING_PATH)
    return ranking.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()


def run():
    m.set_seed(m.SEED)
    train_inputs, train_target, test_inputs, test_target, feature_names, target_scaler = build_full_inputs()
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    ranked_features = load_ranking()

    records = []
    for k in K_VALUES:
        top_k_features = ranked_features[:k]
        idx = [name_to_idx[f] for f in top_k_features]

        train_subset = train_inputs[:, idx]
        test_subset = test_inputs[:, idx]

        X_train, y_train = m.create_sequences(train_subset, train_target, SEQ_LEN, m.HORIZON)
        X_test, y_test = m.create_sequences(test_subset, test_target, SEQ_LEN, m.HORIZON)

        m.set_seed(m.SEED)
        model = m.build_model(MODEL_NAME, SEQ_LEN, k, m.HIDDEN_SIZE)
        n_params = sum(p.numel() for p in model.parameters())

        start = time.time()
        m.train_model(model, X_train, y_train, m.EPOCHS, m.BATCH_SIZE, m.LEARNING_RATE)
        metrics, _, _ = m.evaluate_model(model, X_test, y_test, target_scaler)
        elapsed = time.time() - start

        print(
            f"k={k:>2}  MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
            f"R2={metrics['r2']:.4f} MAPE={metrics['mape']:.2f}% "
            f"params={n_params} time={elapsed:.1f}s  features={top_k_features}"
        )
        records.append({
            "k": k,
            "features": ";".join(top_k_features),
            "params": n_params,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "train_seconds": elapsed,
        })

    phase1_results = pd.read_csv(PHASE1_RESULTS_PATH)
    baseline = phase1_results[
        (phase1_results["model"] == MODEL_NAME) & (phase1_results["window_length_h"] == SEQ_LEN)
    ].iloc[0]
    records.append({
        "k": len(feature_names),
        "features": ";".join(feature_names),
        "params": int(baseline["params"]),
        "mae": baseline["mae"],
        "rmse": baseline["rmse"],
        "r2": baseline["r2"],
        "mape": baseline["mape"],
        "train_seconds": baseline["train_seconds"],
    })
    print(
        f"k=15 (all, from Phase 1)  MAE={baseline['mae']:.3f} RMSE={baseline['rmse']:.3f} "
        f"R2={baseline['r2']:.4f} MAPE={baseline['mape']:.2f}%"
    )

    results = pd.DataFrame(records).sort_values("k").reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nsaved: {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run()
