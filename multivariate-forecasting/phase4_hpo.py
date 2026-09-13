"""
Phase 4 - hyperparameter optimization, BiGRU on the Phase 3 k=4 feature
set (target lag + T2M_toc, T2M_dav, T2M_san), window=72h.

Two methods, equal trial budget, per the advisor's "Optuna plus at least
one faster alternative" requirement:
  - Optuna (TPESampler, Bayesian) -- the required method.
  - Random Search -- the faster alternative. The advisor's own mail
    already calls Bayesian methods "hantal" (cumbersome); random search
    adds no surrogate-model overhead and is the standard baseline
    Bayesian search is expected to beat (Bergstra & Bengio 2012).

Validation split for HPO scoring only (NOT a persistent val set across
the whole study): the original train partition (80% of the series) is
itself split chronologically 85/15 into hpo_train/hpo_val. Every trial
trains on hpo_train and is scored on hpo_val. The original test
partition is never touched here -- it stays locked for Phase 5's final
number, so no hyperparameter decision is made by looking at the test
set. Scalers for this phase are fit on hpo_train only.

Each trial trains for HPO_EPOCHS (well under the 30-epoch Phase 1-3
budget) since the goal here is *relative* ranking of configs, not a
finished model -- Phase 5 retrains the winning config for the full
budget on the full original train partition, then evaluates once on the
untouched test set.
"""
import time

import numpy as np
import optuna
import pandas as pd

import phase1_baseline_models as m

MODEL_NAME = "BiGRU"
SEQ_LEN = 72
TOP_K_FEATURES = ["national_demand_mw (lag)", "T2M_toc", "T2M_dav", "T2M_san"]
HPO_TRAIN_FRACTION = 0.85  # of the original train partition, chronological
HPO_EPOCHS = 12
N_TRIALS = 30  # per method
RESULTS_PATH = "results/phase4_hpo_results.csv"

SEARCH_SPACE = {
    "hidden_size": [32, 64, 128, 256],
    "learning_rate": (1e-4, 1e-2),  # log-uniform bounds
    "batch_size": [32, 64, 128, 256],
    "num_layers": [1, 2],
}


def build_hpo_data():
    """Mirrors phase1_baseline_models.run_all()'s preprocessing, but (a)
    restricted to the k=4 feature set and (b) with an extra chronological
    split carving hpo_val out of the original train partition. Windowing
    (create_sequences) happens once here, not per trial -- the windowed
    data never changes across trials, only the hyperparameters do, so
    recomputing it 60 times was pure waste (and, with create_sequences'
    unvectorized Python loop over ~30k rows, a very expensive one)."""
    df, _ = m.load_data(m.DATA_PATH)
    train_df, _test_df = m.chronological_split(df, m.TRAIN_FRACTION)  # test_df untouched, unused here
    hpo_train_df, hpo_val_df = m.chronological_split(train_df, HPO_TRAIN_FRACTION)

    exogenous = [f for f in TOP_K_FEATURES if f != f"{m.TARGET} (lag)"]

    feature_scaler = m.StandardScaler().fit(hpo_train_df[exogenous].values)
    target_scaler = m.StandardScaler().fit(hpo_train_df[[m.TARGET]].values)

    def to_inputs(split_df):
        features = feature_scaler.transform(split_df[exogenous].values)
        target = target_scaler.transform(split_df[[m.TARGET]].values).ravel()
        inputs = np.concatenate([features, target.reshape(-1, 1)], axis=1)
        return inputs, target

    hpo_train_inputs, hpo_train_target = to_inputs(hpo_train_df)
    hpo_val_inputs, hpo_val_target = to_inputs(hpo_val_df)
    print(
        f"hpo_train: {len(hpo_train_df)} rows, hpo_val: {len(hpo_val_df)} rows "
        "(both inside the original train partition)", flush=True,
    )

    X_train, y_train = m.create_sequences(hpo_train_inputs, hpo_train_target, SEQ_LEN, m.HORIZON)
    X_val, y_val = m.create_sequences(hpo_val_inputs, hpo_val_target, SEQ_LEN, m.HORIZON)
    print(f"windowed once: X_train={X_train.shape}, X_val={X_val.shape}", flush=True)

    return X_train, y_train, X_val, y_val, target_scaler


def train_and_score(hidden_size, learning_rate, batch_size, num_layers, X_train, y_train, X_val, y_val, target_scaler):
    m.set_seed(m.SEED)  # same seed every trial -- isolates the hyperparameter effect from init/shuffle noise
    model = m.build_model(MODEL_NAME, SEQ_LEN, len(TOP_K_FEATURES), hidden_size, num_layers=num_layers)

    start = time.time()
    m.train_model(model, X_train, y_train, HPO_EPOCHS, batch_size, learning_rate)
    metrics, _, _ = m.evaluate_model(model, X_val, y_val, target_scaler)
    elapsed = time.time() - start

    return metrics["rmse"], metrics["r2"], elapsed


def run_random_search(data, rng):
    records = []
    for i in range(N_TRIALS):
        hidden_size = int(rng.choice(SEARCH_SPACE["hidden_size"]))
        batch_size = int(rng.choice(SEARCH_SPACE["batch_size"]))
        num_layers = int(rng.choice(SEARCH_SPACE["num_layers"]))
        low, high = SEARCH_SPACE["learning_rate"]
        learning_rate = float(np.exp(rng.uniform(np.log(low), np.log(high))))

        val_rmse, val_r2, elapsed = train_and_score(hidden_size, learning_rate, batch_size, num_layers, *data)
        print(
            f"[random {i + 1:>2}/{N_TRIALS}] hidden={hidden_size:>3} lr={learning_rate:.5f} "
            f"batch={batch_size:>3} layers={num_layers}  val_RMSE={val_rmse:.3f} val_R2={val_r2:.4f} time={elapsed:.1f}s", flush=True
        )
        records.append({
            "method": "random", "trial": i, "hidden_size": hidden_size, "learning_rate": learning_rate,
            "batch_size": batch_size, "num_layers": num_layers, "val_rmse": val_rmse, "val_r2": val_r2,
            "train_seconds": elapsed,
        })
    return records


def run_optuna(data):
    records = []

    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"])
        learning_rate = trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True)
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])
        num_layers = trial.suggest_categorical("num_layers", SEARCH_SPACE["num_layers"])

        val_rmse, val_r2, elapsed = train_and_score(hidden_size, learning_rate, batch_size, num_layers, *data)
        print(
            f"[optuna {trial.number + 1:>2}/{N_TRIALS}] hidden={hidden_size:>3} lr={learning_rate:.5f} "
            f"batch={batch_size:>3} layers={num_layers}  val_RMSE={val_rmse:.3f} val_R2={val_r2:.4f} time={elapsed:.1f}s", flush=True
        )
        records.append({
            "method": "optuna", "trial": trial.number, "hidden_size": hidden_size, "learning_rate": learning_rate,
            "batch_size": batch_size, "num_layers": num_layers, "val_rmse": val_rmse, "val_r2": val_r2,
            "train_seconds": elapsed,
        })
        return val_rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=m.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return records


def run():
    m.set_seed(m.SEED)
    data = build_hpo_data()

    print(f"\n=== Optuna ({N_TRIALS} trials) ===")
    optuna_records = run_optuna(data)

    print(f"\n=== Random Search ({N_TRIALS} trials) ===")
    rng = np.random.default_rng(m.SEED)
    random_records = run_random_search(data, rng)

    results = pd.DataFrame(optuna_records + random_records)
    results.to_csv(RESULTS_PATH, index=False)

    best_optuna = results[results["method"] == "optuna"].sort_values("val_rmse").iloc[0]
    best_random = results[results["method"] == "random"].sort_values("val_rmse").iloc[0]
    best_overall = results.sort_values("val_rmse").iloc[0]

    print(f"\nbest Optuna:        {best_optuna.to_dict()}")
    print(f"best Random Search: {best_random.to_dict()}")
    print(f"best overall:       {best_overall.to_dict()}")
    print(f"\nsaved: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run()
