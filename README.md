# electricity-consumption-forecasting

## Active study: `multivariate-forecasting/`

Per the advisor's 2026-09-03 revision (deadline 2026-09-21), the binding study is now a multi-dataset, multi-phase electricity load forecasting pipeline (dataset research -> baseline DL models -> SHAP feature ranking -> top-k feature selection -> HPO -> final locked-test-set evaluation) on Panama's national grid load. **See [`multivariate-forecasting/README.md`](multivariate-forecasting/README.md) for the full write-up, all 5 phases, and results.**

## Original study (superseded, kept for reference)

The repository root (this README, `lstm_forecast.py`, `lstm-model.ipynb`, `data/`, `model/`, `plots/`, `report.tex`, etc.) holds the earlier single-dataset study of RNN architectures (LSTM, GRU, BiLSTM, BiGRU) and LSTM-Attention for electricity demand forecasting using PyTorch. It predates the advisor's revision above and is no longer the active work, but is left in place as-is.

