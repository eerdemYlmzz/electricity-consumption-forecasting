# Multivariate Forecasting Study

Advisor-mandated revision (2026-09-03), superseding the single-dataset work in the repo root. Deadline: 2026-09-21.

## Structure

- `data/raw/` — candidate datasets as downloaded, one subfolder per dataset (e.g. `data/raw/opsd_time_series/`, `data/raw/nasa_power/`). Not tracked in git (see `.gitignore`) — can be large.
- `data/processed/` — cleaned/merged datasets ready for windowing.
- `notebooks/` — one processing notebook per candidate dataset. `00_template.ipynb` is the skeleton to copy for each new one (`0N_<dataset_name>.ipynb`); `common.py` holds the shared checks (`report_candidate`, `check_cumulative`) every notebook imports so the criteria report stays consistent across candidates. `01_opsd_household.ipynb` is the first worked example (raw meter data -> `data/processed/opsd_household_industrial3.csv`).

## Candidate dataset criteria

- ≥10 usable features besides the target (timestamps/IDs/constant columns don't count)
- Hourly sampling preferred
- Report per candidate: source, date range, sampling frequency, row count, feature count, target variable, units, missing-data situation

## Candidate summary (8/8 gathered and processed)

| # | Notebook | Domain | Target | Rows | Features | Date range | Note |
|---|---|---|---|---|---|---|---|
| 1 | `01_opsd_household` | building energy | grid_import (kWh, diffed from cumulative) | 10,429 | 19 | 2016-03 → 2017-06 | only the `industrial3` sub-meter group clears 10 features |
| 2 | `02_uci_appliances_energy` | building energy | Appliances (Wh) | 3,289 | 25 | 2016-01 → 2016-05 | resampled 10min → 1h; dropped duplicate `rv1`/`rv2` noise cols |
| 3 | `03_nasa_power_izmir` | solar | ALLSKY_SFC_SW_DWN (GHI, Wh/m^2) | 43,824 | 13 | 2018 → 2022 | 100% complete; DNI/DIFF excluded (leaky decomposition of target) |
| 4 | `04_beijing_air_quality` | air quality | PM2.5 | 33,339 | 12 | 2013-03 → 2017-02 | ~5% of hourly steps have gaps after cleaning — must respect in windowing |
| 5 | `05_bdg2_building` | sustainable buildings | electricity_kwh | 15,979 | 11 | 2016 → 2017 | **weakest fit**: only 7 real features, 4 are calendar-derived |
| 6 | `06_gefcom_wind` | wind generation | wind_power_norm (0-1) | 16,800 | 11 | 2012-01 → 2013-12 | 100% complete; features derived from wind vectors (speed/dir/shear) |
| 7 | `07_panama_load` | grid load | national_demand_mw | 45,215 | 14 | 2015-01 → 2020-02 | **cleanest**: 0 missing, 0 gaps; truncated before 2020-03 to drop the COVID-19 demand shock (see notebook) |
| 8 | `08_household_power_weather` | household energy | Global_active_power (kW) | 34,142 | 14 | 2006-12 → 2010-11 | UCI household + NASA POWER weather join |

All auth-free sources (no account/API key needed): OPSD, UCI (x3), NASA POWER, GitHub (BDG2, GEFCom2014, Panama). Skipped for now due to required account/API-key registration (not something Claude can do on your behalf): ENTSO-E, EIA, Copernicus/ERA5, Kaggle-hosted sets (ASHRAE, Panama's original host, etc.) — worth adding later if you register for any of these yourself.

Every notebook's `report_candidate` output and markdown cells document the cleaning decisions and known weak points (see each notebook for details).

## Detailed comparison: #7, #3, #8

Selected for the ≥3-detailed-comparison requirement. Deliberately kept to the same problem type (continuous physical/consumption regression target, not mixed with a bounded normalized fraction like GEFCom's `wind_power_norm`) so the comparison is apples-to-apples:

- **#7 Panama load** — grid-scale, cleanest (0 missing, 0 gaps), largest range (5.5y)
- **#3 NASA POWER Izmir** — resource-level (solar irradiance), 100% complete, 5y
- **#8 Household power + weather** — granular single-household consumption with real sub-metering, replaces #1 (OPSD) which had the same feature quality but only 14 months / 10,429 rows vs #8's ~4 years / 34,142 rows

**Final dataset for Phases 1-5: #7, Panama national load** (cleanest, largest, most standard problem type, strongest expected XAI narrative).

## Phases

0. Dataset research (≥8 candidates gathered; #7/#3/#8 compared in detail) — done, Panama selected
1. Baseline DL models — MLP, SimpleRNN, LSTM, GRU, LSTM+Attention, BiGRU, 1D-CNN, TCN (8 architectures) × 6 window lengths (48h-168h → 1h) = 48 runs, chronological split, no validation set, fixed 30-epoch budget. Inputs are the 14 exogenous features plus the target's own lag (`national_demand_mw` at t-seq_len..t-1) — standard autoregressive input, not leakage (see `phase1_baseline_models.py`'s `run_all()` for the reasoning). See `phase1_baseline_models.py`.
   - **Overfitting, documented not hidden:** with no validation signal, train loss keeps falling all 30 epochs while test RMSE bottoms out earlier (best epoch ~23/30 for the representative GRU/48h case) — dropout/weight decay were tried and barely helped, so Phase 1 results reflect "whatever a fixed 30-epoch budget produces" more than a tuned stopping point; see `plots/phase1_overfitting_evidence.png`.
   - **Target lag matters a lot:** without the target's own history as input, R² capped out around 0.32-0.55 across all 8 architectures/6 windows — barely better than a naive persistence guess for a series this autocorrelated. Adding the lag pushed R² to ~0.68-0.99, confirming the target's own recent past was the dominant missing signal. LSTM_Attention and CNN1D degrade at longer windows (R² drops to 0.68-0.89 past 120h) while MLP/SimpleRNN/LSTM/GRU/BiGRU/TCN stay in the high-0.9s across all 6 window lengths — best single config: **BiGRU @ 72h, R²=0.988, RMSE=20.99 MW, MAPE=1.21%**.
   - Dataset is also truncated before 2020-03 (see #7 above) to remove a genuine train/test distribution shift (COVID-19 demand shock) that was compounding the overfitting signal.
   - Plots: `plots/phase1_window_length_comparison.png`, `plots/phase1_overfitting_evidence.png`, `plots/phase1_actual_vs_predicted.png`, `plots/phase1_error_over_time.png`.
2. XAI feature ranking (SHAP/LIME, feature-level not lag-level)
3. Top-k feature subset experiments (k = 3,4,5,6,8,10)
4. HPO (Optuna + one faster alternative), selected features only
5. Final evaluation on locked test set (MAE, RMSE, R²)
