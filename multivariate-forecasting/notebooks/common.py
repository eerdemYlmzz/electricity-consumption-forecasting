"""Shared checks reused across the per-dataset processing notebooks.

Every candidate dataset gets evaluated against the same criteria (row count,
feature count, sampling frequency, missing data, constant columns), and
energy meter datasets need the same cumulative-counter check before diffing.
Keeping these in one place avoids re-deriving the same pandas one-liners in
each of the 8 notebooks.
"""
import pandas as pd


def report_candidate(df, target_col, feature_cols, freq="1h", name=""):
    """Print the standard per-candidate report: date range, row count,
    feature count, sampling frequency compliance, missing data, constant columns.
    """
    idx = df.index
    print(f"=== {name or target_col} ===")
    print(f"date range: {idx.min()} -> {idx.max()}")
    print(f"rows: {len(df)}")
    print(f"features (excl. target): {len(feature_cols)}")

    diffs = idx.to_series().diff().dropna()
    freq_ok = (diffs == pd.Timedelta(freq)).mean()
    print(f"fraction of steps matching expected freq ({freq}): {freq_ok:.4f}")

    cols = [target_col] + list(feature_cols)
    nn = df[cols].notna().mean().sort_values() * 100
    print("\nnon-null % (lowest first):")
    print(nn.to_string())

    const_cols = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        print(f"\nWARNING constant columns (must not count toward the 10-feature minimum): {const_cols}")


def check_cumulative(df, cols):
    """Fraction of decreasing steps per column. Near 0 => monotonic cumulative
    counter (needs .diff() before modeling); well above 0 => already an
    instantaneous/interval value.
    """
    frac_negative = {}
    for c in cols:
        s = df[c].dropna()
        d = s.diff().dropna()
        frac_negative[c] = (d < 0).mean() if len(d) else float("nan")
    return pd.Series(frac_negative, name="frac_negative_diff").sort_values()
