"""
Phase 1 — Grid Data Simulator
Generates 50,000 rows of synthetic power grid sensor data with IEC 61850 column names.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility
np.random.seed(42)

TOTAL_ROWS  = 50000
EXTRA_ROWS  = 50020         # generate extra to absorb rolling-window NaN drops
FAULT_RATIO = 0.10          # 10% faults
FAULT_TYPES = [1, 2, 3, 4, 5]
MISSING_RATIO = 0.01        # 1% missing values
OUTPUT_PATH = Path(__file__).parent / "grid_data.csv"


def inject_faults(df: pd.DataFrame) -> pd.DataFrame:
    """Inject fault-specific sensor anomalies into the dataframe."""
    n = len(df)
    n_faults = int(n * FAULT_RATIO)
    n_per_type = n_faults // len(FAULT_TYPES)

    # Start with all-normal fault_type
    df['fault_type'] = 0

    used_indices: set = set()

    for ft in FAULT_TYPES:
        # Pick random unused indices
        candidates = list(set(range(n)) - used_indices)
        chosen = np.random.choice(candidates, size=n_per_type, replace=False)
        used_indices.update(chosen.tolist())
        df.loc[chosen, 'fault_type'] = ft

        if ft == 1:  # Overvoltage
            df.loc[chosen, 'MMXU1.PhV.phsA'] = np.random.uniform(261, 290, size=len(chosen))
            df.loc[chosen, 'MMXU1.PhV.phsB'] = np.random.uniform(261, 290, size=len(chosen))

        elif ft == 2:  # Voltage Sag
            df.loc[chosen, 'MMXU1.PhV.phsA'] = np.random.uniform(140, 179, size=len(chosen))
            df.loc[chosen, 'MMXU1.PhV.phsB'] = np.random.uniform(140, 179, size=len(chosen))

        elif ft == 3:  # Overload
            df.loc[chosen, 'MMXU1.A.phsA'] = np.random.uniform(31, 50, size=len(chosen))

        elif ft == 4:  # Spike — 1-3 consecutive rows each
            for idx in chosen:
                spike_len = np.random.randint(1, 4)
                spike_range = range(idx, min(idx + spike_len, n))
                df.loc[list(spike_range), 'MMXU1.PhV.phsA'] = 300.0
                df.loc[list(spike_range), 'MMXU1.PhV.phsB'] = 300.0
                df.loc[list(spike_range), 'fault_type'] = 4

        elif ft == 5:  # Line Fault
            df.loc[chosen, 'MMXU1.PhV.phsA'] = 0.0
            df.loc[chosen, 'MMXU1.PhV.phsB'] = 0.0
            df.loc[chosen, 'MMXU1.A.phsA'] = np.random.uniform(60, 100, size=len(chosen))

    return df


def add_gaussian_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Add small Gaussian noise to all sensor readings."""
    sensor_cols = ['MMXU1.PhV.phsA', 'MMXU1.PhV.phsB', 'MMXU1.A.phsA', 'MMXU1.Hz', 'MMXU1.W']
    noise_std = {'MMXU1.PhV.phsA': 0.5, 'MMXU1.PhV.phsB': 0.5,
                 'MMXU1.A.phsA': 0.1, 'MMXU1.Hz': 0.02, 'MMXU1.W': 5.0}
    for col in sensor_cols:
        df[col] += np.random.normal(0, noise_std[col], size=len(df))
    return df


def inject_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inject 1% missing values randomly into sensor columns."""
    sensor_cols = ['MMXU1.PhV.phsA', 'MMXU1.PhV.phsB', 'MMXU1.A.phsA', 'MMXU1.Hz', 'MMXU1.W']
    for col in sensor_cols:
        mask = np.random.random(len(df)) < MISSING_RATIO
        df.loc[mask, col] = np.nan
    df[sensor_cols] = df[sensor_cols].ffill()
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics and lag features."""
    df['volt_rolling_mean'] = df['MMXU1.PhV.phsA'].rolling(10).mean()
    df['volt_rolling_std']  = df['MMXU1.PhV.phsA'].rolling(10).std()
    df['freq_rolling_mean'] = df['MMXU1.Hz'].rolling(10).mean()
    df['volt_lag1']         = df['MMXU1.PhV.phsA'].shift(1)
    df['volt_lag5']         = df['MMXU1.PhV.phsA'].shift(5)
    return df


def main():
    print("Generating grid sensor data...")

    # --- Generate base normal data ---
    timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=EXTRA_ROWS, freq='s')
    df = pd.DataFrame({
        'timestamp':        timestamps,
        'MMXU1.PhV.phsA':  np.random.uniform(220, 240, EXTRA_ROWS),
        'MMXU1.PhV.phsB':  np.random.uniform(220, 240, EXTRA_ROWS),
        'MMXU1.A.phsA':    np.random.uniform(5, 15, EXTRA_ROWS),
        'MMXU1.Hz':        np.random.uniform(49.8, 50.2, EXTRA_ROWS),
        'MMXU1.W':         np.random.uniform(1000, 3000, EXTRA_ROWS),
    })

    # --- Inject faults ---
    df = inject_faults(df)

    # --- Add Gaussian noise ---
    df = add_gaussian_noise(df)

    # --- Inject 1% missing values, then ffill ---
    df = inject_missing_values(df)

    # --- Engineered features ---
    df = add_engineered_features(df)

    # --- Drop NaN rows from rolling window start, then trim to exactly 50000 ---
    df.dropna(inplace=True)
    df = df.iloc[:TOTAL_ROWS].copy()
    df.reset_index(drop=True, inplace=True)

    # --- Save ---
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Grid data saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Fault distribution:\n{df['fault_type'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
