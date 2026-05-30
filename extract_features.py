"""
extract_features.py
===================
Batch feature extractor for raw IMU dataset.

Usage:
    python extract_features.py

Reads all CSV files from dataset/raw/, extracts biomechanical features
via motion_analysis.py, handles brief slave dropouts, computes
person-normalized features, and writes dataset/features.csv.
"""

import os
import glob
import pandas as pd
import numpy as np
from dataclasses import asdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_analysis import extract_features, RepFeatures

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "features.csv")


def is_slave_dropout(row):
    """Detect the master's placeholder when slave ESP-NOW packet is lost."""
    return (row['qx2'] == 0.0 and row['qy2'] == 0.0 and
            row['qz2'] == 0.0 and row['qw2'] == 1.0)


def interpolate_dropouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolate brief slave dropout frames (0,0,0,1 placeholders).
    Only touches qx2,qy2,qz2,qw2 columns.
    """
    mask = df.apply(is_slave_dropout, axis=1)
    if not mask.any():
        return df

    n_drop = mask.sum()
    print(f"    WARNING: {n_drop} slave dropout frame(s) detected -- interpolating...")

    for col in ['qx2', 'qy2', 'qz2', 'qw2']:
        df[col] = df[col].replace({0.0: np.nan, 1.0: np.nan})  # crude but safe
        # Actually, safer: set mask positions to NaN, then interpolate
    # Re-do properly
    for col in ['qx2', 'qy2', 'qz2', 'qw2']:
        df.loc[mask, col] = np.nan
        df[col] = df[col].interpolate(method='linear').ffill().bfill()

    return df


def parse_filename(filename: str):
    """Extract metadata from filename like P999_good_20260528_180833_rep001.csv"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    # Format: <patient>_<label>_<date>_<time>_rep<N>
    if len(parts) >= 5 and parts[-1].startswith('rep'):
        patient_id = parts[0]
        label = parts[1]
        rep_index = parts[-1]
        return patient_id, label, rep_index
    return base, "unknown", "rep000"


def main():
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} raw file(s). Extracting features...\n")

    records = []

    for path in csv_files:
        filename = os.path.basename(path)
        patient_id, label, rep_index = parse_filename(filename)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"    ERROR: {filename}: cannot read ({e})")
            continue

        if len(df) < 10:
            print(f"  WARNING: {filename}: too short ({len(df)} rows) -- skipping")
            continue

        # Interpolate slave dropouts
        df = interpolate_dropouts(df)

        try:
            features: RepFeatures = extract_features(df)
        except Exception as e:
            print(f"  ERROR: {filename}: feature extraction failed ({e})")
            continue

        # Person-normalized scale features
        rom = features.rom if features.rom > 0 else 1.0
        comp_norm = features.comp_mean / rom
        swing_norm = features.upper_rom_y / rom

        record = {
            "filename": filename,
            "patient_id": patient_id,
            "label": label,
            "rep_index": rep_index,
            "rom": features.rom,
            "min_angle": features.min_angle,
            "max_angle": features.max_angle,
            "sparc": features.sparc,
            "comp_mean": features.comp_mean,
            "comp_max": features.comp_max,
            "upper_rom_x": features.upper_rom_x,
            "upper_rom_y": features.upper_rom_y,
            "upper_rom_z": features.upper_rom_z,
            "peak_velocity": features.peak_velocity,
            "mean_velocity": features.mean_velocity,
            "mean_jerk": features.mean_jerk,
            "duration_s": features.duration_s,
            "n_samples": features.n_samples,
            "comp_norm": comp_norm,
            "swing_norm": swing_norm,
        }
        records.append(record)
        print(f"  OK: {filename}  ROM={features.rom:.1f} deg  dur={features.duration_s:.1f}s")

    if not records:
        print("ERROR: No features extracted.")
        sys.exit(1)

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDONE: Extracted {len(records)} feature row(s)")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nFeature summary:")
    print(out_df[['rom', 'sparc', 'mean_jerk', 'peak_velocity',
                  'duration_s', 'comp_norm', 'swing_norm']].describe().round(2))


if __name__ == "__main__":
    main()
