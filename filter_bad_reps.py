"""
filter_bad_reps.py
==================
Auto-filter accidental/invalid recordings from dataset.

Criteria for removal:
  - duration_s < 3.0 seconds   (accidental keypress)
  - rom < 40 degrees           (no real movement)
  - mean_jerk > 50000         (noise spike from short recording)

Usage:
    python filter_bad_reps.py          # Preview only (safe)
    python filter_bad_reps.py --delete # Actually delete bad files
"""

import pandas as pd
import os
import sys
import argparse

FEATURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "features.csv")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw")


def main():
    parser = argparse.ArgumentParser(description="Filter bad reps from dataset")
    parser.add_argument("--delete", action="store_true", help="Actually delete bad files (default: preview only)")
    args = parser.parse_args()

    if not os.path.exists(FEATURES_PATH):
        print(f"ERROR: {FEATURES_PATH} not found. Run extract_features.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH)
    n_total = len(df)

    # Define bad criteria
    mask_short = df['duration_s'] < 3.0
    mask_low_rom = df['rom'] < 40.0
    mask_noise = df['mean_jerk'] > 50000

    mask_bad = mask_short | mask_low_rom | mask_noise
    bad_df = df[mask_bad].copy()
    good_df = df[~mask_bad].copy()

    print("=" * 60)
    print("  DATASET FILTER REPORT")
    print("=" * 60)
    print(f"\nTotal reps:  {n_total}")
    print(f"Bad reps:    {len(bad_df)}  ({len(bad_df)/n_total*100:.1f}%)")
    print(f"Good reps:   {len(good_df)}  ({len(good_df)/n_total*100:.1f}%)")

    if len(bad_df) == 0:
        print("\nNo bad reps found. Dataset is clean.")
        return

    print("\n" + "-" * 60)
    print("  BAD REPS (will be removed):")
    print("-" * 60)
    print(f"{'Filename':<50} {'Duration':>8} {'ROM':>8} {'Jerk':>10}")
    print("-" * 60)

    for _, row in bad_df.iterrows():
        reason = []
        if row['duration_s'] < 3.0:
            reason.append("short")
        if row['rom'] < 40.0:
            reason.append("low_ROM")
        if row['mean_jerk'] > 50000:
            reason.append("noise")
        reason_str = ",".join(reason)

        print(f"{row['filename']:<50} {row['duration_s']:>7.1f}s {row['rom']:>7.1f} deg {row['mean_jerk']:>10.1f}  ({reason_str})")

        if args.delete:
            raw_path = os.path.join(RAW_DIR, row['filename'])
            if os.path.exists(raw_path):
                os.remove(raw_path)
                print(f"    -> DELETED")

    if args.delete:
        # Save cleaned features CSV
        good_df.to_csv(FEATURES_PATH, index=False)
        print(f"\nCleaned features saved: {FEATURES_PATH}")
        print(f"Remaining good reps: {len(good_df)}")
    else:
        print("\n" + "=" * 60)
        print("  PREVIEW MODE -- no files deleted.")
        print("  Run with --delete to actually remove bad reps.")
        print("=" * 60)


if __name__ == "__main__":
    main()
