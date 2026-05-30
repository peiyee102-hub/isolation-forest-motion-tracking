"""
train_isolation_forest.py
=========================
Train an Isolation Forest anomaly detector on good-form rehab reps.

Usage:
    python train_isolation_forest.py

Input:
    dataset/features.csv  (from extract_features.py)

Output:
    models/isolation_forest.joblib  -- trained model
    models/training_summary.json    -- stats and thresholds

Features used (person-invariant):
    sparc, mean_jerk, peak_velocity, mean_velocity,
    duration_s, comp_norm, swing_norm
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ─── CONFIG ───────────────────────────────────────────────────
FEATURES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "features.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.joblib")
SUMMARY_PATH = os.path.join(MODELS_DIR, "training_summary.json")

# Person-invariant features for global model
FEATURE_COLS = [
    "sparc",
    "mean_jerk",
    "peak_velocity",
    "mean_velocity",
    "duration_s",
    "comp_norm",
    "swing_norm",
]

CONTAMINATION = 0.05   # expect ~5% of good reps to be mild outliers
N_ESTIMATORS = 100
RANDOM_STATE = 42


def main():
    print("=" * 55)
    print("  ISOLATION FOREST TRAINER")
    print("=" * 55)

    if not os.path.exists(FEATURES_CSV):
        print(f"\nERROR: {FEATURES_CSV} not found.")
        print("Run extract_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    n_total = len(df)
    print(f"\nLoaded {n_total} reps from {FEATURES_CSV}")

    # Ensure all required columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in features.csv: {missing}")
        return

    X = df[FEATURE_COLS].values

    print(f"\nTraining features: {FEATURE_COLS}")
    print(f"Contamination: {CONTAMINATION}  (~{int(n_total * CONTAMINATION)} expected outliers)")

    # ─── Train Isolation Forest ───────────────────────────────
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # ─── Score training data ──────────────────────────────────
    scores = model.decision_function(X)   # higher = more normal
    labels = model.predict(X)              # 1 = normal, -1 = anomaly

    df["if_score"] = scores
    df["if_label"] = labels

    n_anomaly = int((labels == -1).sum())
    n_normal = int((labels == 1).sum())

    print(f"\nTraining complete!")
    print(f"  Normal reps:   {n_normal}")
    print(f"  Outlier reps:  {n_anomaly}")

    # ─── Show flagged outliers ────────────────────────────────
    if n_anomaly > 0:
        print(f"\n  Flagged as mild outliers (still kept in training):")
        outliers = df[df["if_label"] == -1][["filename", "rom", "duration_s", "sparc", "mean_jerk", "if_score"]]
        for _, row in outliers.iterrows():
            print(f"    {row['filename']:<50}  score={row['if_score']:.3f}")

    # ─── Feature importances (approx via permutation not needed for IF) ───
    # Instead, show mean/std per feature for interpretability
    print(f"\n  Feature statistics (normal reps only):")
    normal_df = df[df["if_label"] == 1]
    stats = {}
    for col in FEATURE_COLS:
        mean_v = float(normal_df[col].mean())
        std_v = float(normal_df[col].std())
        stats[col] = {"mean": round(mean_v, 3), "std": round(std_v, 3)}
        print(f"    {col:<18}  mean={mean_v:>10.3f}  std={std_v:>10.3f}")

    # ─── Save model ───────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved: {MODEL_PATH}")

    # ─── Save summary ─────────────────────────────────────────
    summary = {
        "n_total": n_total,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "contamination": CONTAMINATION,
        "feature_cols": FEATURE_COLS,
        "feature_stats": stats,
        "score_range": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
        },
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {SUMMARY_PATH}")

    # ─── Thresholds for live scoring ──────────────────────────
    # Use the anomaly score of the most borderline normal rep as threshold
    normal_scores = scores[labels == 1]
    threshold = float(np.percentile(normal_scores, 5))  # 5th percentile of normal scores
    print(f"\n  Suggested live threshold (5th pct of normal): {threshold:.4f}")
    print(f"  Reps scoring below this are flagged as anomalous.")

    print("\n" + "=" * 55)
    print("  DONE. Model ready for integration into live_main.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
