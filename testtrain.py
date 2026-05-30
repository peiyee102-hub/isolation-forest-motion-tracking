import os
import pandas as pd
import joblib

csv_filename = 'dataset/features.csv'
model_filename = 'models/isolation_forest.joblib'

if not os.path.exists(csv_filename):
    print(f"❌ Error: Could not find '{csv_filename}'.")
elif not os.path.exists(model_filename):
    print(f"❌ Error: Could not find your saved model at '{model_filename}'.")
else:
    print("🔄 Loading data and saved model...")
    df = pd.read_csv(csv_filename)
    iso_forest = joblib.load(model_filename)

    # 🎯 FIX: Exact feature columns and order used in train_isolation_forest.py
    features_to_use = [
        "sparc",
        "mean_jerk",
        "peak_velocity",
        "mean_velocity",
        "duration_s",
        "comp_norm",
        "swing_norm",
    ]
    
    # Extract raw values to match the array layout the model was trained on
    X = df[features_to_use].values

    print("🔮 Predicting anomalies using your saved joblib model...")
    # Using the array directly removes the feature names warning
    df['anomaly_label'] = iso_forest.predict(X)
    df['anomaly_score'] = iso_forest.decision_function(X) # Matches original script scoring

    print("\n--- 📊 Saved Model Results ---")
    print(df['anomaly_label'].value_counts().rename({1: 'Normal (1)', -1: 'Anomaly (-1)'}))

    print("\n--- 🎯 Flagged Anomalies by Patient/Session ---")
    anomalies = df[df['anomaly_label'] == -1]
    if len(anomalies) == 0:
        print("Your saved model flagged zero anomalies on this dataset.")
    else:
        print(anomalies['patient_id'].value_counts())