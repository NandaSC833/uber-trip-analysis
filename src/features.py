# src/features.py
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

def create_features(df, target_col='Count', lags=[1, 2, 3, 7, 14]):
    df_feat = pd.DataFrame(index=df.index)
    df_feat[target_col] = df[target_col]

    # Lag features
    for lag in lags:
        df_feat[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling stats
    df_feat['roll_mean_7'] = df[target_col].shift(1).rolling(window=7).mean()
    df_feat['roll_std_7'] = df[target_col].shift(1).rolling(window=7).std()

    # Date-based features
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)

    # Drop rows with NaNs from lag creation
    df_feat = df_feat.dropna()
    return df_feat

def split_data(df, target_col='Count', test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train.drop(columns=[target_col]), train[target_col]
    X_test, y_test = test.drop(columns=[target_col]), test[target_col]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    src = sys.argv[1]   # e.g. data/processed/uber_hourly.csv
    out_dir = Path(sys.argv[2])  # e.g. data/processed/features
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, parse_dates=['datetime'], index_col='datetime')
    print("✅ Loaded data:", df.shape)

    df_feat = create_features(df)
    X_train, X_test, y_train, y_test = split_data(df_feat)

    # Save all for modeling
    joblib.dump((X_train, y_train), out_dir / 'train.joblib')
    joblib.dump((X_test, y_test), out_dir / 'test.joblib')

    print("📦 Features created and saved successfully!")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
