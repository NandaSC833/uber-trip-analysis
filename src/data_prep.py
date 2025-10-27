# src/data_prep.py
import pandas as pd
import sys
from pathlib import Path

def load_and_clean(path):
    # Load CSV
    df = pd.read_csv(path, low_memory=False)
    print("✅ Columns found:", df.columns.tolist())

    # Convert date column
    if 'date' not in df.columns:
        raise ValueError("❌ 'date' column not found in dataset!")

    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')

    # If data is per base, sum up to get total trips per datetime
    if 'trips' in df.columns:
        hourly = df.groupby('datetime')['trips'].sum().reset_index()
    else:
        # fallback: count rows per hour
        hourly = df.groupby('datetime').size().reset_index(name='trips')

    # Rename to consistent format
    hourly = hourly.rename(columns={'trips': 'Count'})
    hourly = hourly.set_index('datetime')

    print(f"✅ Cleaned data shape: {hourly.shape}")
    return hourly

def save_processed(df, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"📁 Saved processed hourly data to: {out_path}")
    print(df.head())

if __name__ == "__main__":
    src = sys.argv[1]  # input file path
    out = sys.argv[2]  # output file path
    df = load_and_clean(src)
    save_processed(df, out)
