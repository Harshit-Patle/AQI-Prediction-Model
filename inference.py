import argparse
import sys
import os
import pandas as pd
import numpy as np
import joblib

# ── 1. Model & Features Configuration ──────────────────────────────────────────
MODEL_PATH = 'best_model.pkl'

FEATURE_COLS = [
    'PM2.5', 'NO2', 'year', 'month', 'day', 'day_of_week', 
    'is_weekend', 'is_north', 'pollution_ratio', 'season_encoded'
]

AQI_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    df = df.copy()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['day_of_week'] = df['Date'].dt.dayofweek
    
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    north_cities = ['Delhi', 'Lucknow', 'Patna', 'Gurugram', 'Jaipur', 'Chandigarh']
    if 'City' in df.columns:
        df['is_north'] = df['City'].isin(north_cities).astype(int)
    else:
        df['is_north'] = 0
        
    if 'PM2.5' in df.columns:
        df['PM2.5'] = df['PM2.5'].fillna(45.0) 
    if 'NO2' in df.columns:
        df['NO2'] = df['NO2'].fillna(20.0)
        
    if 'PM2.5' in df.columns and 'NO2' in df.columns:
        df['pollution_ratio'] = df['PM2.5'] / (df['NO2'] + 0.001)
        
    season_map = {'Winter': 1, 'Summer': 2, 'Monsoon': 3, 'Post-Monsoon': 4}
    if 'season' in df.columns:
        df['season_encoded'] = df['season'].map(season_map).fillna(0)
    else:
        df['season_encoded'] = 0
        
    return df

# ── 3. Main Execution Block (This is what actually runs!) ─────────────────────
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AQI Hackathon Inference Script')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV (e.g., test.csv)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to save output CSV')
    args = parser.parse_args()

    # 1. Load Data
    print(f"[*] Loading input data from: {args.input}")
    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Could not find the file {args.input}. Are you in the right directory?")
    df = pd.read_csv(args.input)

    # 2. Engineer Features
    print("[*] Applying feature engineering...")
    df_fe = engineer_features(df)
    
    # Check for missing features
    missing_cols = [col for col in FEATURE_COLS if col not in df_fe.columns]
    if missing_cols:
        sys.exit(f"[ERROR] Missing engineered features: {missing_cols}")
        
    X = df_fe[FEATURE_COLS]

    # 3. Load Model
    print(f"[*] Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] Could not find {MODEL_PATH}. Make sure it is in the same folder.")
    model = joblib.load(MODEL_PATH)

    # 4. Predict
    print("[*] Generating predictions...")
    preds = model.predict(X)
    
    # Map integer predictions back to text classes if necessary
    if np.issubdtype(preds.dtype, np.integer):
        preds = [AQI_ORDER[p] for p in preds]

    # 5. Format Output
    print("[*] Formatting submission file...")
    out = df[['City', 'StationId', 'Date']].copy()
    out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
    out['AQI_Bucket'] = preds
    
    # 6. Save File
    out.to_csv(args.output, index=False)
    print(f"[SUCCESS] {len(out)} predictions saved to {args.output}")