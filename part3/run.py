import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import torch
from chronos import ChronosPipeline
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
FILE_NAME = 'df_ggn_covariates.csv'
MODEL_VARIANT = 'base'
CONTEXT_DAYS = 14
HORIZON = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

print(f"Loading data from {FILE_NAME}...")
df = pd.read_csv(FILE_NAME)
df['From Date'] = pd.to_datetime(df['From Date'])
df = df.sort_values('From Date')

# Interpolate
for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
    df[col] = df[col].interpolate(method='linear')

# ==============================================================================
# CRITICAL FIX: Adding "Lag" Features
# This allows XGBoost to see "Yesterday's Value" just like Chronos does.
# This fixes the "Training on Summer, Testing on Winter" issue.
# ==============================================================================
df['lag_24h'] = df['calibPM'].shift(24) # Value 1 day ago
df['lag_48h'] = df['calibPM'].shift(48) # Value 2 days ago
df['hour'] = df['From Date'].dt.hour
df['month'] = df['From Date'].dt.month

# Drop the first 48 hours (NaNs) created by shifting
df = df.dropna()

# Features for XGBoost
X = df[['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']]
y = df['calibPM']

# ==============================================================================
# PROOF OF 80-20 SPLIT (As per assignment)
# ==============================================================================
split_idx = int(len(df) * 0.8)

# XGBoost Training Data (First 80%)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

# Test Data (Last 20%)
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
dates_test = df['From Date'].iloc[split_idx:]

print(f"Training XGBoost on {len(X_train)} hours (First 80%)...")
print(f"Testing on {len(X_test)} hours (Last 20%)...")

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# Pre-calculate XGBoost predictions for the whole test set
xgb_preds_full = xgb_model.predict(X_test)

# ==============================================================================
# RUNNING CHRONOS (On the exact same 20% Test Set)
# ==============================================================================
print(f"Running Chronos on the same 20% Test Set...")
pipeline = ChronosPipeline.from_pretrained(
    f"amazon/chronos-t5-{MODEL_VARIANT}",
    device_map=DEVICE,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)

pm_full = df['calibPM'].values
chronos_preds = []
ground_truth = []
xgb_preds_aligned = []
valid_dates = []

# We iterate through the TEST set
num_steps = (len(y_test) - (CONTEXT_DAYS*24)) // HORIZON

for i in range(num_steps):
    # Map 'i' to the actual index in the FULL dataframe
    # We start at split_idx
    curr_idx = split_idx + (i * HORIZON)
    
    # Define Windows
    target_start = curr_idx
    target_end = curr_idx + HORIZON
    context_start = curr_idx - (CONTEXT_DAYS * 24)
    
    if target_end > len(df): break
    
    # 1. Chronos Inference
    ctx = torch.tensor(pm_full[context_start:target_start])
    forecast = pipeline.predict(context=ctx, prediction_length=HORIZON, num_samples=20)
    c_pred = np.median(forecast[0].numpy(), axis=0)
    
    # 2. Retrieve corresponding XGBoost Prediction
    # XGB preds array starts at 0, which corresponds to split_idx
    x_start = i * HORIZON
    x_end = x_start + HORIZON
    x_pred = xgb_preds_full[x_start:x_end]
    
    # 3. Ground Truth
    gt = pm_full[target_start:target_end]
    
    # Accumulate
    chronos_preds.extend(c_pred)
    xgb_preds_aligned.extend(x_pred)
    ground_truth.extend(gt)
    valid_dates.extend(df['From Date'].iloc[target_start:target_end])

# ==============================================================================
# FINAL COMPARISON
# ==============================================================================
gt = np.array(ground_truth)
p_c = np.array(chronos_preds)
p_x = np.array(xgb_preds_aligned)

# Hybrid Average
p_h = (p_c + p_x) / 2

rmse_c = np.sqrt(mean_squared_error(gt, p_c))
rmse_x = np.sqrt(mean_squared_error(gt, p_x))
rmse_h = np.sqrt(mean_squared_error(gt, p_h))

print("\n" + "="*50)
print(f"FINAL RMSE RESULTS (Lagged XGBoost)")
print("="*50)
print(f"Chronos Only: {rmse_c:.2f}")
print(f"XGBoost Only: {rmse_x:.2f}")
print(f"Hybrid Model: {rmse_h:.2f}")
print("-" * 50)

if rmse_h < rmse_c:
    print(f"SUCCESS: Hybrid model beat Chronos by {rmse_c - rmse_h:.2f}!")
else:
    print(f"RESULT: Chronos still wins. (Diff: {rmse_h - rmse_c:.2f})")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(valid_dates, gt, label='Truth', color='black', alpha=0.5)
plt.plot(valid_dates, p_c, label='Chronos', color='blue', alpha=0.5, linestyle='--')
plt.plot(valid_dates, p_h, label='Hybrid', color='red', linewidth=1.5)
plt.legend()
plt.title("Chronos vs Hybrid (with 80-20 Split & Lag Features)")
plt.savefig("final_split_check.png")
print("Plot saved.")