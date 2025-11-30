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
# CONFIGURATION (Best Params from Part 1)
# ==========================================
FILE_NAME = 'df_ggn_covariates.csv' # Or Patna
MODEL_VARIANT = 'base'              # Your best model
CONTEXT_DAYS = 14                   # Your best context
HORIZON = 4                         # <--- CHANGED TO 4 HOURS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

print(f"Loading data from {FILE_NAME}...")
df = pd.read_csv(FILE_NAME)
df['From Date'] = pd.to_datetime(df['From Date'])
df = df.sort_values('From Date')

# Interpolate
for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
    df[col] = df[col].interpolate(method='linear')

# 80-20 Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[['RH_avg', 'AT_avg', 'WS_avg']]
y_train = train_df['calibPM']
X_test = test_df[['RH_avg', 'AT_avg', 'WS_avg']]

# ---------------------------------------------------------
# 1. Train XGBoost (Weather Model)
# ---------------------------------------------------------
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds_full = xgb_model.predict(X_test) # Predictions for every hour in test set

# ---------------------------------------------------------
# 2. Run Chronos (4-Hour Rolling Forecast)
# ---------------------------------------------------------
print(f"Loading Chronos ({MODEL_VARIANT})...")
pipeline = ChronosPipeline.from_pretrained(
    f"amazon/chronos-t5-{MODEL_VARIANT}",
    device_map=DEVICE,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)

pm_full = df['calibPM'].values
context_len = CONTEXT_DAYS * 24

# Storage for granular 4-hour segments
results_storage = []

# Slide by HORIZON (4 hours) through the test set
# We stop slightly early to ensure we have a full final window
num_steps = (len(test_df) - context_len) // HORIZON

print(f"Forecasting {num_steps} chunks of {HORIZON} hours...")

for i in range(num_steps):
    # Indices in the FULL dataframe
    start_idx = split_idx + (i * HORIZON)
    end_idx = start_idx + HORIZON
    context_start = start_idx - context_len
    
    if end_idx > len(pm_full): break

    # Ground Truth
    truth = pm_full[start_idx:end_idx]
    
    # Chronos Prediction
    context = torch.tensor(pm_full[context_start:start_idx])
    forecast = pipeline.predict(context=context, prediction_length=HORIZON, num_samples=20)
    chronos_pred = np.median(forecast[0].numpy(), axis=0)
    
    # XGBoost Prediction (Matching indices)
    # XGB preds array starts at 0, which corresponds to split_idx in df
    xgb_idx_start = i * HORIZON
    xgb_idx_end = xgb_idx_start + HORIZON
    xgb_pred = xgb_preds_full[xgb_idx_start:xgb_idx_end]
    
    # Store every single hour's data with its date
    dates = df['From Date'].iloc[start_idx:end_idx].values
    
    for d, t, c, x in zip(dates, truth, chronos_pred, xgb_pred):
        results_storage.append({
            'date': d,
            'truth': t,
            'chronos': c,
            'xgb': x,
            'hybrid': (c + x) / 2
        })

# Convert to DataFrame
res_df = pd.DataFrame(results_storage)
res_df['day_date'] = pd.to_datetime(res_df['date']).dt.date

# ---------------------------------------------------------
# 3. AGGREGATE TO DAILY RMSE (The Correct Way)
# ---------------------------------------------------------
print("\nAggregating 4-hour chunks into Daily Stats...")

daily_stats = []
unique_days = res_df['day_date'].unique()

for day in unique_days:
    day_data = res_df[res_df['day_date'] == day]
    
    # Only calculate if we have significant data for the day (e.g. >20 hours)
    if len(day_data) < 20: 
        continue
        
    rmse_c = np.sqrt(mean_squared_error(day_data['truth'], day_data['chronos']))
    rmse_x = np.sqrt(mean_squared_error(day_data['truth'], day_data['xgb']))
    rmse_h = np.sqrt(mean_squared_error(day_data['truth'], day_data['hybrid']))
    
    daily_stats.append({
        'date': day,
        'rmse_chronos': rmse_c,
        'rmse_xgb': rmse_x,
        'rmse_hybrid': rmse_h
    })

daily_df = pd.DataFrame(daily_stats)
daily_df = daily_df.sort_values('date')

# Global RMSE
global_rmse_c = np.sqrt(mean_squared_error(res_df['truth'], res_df['chronos']))
global_rmse_h = np.sqrt(mean_squared_error(res_df['truth'], res_df['hybrid']))

print("\n" + "="*50)
print("RESULTS (Horizon=4h)")
print("="*50)
print(f"Chronos RMSE: {global_rmse_c:.2f}")
print(f"Hybrid RMSE:  {global_rmse_h:.2f}")

if global_rmse_h < global_rmse_c:
    print(f"SUCCESS: Hybrid Improved by {global_rmse_c - global_rmse_h:.2f}")
else:
    print("No improvement with Hybrid.")

# ---------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------
plt.figure(figsize=(15, 6))
# Plot Daily RMSEs
plt.plot(daily_df['date'], daily_df['rmse_chronos'], label='Chronos Daily RMSE', linestyle='--', color='blue', alpha=0.6)
plt.plot(daily_df['date'], daily_df['rmse_hybrid'], label='Hybrid Daily RMSE', color='red', linewidth=2)

plt.title(f'Daily Forecasting Error (Aggregated from {HORIZON}h Horizon)', fontsize=14)
plt.ylabel('RMSE (Daily)', fontweight='bold')
plt.xlabel('Test Date', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(f'{FILE_NAME.split(".")[0]}_4h_analysis.png')
print("Plot saved.")