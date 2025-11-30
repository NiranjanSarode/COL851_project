import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from chronos import ChronosPipeline
import time
import os

# Initialize App
app = FastAPI(title="Chronos Forecasting Server")

# ==========================================
# GLOBAL STATE
# ==========================================
state = {
    "model_name": None,
    "pipeline": None,
    "xgb_model": None,
    "df": None,
    "data_path": "df_ggn_covariates.csv" # Ensure this file exists
}

# ==========================================
# DATA LOADING & PREP
# ==========================================
def load_data():
    print("Loading Dataset...")
    df = pd.read_csv(state["data_path"])
    df['From Date'] = pd.to_datetime(df['From Date'])
    df = df.sort_values('From Date')
    
    # Interpolate
    for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            
    # Feature Engineering (For Hybrid)
    df['lag_24h'] = df['calibPM'].shift(24)
    df['lag_48h'] = df['calibPM'].shift(48)
    df['hour'] = df['From Date'].dt.hour
    df['month'] = df['From Date'].dt.month
    df = df.dropna().reset_index(drop=True)
    
    state["df"] = df
    
    # Train XGBoost once at startup
    print("Training XGBoost (Hybrid Component)...")
    split_idx = int(len(df) * 0.8)
    X = df[['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']]
    y = df['calibPM']
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=1)
    xgb_model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    state["xgb_model"] = xgb_model
    print("Data & XGBoost Ready.")

def load_model(variant):
    """Loads model if not already loaded"""
    if state["model_name"] == variant and state["pipeline"] is not None:
        return # Already loaded
        
    print(f"Loading Model: amazon/chronos-t5-{variant}...")
    state["pipeline"] = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{variant}",
        device_map="cpu", # Server on CPU
        torch_dtype=torch.float32
    )
    state["model_name"] = variant
    print("Model Loaded.")

# ==========================================
# API DEFINITION
# ==========================================
class ForecastRequest(BaseModel):
    model_variant: str = "base"   # tiny, mini, small, base
    context_len: int = 14         # days
    horizon: int = 4              # hours
    use_covariates: bool = False  # True = Hybrid, False = Chronos Only
    request_id: int               # To track requests

@app.on_event("startup")
async def startup_event():
    load_data()
    load_model("base") # Preload default

@app.post("/predict")
async def predict(req: ForecastRequest):
    t0 = time.perf_counter()
    
    # 1. Ensure correct model is loaded
    # (In a naive server, this blocks everything if model swapping is needed)
    load_model(req.model_variant)
    
    # 2. Prepare Data Context
    # We grab the *last* available data from our CSV to simulate "Live" forecasting
    # In a real scenario, the client sends the context, but here we use the CSV trace
    df = state["df"]
    pm_values = df['calibPM'].values
    
    # We define "Now" as the end of the dataset for simulation
    target_idx = len(df) - 1
    
    context_hours = req.context_len * 24
    start_idx = target_idx - context_hours
    
    if start_idx < 0:
        raise HTTPException(status_code=400, detail="Context length too long for data")
    
    # 3. Chronos Inference
    context_tensor = torch.tensor(pm_values[start_idx:target_idx], dtype=torch.float32)
    
    forecast = state["pipeline"].predict(
        context=context_tensor,
        prediction_length=req.horizon,
        num_samples=20,
        limit_prediction_length=False
    )
    prediction = np.median(forecast[0].numpy(), axis=0)
    
    # 4. Hybrid (Covariates) Logic
    if req.use_covariates:
        # We need "future" weather features. In this simulation, we grab them from CSV
        # (Assuming we have ground truth weather for the horizon)
        # Note: If horizon extends beyond CSV, we clip it. 
        # For simulation, we just take the last 'horizon' rows of features
        feat_cols = ['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']
        xgb_feats = df[feat_cols].iloc[target_idx-req.horizon : target_idx] 
        
        # XGBoost Prediction
        xgb_pred = state["xgb_model"].predict(xgb_feats)
        
        # Weighted Avg (Hybrid)
        prediction = (0.6 * prediction) + (0.4 * xgb_pred)

    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    
    return {
        "request_id": req.request_id,
        "forecast": prediction.tolist(),
        "latency_ms": latency,
        "model_used": req.model_variant
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)