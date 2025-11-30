import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from chronos import ChronosPipeline
import time
import asyncio
from collections import defaultdict

# IMPORT QUANTIZATION TOOL
from torch.quantization import quantize_dynamic

app = FastAPI(title="Optimized Chronos Server (Batching + Caching + Quantization)")

# ==========================================
# CONFIGURATION
# ==========================================
MODELS_TO_LOAD = ['tiny', 'small', 'base']
BATCH_SIZE = 8          
BATCH_TIMEOUT = 0.1     
DATA_PATH = "df_ggn_covariates.csv" # Ensure this matches your file

# ==========================================
# GLOBAL STATE 
# ==========================================
state = {
    "pipelines": {},    
    "xgb_model": None,
    "df": None,
    "queues": {}        
}

# ==========================================
# 1. OPTIMIZATIONS: Caching + Quantization
# ==========================================
def load_resources():
    # A. Load Data
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    df['From Date'] = pd.to_datetime(df['From Date'])
    df = df.sort_values('From Date')
    for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
    
    # Feature Eng for XGB
    df['lag_24h'] = df['calibPM'].shift(24)
    df['lag_48h'] = df['calibPM'].shift(48)
    df['hour'] = df['From Date'].dt.hour
    df['month'] = df['From Date'].dt.month
    df = df.dropna().reset_index(drop=True)
    state["df"] = df

    # B. Load XGBoost
    print("Training XGBoost...")
    split_idx = int(len(df) * 0.8)
    X = df[['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']]
    y = df['calibPM']
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=1)
    xgb_model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    state["xgb_model"] = xgb_model

    # C. Load + QUANTIZE Chronos Models
    for variant in MODELS_TO_LOAD:
        print(f"Loading {variant}...", end="", flush=True)
        
        # 1. Load standard model
        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{variant}",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # =========================================================
        # NEW: APPLY INT8 QUANTIZATION HERE
        # =========================================================
        print(f" Quantizing...", end="", flush=True)
        
        # We perform Dynamic Quantization on the internal T5 model
        # This converts Linear layers from FP32 -> INT8
        pipeline.model = quantize_dynamic(
            pipeline.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        # =========================================================

        state["pipelines"][variant] = pipeline
        state["queues"][variant] = asyncio.Queue()
        print(" Done (INT8 Ready).")

# ==========================================
# 2. OPTIMIZATION: Dynamic Batching Logic
# ==========================================
async def batch_processor(variant):
    queue = state["queues"][variant]
    pipeline = state["pipelines"][variant]
    
    print(f"Started Batch Processor for {variant}")
    
    while True:
        # 1. Collect Batch
        batch = []
        try:
            item = await queue.get()
            batch.append(item)
            
            expiry = time.time() + BATCH_TIMEOUT
            while len(batch) < BATCH_SIZE:
                remaining = expiry - time.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            print(f"Queue Error: {e}")
            continue

        if not batch:
            continue

        # 2. Process Batch
        reqs, futures = zip(*batch)
        
        df = state["df"]
        pm_values = df['calibPM'].values
        last_idx = len(df) - 1
        
        contexts = []
        horizons = []
        
        for r in reqs:
            ctx_len_hours = r.context_len * 24
            start = last_idx - ctx_len_hours
            contexts.append(torch.tensor(pm_values[start:last_idx], dtype=torch.float32))
            horizons.append(r.horizon)
        
        max_horizon = max(horizons)
        
        try:
            # Run Batch Inference (Now using the Quantized Model automatically)
            forecast = pipeline.predict(
                context=contexts, 
                prediction_length=max_horizon,
                num_samples=20,
                limit_prediction_length=False
            )
            
            for i, f in enumerate(forecast):
                pred_seq = np.median(f.numpy(), axis=0)
                req_horizon = reqs[i].horizon
                final_pred = pred_seq[:req_horizon]
                
                if reqs[i].use_covariates:
                    feat_cols = ['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']
                    xgb_feats = df[feat_cols].iloc[last_idx-req_horizon : last_idx]
                    xgb_res = state["xgb_model"].predict(xgb_feats)
                    final_pred = (0.6 * final_pred) + (0.4 * xgb_res)
                
                futures[i].set_result(final_pred.tolist())
                
        except Exception as e:
            print(f"Batch Error: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

# ==========================================
# API HANDLERS
# ==========================================
class ForecastRequest(BaseModel):
    model_variant: str = "base"
    context_len: int = 14
    horizon: int = 4
    use_covariates: bool = False
    request_id: int

@app.on_event("startup")
async def startup_event():
    load_resources()
    for variant in MODELS_TO_LOAD:
        asyncio.create_task(batch_processor(variant))

@app.post("/predict")
async def predict(req: ForecastRequest):
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    
    if req.model_variant not in state["queues"]:
        return {"error": "Model not loaded"}
        
    await state["queues"][req.model_variant].put((req, future))
    result = await future
    
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    
    return {
        "request_id": req.request_id,
        "forecast": result,
        "latency_ms": latency,
        "model_used": req.model_variant,
        "optimization": "Batching+Caching+Quantization"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)