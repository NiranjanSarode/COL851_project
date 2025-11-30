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
from torch.quantization import quantize_dynamic

app = FastAPI(title="Configurable Chronos Server")

# ==========================================
# 🎛️ OPTIMIZATION SWITCHES (TOGGLE THESE)
# ==========================================
ENABLE_CACHING      = False   # Keep models in RAM (False = Reload from disk every time)
ENABLE_BATCHING     = False   # Group requests (False = Process 1-by-1 sequentially)
ENABLE_QUANTIZATION = False   # Use INT8 weights (False = Use FP32 original weights)

# ==========================================
# CONFIGURATION
# ==========================================
MODELS_TO_LOAD = ['tiny', 'small', 'base']
BATCH_SIZE = 8          
BATCH_TIMEOUT = 0.1     
DATA_PATH = "df_ggn_covariates.csv" 

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
# MODEL LOADING HELPER
# ==========================================
def load_model_instance(variant):
    """
    Loads a single model instance from disk.
    Applies Quantization if the switch is ON.
    """
    print(f" [System] Loading {variant} from disk... ", end="", flush=True)
    t0 = time.time()
    
    # 1. Load Standard Model
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{variant}",
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # 2. Apply Quantization (If Switch is ON)
    if ENABLE_QUANTIZATION:
        pipeline.model = quantize_dynamic(
            pipeline.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
    
    print(f"Done ({time.time()-t0:.2f}s) | Quantization={'ON' if ENABLE_QUANTIZATION else 'OFF'}")
    return pipeline

def get_pipeline(variant):
    """
    Smart Getter:
    - If Caching is ON: Returns the pre-loaded model from RAM.
    - If Caching is OFF: Loads the model from scratch (simulating naive server).
    """
    # Case A: Caching is ON and model exists
    if ENABLE_CACHING and variant in state["pipelines"]:
        return state["pipelines"][variant]
    
    # Case B: Caching is OFF (or model not loaded yet) -> Load fresh
    return load_model_instance(variant)

# ==========================================
# STARTUP RESOURCE LOADING
# ==========================================
def load_resources():
    # 1. Load Data (Always needed)
    print("Loading Data & Training XGBoost...")
    df = pd.read_csv(DATA_PATH)
    df['From Date'] = pd.to_datetime(df['From Date'])
    df = df.sort_values('From Date')
    for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
    
    df['lag_24h'] = df['calibPM'].shift(24)
    df['lag_48h'] = df['calibPM'].shift(48)
    df['hour'] = df['From Date'].dt.hour
    df['month'] = df['From Date'].dt.month
    df = df.dropna().reset_index(drop=True)
    state["df"] = df

    split_idx = int(len(df) * 0.8)
    X = df[['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']]
    y = df['calibPM']
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=1)
    xgb_model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
    state["xgb_model"] = xgb_model

    # 2. Pre-load Models (ONLY IF Caching is ON)
    if ENABLE_CACHING:
        print("Caching ENABLED: Pre-loading all models...")
        for variant in MODELS_TO_LOAD:
            state["pipelines"][variant] = load_model_instance(variant)
    else:
        print("Caching DISABLED: Models will be loaded on-demand per request.")

    # 3. Setup Queues (ONLY IF Batching is ON)
    if ENABLE_BATCHING:
        print("Batching ENABLED: Starting background processors...")
        for variant in MODELS_TO_LOAD:
            state["queues"][variant] = asyncio.Queue()

# ==========================================
# BATCH PROCESSOR (Runs if Batching=True)
# ==========================================
async def batch_processor(variant):
    queue = state["queues"][variant]
    print(f"Started Batch Processor for {variant}")
    
    while True:
        # --- 1. Collect Batch ---
        batch = []
        try:
            item = await queue.get()
            batch.append(item)
            
            expiry = time.time() + BATCH_TIMEOUT
            while len(batch) < BATCH_SIZE:
                remaining = expiry - time.time()
                if remaining <= 0: break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
        except Exception:
            continue

        if not batch: continue

        # --- 2. Process Batch ---
        reqs, futures = zip(*batch)
        
        # Get Pipeline (Cached or Fresh)
        pipeline = get_pipeline(variant)

        try:
            # Prepare Data
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
            
            # Inference
            forecast = pipeline.predict(
                context=contexts, 
                prediction_length=max_horizon,
                num_samples=20,
                limit_prediction_length=False
            )
            
            # Post-processing & XGBoost Hybrid
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
                if not fut.done(): fut.set_exception(e)

        # Cleanup if Caching is OFF (Simulate discard)
        if not ENABLE_CACHING:
            del pipeline
            import gc
            gc.collect()

# ==========================================
# SEQUENTIAL PROCESSOR (Runs if Batching=False)
# ==========================================
def run_sequential_prediction(req):
    # Get Pipeline (Cached or Fresh)
    pipeline = get_pipeline(req.model_variant)
    
    df = state["df"]
    pm_values = df['calibPM'].values
    last_idx = len(df) - 1
    
    ctx_len_hours = req.context_len * 24
    start = last_idx - ctx_len_hours
    
    # Prepare single item
    context_tensor = torch.tensor(pm_values[start:last_idx], dtype=torch.float32)
    
    forecast = pipeline.predict(
        context=context_tensor,
        prediction_length=req.horizon,
        num_samples=20,
        limit_prediction_length=False
    )
    
    pred_seq = np.median(forecast[0].numpy(), axis=0)
    final_pred = pred_seq
    
    if req.use_covariates:
        feat_cols = ['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']
        xgb_feats = df[feat_cols].iloc[last_idx-req.horizon : last_idx]
        xgb_res = state["xgb_model"].predict(xgb_feats)
        final_pred = (0.6 * final_pred) + (0.4 * xgb_res)

    # Cleanup if Caching is OFF
    if not ENABLE_CACHING:
        del pipeline
        import gc
        gc.collect()
        
    return final_pred.tolist()

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
    if ENABLE_BATCHING:
        for variant in MODELS_TO_LOAD:
            asyncio.create_task(batch_processor(variant))

@app.post("/predict")
async def predict(req: ForecastRequest):
    t0 = time.perf_counter()
    
    # CASE 1: Batching is ON
    if ENABLE_BATCHING:
        if req.model_variant not in state["queues"]:
            return {"error": "Model variant not supported"}
            
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await state["queues"][req.model_variant].put((req, future))
        result = await future
    
    # CASE 2: Batching is OFF (Sequential)
    else:
        result = run_sequential_prediction(req)
    
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    
    return {
        "request_id": req.request_id,
        "forecast": result,
        "latency_ms": latency,
        "model_used": req.model_variant,
        "config": {
            "caching": ENABLE_CACHING,
            "batching": ENABLE_BATCHING,
            "quantization": ENABLE_QUANTIZATION
        }
    }

if __name__ == "__main__":
    # Print status clearly
    print("\n" + "="*50)
    print("STARTING CHRONOS SERVER")
    print("="*50)
    print(f"Caching:      {'ON' if ENABLE_CACHING else 'OFF (High Latency)'}")
    print(f"Batching:     {'ON' if ENABLE_BATCHING else 'OFF (Low Throughput)'}")
    print(f"Quantization: {'ON' if ENABLE_QUANTIZATION else 'OFF (High Memory)'}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)