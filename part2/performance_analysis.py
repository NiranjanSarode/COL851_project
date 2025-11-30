#!/usr/bin/env python3
import time
import argparse
import pandas as pd
import numpy as np
import torch
import psutil
import os
import threading
import xgboost as xgb
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns

# Import your cache tool
try:
    from get_cache_stats import get_cache_stats
except ImportError:
    print("Warning: get_cache_stats.py not found. Cache stats will be 0.")
    def get_cache_stats(duration=0.1):
        return {'cache_hit_rate': 0.0, 'l1_hit_rate': 0.0, 'cache_misses': 0}

try:
    from chronos import ChronosPipeline
except Exception as e:
    print("Error importing ChronosPipeline:", e)

# ==========================================
# 1. Background Monitor (From Part 2 Code)
# ==========================================
class BackgroundCacheMonitor:
    def __init__(self):
        self.current_stats = {'cache_hit_rate': 0.0, 'l1_hit_rate': 0.0, 'cache_misses': 0}
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor_loop(self):
        while self.running:
            try:
                # Short duration to not block execution
                stats = get_cache_stats(duration=0.1) 
                if stats.get('cache_references', 0) > 0:
                    self.current_stats = stats
            except Exception:
                pass
            time.sleep(0.1) # Update frequently
    
    def get_stats(self):
        return self.current_stats

cache_monitor = BackgroundCacheMonitor()

# ==========================================
# 2. Data & XGBoost Prep (Hybrid Logic)
# ==========================================
def prepare_data_and_train_xgb(filename):
    print(f"Loading and Preprocessing {filename}...")
    df = pd.read_csv(filename)
    df['From Date'] = pd.to_datetime(df['From Date'])
    df = df.sort_values('From Date')
    
    for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            
    # Feature Engineering
    df['lag_24h'] = df['calibPM'].shift(24)
    df['lag_48h'] = df['calibPM'].shift(48)
    df['hour'] = df['From Date'].dt.hour
    df['month'] = df['From Date'].dt.month
    
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = ['RH_avg', 'AT_avg', 'WS_avg', 'hour', 'month', 'lag_24h', 'lag_48h']
    
    # 80-20 Split
    split_idx = int(len(df) * 0.8)
    X = df[feature_cols]
    y = df['calibPM']
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    print("Training XGBoost...")
    # n_jobs=1 to isolate latency impact
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=1)
    xgb_model.fit(X_train, y_train)
    
    return df, xgb_model, split_idx, feature_cols

# ==========================================
# 3. Benchmark Function (Robust)
# ==========================================
def benchmark_hybrid_robust(pipeline, xgb_model, df, split_idx, feature_cols,
                           context_h, horizon, mode_name, max_windows=20):
    
    pm_values = df['calibPM'].values
    
    # Lists to store per-window metrics (Like Part 2)
    latencies = []
    rmses = []
    cpu_usages = []
    cache_stats_list = []
    
    start_test_idx = split_idx
    
    print(f"  > Benchmarking {mode_name} ({max_windows} windows)...")
    
    process = psutil.Process(os.getpid())
    
    for i in range(max_windows):
        target_start = start_test_idx + (i * horizon)
        target_end = target_start + horizon
        context_start = target_start - context_h
        
        if target_end > len(df) or context_start < 0:
            break
            
        # Prepare Data
        ctx_tensor = torch.tensor(pm_values[context_start:target_start], dtype=torch.float32)
        if xgb_model:
            xgb_features = df[feature_cols].iloc[target_start:target_end]
        ground_truth = pm_values[target_start:target_end]
        
        # --- MEASUREMENT BLOCK ---
        process.cpu_percent(interval=None) # Reset counter
        
        t0 = time.perf_counter()
        
        # 1. Chronos
        forecast = pipeline.predict(
            context=ctx_tensor,
            prediction_length=horizon,
            num_samples=20,
            limit_prediction_length=False
        )
        chronos_pred = np.median(forecast[0].numpy(), axis=0)
        
        final_pred = chronos_pred
        
        # 2. XGBoost (If Hybrid)
        if xgb_model:
            xgb_pred = xgb_model.predict(xgb_features)
            final_pred = (0.6 * chronos_pred) + (0.4 * xgb_pred)
            
        t1 = time.perf_counter()
        
        # Capture Metrics immediately after work is done
        cpu_val = process.cpu_percent(interval=None)
        cache_snapshot = cache_monitor.get_stats()
        
        # -------------------------
        
        latencies.append((t1 - t0) * 1000.0)
        cpu_usages.append(cpu_val)
        cache_stats_list.append(cache_snapshot)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(ground_truth, final_pred))
        rmses.append(rmse)
        
    # Aggregate
    if not latencies: return None

    # Calculate Cache Averages
    avg_cache_hit = mean([c['cache_hit_rate'] for c in cache_stats_list])
    avg_l1_hit = mean([c['l1_hit_rate'] for c in cache_stats_list])

    return {
        'mode': mode_name,
        'mean_latency_ms': mean(latencies),
        'throughput_fps': len(latencies) / (sum(latencies)/1000.0),
        'mean_rmse': mean(rmses),
        'cpu_percent': mean(cpu_usages),
        'cache_hit_rate': avg_cache_hit,
        'l1_hit_rate': avg_l1_hit
    }

# ==========================================
# 4. Plotting
# ==========================================
def plot_results(results_df, model_name, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    
    # We only plot the Real runs, not Warmup
    df = results_df[results_df['mode'] != 'Warmup']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Latency & RMSE (Dual Axis)
    ax1 = axes[0,0]
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df['mean_rmse'], width, label='RMSE', color='#d62728')
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, df['mean_latency_ms'], width, label='Latency', color='#1f77b4')
    
    ax1.set_ylabel('RMSE', color='#d62728', fontweight='bold')
    ax2.set_ylabel('Latency (ms)', color='#1f77b4', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['mode'])
    ax1.set_title("RMSE vs Latency")
    
    # 2. CPU Usage
    sns.barplot(data=df, x='mode', y='cpu_percent', ax=axes[0,1], palette='magma')
    axes[0,1].set_title("CPU Utilization (%)")
    
    # 3. Throughput
    sns.barplot(data=df, x='mode', y='throughput_fps', ax=axes[1,0], palette='viridis')
    axes[1,0].set_title("Throughput (FPS)")
    
    # 4. Cache Hit Rate
    sns.barplot(data=df, x='mode', y='cache_hit_rate', ax=axes[1,1], palette='Blues')
    axes[1,1].set_title("Cache Hit Rate (0-1)")
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/final_performance_metrics.png')
    print(f"Plots saved to {plot_dir}/final_performance_metrics.png")

# ==========================================
# 5. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="df_ggn_covariates.csv")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--context-days", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    
    # Prep
    df, xgb_model, split_idx, feat_cols = prepare_data_and_train_xgb(args.data_file)
    
    print(f"\nLoading Chronos ({args.model})...")
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{args.model}",
        device_map="cpu", # Force CPU for accurate perf measurement
        torch_dtype=torch.float32
    )
    
    cache_monitor.start()
    results = []
    
    # --- WARMUP ---
    print("\n[Warmup Phase] Running dummy predictions...")
    benchmark_hybrid_robust(
        pipeline, None, df, split_idx, feat_cols, 
        args.context_days*24, args.horizon, "Warmup", max_windows=5
    )
    print("Warmup done.\n")
    
    # --- RUN A: Without Covariates ---
    res_a = benchmark_hybrid_robust(
        pipeline, None, df, split_idx, feat_cols,
        args.context_days*24, args.horizon, "Without Covariates", max_windows=50
    )
    results.append(res_a)
    print(f"  RMSE: {res_a['mean_rmse']:.2f} | CPU: {res_a['cpu_percent']:.1f}%")

    # --- RUN B: With Covariates ---
    res_b = benchmark_hybrid_robust(
        pipeline, xgb_model, df, split_idx, feat_cols,
        args.context_days*24, args.horizon, "With Covariates", max_windows=50
    )
    results.append(res_b)
    print(f"  RMSE: {res_b['mean_rmse']:.2f} | CPU: {res_b['cpu_percent']:.1f}%")
    
    cache_monitor.stop()
    
    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv("part3_final_metrics.csv", index=False)
    plot_results(res_df, args.model)
    print("\nDone.")

if __name__ == "__main__":
    main()