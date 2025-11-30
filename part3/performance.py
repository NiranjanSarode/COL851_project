#!/usr/bin/env python3
"""
part3_performance_comparison.py
Benchmarks "Chronos Only" vs "Hybrid (Chronos + XGBoost)"
Captures Latency, Throughput, CPU, RAM, and Cache Hits/Misses.
"""

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
from prometheus_client import Gauge, start_http_server
import matplotlib.pyplot as plt
import seaborn as sns

# Import your cache tool
try:
    from get_cache_stats import get_cache_stats
except ImportError:
    # Fallback if file missing
    def get_cache_stats(duration=1):
        return {'cache_hit_rate': 0.0, 'l1_hit_rate': 0.0, 'cache_misses': 0, 'cache_references': 0}

try:
    from chronos import ChronosPipeline
except Exception as e:
    print("Error importing ChronosPipeline:", e)
    ChronosPipeline = None

# ========== Background Cache Monitor (Same as your code) ==========
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
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        while self.running:
            try:
                stats = get_cache_stats(duration=2) # Shorter duration for tighter loops
                if stats.get('cache_references', 0) > 0:
                    self.current_stats = stats
            except Exception:
                pass
            time.sleep(0.1)
    
    def get_stats(self):
        return self.current_stats

cache_monitor = BackgroundCacheMonitor()

# ========== 1. Data Preparation & XGBoost Training ==========
def prepare_data_and_train_xgb(filename):
    """
    Loads data, adds Lag features, splits 80/20, and trains XGBoost.
    Returns: df (processed), xgb_model, split_idx, feature_cols
    """
    print(f"Loading and Preprocessing {filename}...")
    df = pd.read_csv(filename)
    df['From Date'] = pd.to_datetime(df['From Date'])
    df = df.sort_values('From Date')
    
    # Interpolate
    for col in ['calibPM', 'RH_avg', 'AT_avg', 'WS_avg']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')
            
    # --- Feature Engineering (The "Hybrid" Logic) ---
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
    
    print("Training XGBoost for Hybrid model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=1 
        # n_jobs=1 ensures we measure the single-thread latency cost accurately
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete.")
    
    return df, xgb_model, split_idx, feature_cols

# ========== 2. Benchmarking Function (Hybrid Aware) ==========
def benchmark_hybrid(pipeline, xgb_model, df, split_idx, feature_cols,
                     context_h, horizon, mode_name, max_windows=20, num_samples=20):
    """
    Runs benchmarking for either Univariate (if xgb_model is None) 
    or Hybrid (if xgb_model is provided).
    
    *UPDATED* with Delta CPU Calculation to fix missing CPU bars.
    """
    from sklearn.metrics import mean_squared_error # Ensure import exists
    
    pm_values = df['calibPM'].values
    
    # Metrics storage
    latencies = []
    rmses = []
    cpu_usages = [] # Store CPU % for every single window
    cache_stats_list = []
    
    # We loop through the TEST set
    start_test_idx = split_idx
    num_steps = 0
    
    print(f"  > Benchmarking {mode_name}...")
    
    # Get current process for CPU tracking
    process = psutil.Process(os.getpid())
    
    for i in range(max_windows):
        # Calculate indices
        target_start = start_test_idx + (i * horizon)
        target_end = target_start + horizon
        context_start = target_start - context_h
        
        # Check bounds
        if target_end > len(df) or context_start < 0:
            break
            
        # Prepare Inputs
        ctx_tensor = torch.tensor(pm_values[context_start:target_start], dtype=torch.float32)
        
        if xgb_model:
            xgb_features = df[feature_cols].iloc[target_start:target_end]
            
        ground_truth = pm_values[target_start:target_end]
        
        # --- MEASUREMENT START ---
        # Reset CPU counter for accuracy
        process.cpu_percent(interval=None) 
        
        t_start = time.perf_counter()
        cpu_start = process.cpu_times() # Get exact Kernel CPU time
        
        # 1. Step A: Chronos Inference
        forecast = pipeline.predict(
            context=ctx_tensor,
            prediction_length=horizon,
            num_samples=num_samples,
            limit_prediction_length=False
        )
        chronos_pred = np.median(forecast[0].numpy(), axis=0)
        final_pred = chronos_pred
        
        # 2. Step B: XGBoost Inference (If Hybrid)
        if xgb_model:
            xgb_pred = xgb_model.predict(xgb_features)
            # Weighted Average (0.6 Chronos / 0.4 XGB)
            final_pred = (0.6 * chronos_pred) + (0.4 * xgb_pred)
            
        # --- MEASUREMENT END ---
        t_end = time.perf_counter()
        cpu_end = process.cpu_times()
        
        # CALCULATIONS
        # 1. Latency (Wall Clock)
        wall_time = t_end - t_start
        latencies.append(wall_time * 1000.0)
        
        # 2. CPU Percentage (Delta Method)
        # (User Time + System Time used) / Wall Time
        cpu_time_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        
        if wall_time > 0.000001:
            # Multiplied by 100 for percentage
            # Dividing by logical_cpus is optional, usually we want total system load relative to 1 core
            cpu_val = (cpu_time_used / wall_time) * 100.0 
        else:
            cpu_val = 0.0
        cpu_usages.append(cpu_val)
        
        # 3. Cache Stats
        cache_stats_list.append(cache_monitor.get_stats())
        
        # 4. RMSE
        rmse = np.sqrt(mean_squared_error(ground_truth, final_pred))
        rmses.append(rmse)
        
        num_steps += 1
        
    # Aggregate Metrics (Averaging over all windows)
    avg_latency = mean(latencies) if latencies else 0
    throughput = num_steps / (sum(latencies)/1000.0) if latencies else 0
    avg_rmse = mean(rmses) if rmses else 0
    avg_cpu = mean(cpu_usages) if cpu_usages else 0 # Average of the per-window CPU
    
    # Cache Aggregation
    if cache_stats_list:
        avg_cache_hit = mean([c['cache_hit_rate'] for c in cache_stats_list])
        avg_l1_hit = mean([c['l1_hit_rate'] for c in cache_stats_list])
        total_misses = sum([c['cache_misses'] for c in cache_stats_list])
    else:
        avg_cache_hit, avg_l1_hit, total_misses = 0, 0, 0
        
    # Memory snapshot (Memory doesn't fluctuate as fast as CPU, so snapshot is fine)
    mem = psutil.virtual_memory().used / (1024 * 1024)

    return {
        'mode': mode_name,
        'mean_latency_ms': avg_latency,
        'throughput_fps': throughput,
        'mean_rmse': avg_rmse,
        'cpu_percent': avg_cpu, # Using the calculated average
        'mem_mb': mem,
        'cache_hit_rate': avg_cache_hit,
        'l1_hit_rate': avg_l1_hit,
        'cache_misses': total_misses
    }

# ========== 3. Plotting ==========
def plot_comparison(results_df, model_name, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Trade-off Plot (RMSE vs Latency)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    modes = results_df['mode'].unique()
    x = np.arange(len(modes))
    width = 0.35
    
    # Metric 1: RMSE (Bar)
    ax1.bar(x - width/2, results_df['mean_rmse'], width, label='RMSE', color='#d62728', alpha=0.7)
    ax1.set_ylabel('RMSE (Lower is Better)', color='#d62728', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=12, fontweight='bold')
    
    # Metric 2: Latency (Bar)
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, results_df['mean_latency_ms'], width, label='Latency', color='#1f77b4', alpha=0.7)
    ax2.set_ylabel('Latency (ms) (Lower is Better)', color='#1f77b4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    
    plt.title(f'Trade-off: Accuracy vs Speed\nModel: {model_name}', fontsize=14)
    fig.tight_layout()
    plt.savefig(f'{plot_dir}/tradeoff_rmse_latency.png')
    
    # 2. Performance Metrics Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Throughput
    sns.barplot(data=results_df, x='mode', y='throughput_fps', ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('Throughput (Predictions/sec)')
    
    # CPU
    sns.barplot(data=results_df, x='mode', y='cpu_percent', ax=axes[0,1], palette='magma')
    axes[0,1].set_title('CPU Utilization (%)')
    
    # Cache Hit Rate
    sns.barplot(data=results_df, x='mode', y='cache_hit_rate', ax=axes[1,0], palette='Blues')
    axes[1,0].set_title('Cache Hit Rate (0-1)')
    
    # L1 Hit Rate
    sns.barplot(data=results_df, x='mode', y='l1_hit_rate', ax=axes[1,1], palette='Greens')
    axes[1,1].set_title('L1 Cache Hit Rate (0-1)')
    
    plt.suptitle(f'System Performance: With vs Without Covariates ({model_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/system_metrics_comparison.png')
    print(f"Plots saved to {plot_dir}/")

# ========== 4. Main ==========
# Copy and paste this OVER the existing main() function in your script

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="df_ggn_covariates.csv")
    parser.add_argument("--model", type=str, default="base", help="Model variant from Part 1")
    parser.add_argument("--context-days", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu") 
    args = parser.parse_args()
    
    # 1. Prepare Data & Models
    df, xgb_model, split_idx, feat_cols = prepare_data_and_train_xgb(args.data_file)
    
    print(f"\nLoading Chronos ({args.model})...")
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{args.model}",
        device_map=args.device,
        torch_dtype=torch.float32
    )
    
    # Start Cache Monitor
    cache_monitor.start()
    
    # =========================================================
    # NEW: WARMUP PHASE (Crucial for fair comparison)
    # =========================================================
    print("\nWarmup phase (ignoring results)...")
    # Run a few dummy inferences to wake up the CPU and fill caches
    benchmark_hybrid(
        pipeline=pipeline, xgb_model=None, df=df, split_idx=split_idx,
        feature_cols=feat_cols, context_h=args.context_days*24,
        horizon=args.horizon, mode_name="Warmup", max_windows=5
    )
    print("Warmup complete. Starting real measurements...\n")
    # =========================================================

    results = []
    
    # 2. Run Benchmark A: Without Covariates
    res_no_cov = benchmark_hybrid(
        pipeline=pipeline,
        xgb_model=None, 
        df=df,
        split_idx=split_idx,
        feature_cols=feat_cols,
        context_h=args.context_days*24,
        horizon=args.horizon,
        mode_name="Without Covariates",
        max_windows=50 # Increased windows to average out noise
    )
    results.append(res_no_cov)
    print(f"\n[Without Covariates] RMSE: {res_no_cov['mean_rmse']:.2f} | Latency: {res_no_cov['mean_latency_ms']:.1f}ms")
    
    # 3. Run Benchmark B: With Covariates
    res_with_cov = benchmark_hybrid(
        pipeline=pipeline,
        xgb_model=xgb_model, 
        df=df,
        split_idx=split_idx,
        feature_cols=feat_cols,
        context_h=args.context_days*24,
        horizon=args.horizon,
        mode_name="With Covariates",
        max_windows=50 # Increased windows here too
    )
    results.append(res_with_cov)
    print(f"\n[With Covariates]    RMSE: {res_with_cov['mean_rmse']:.2f} | Latency: {res_with_cov['mean_latency_ms']:.1f}ms")
    
    cache_monitor.stop()
    
    # 4. Save & Plot
    res_df = pd.DataFrame(results)
    res_df.to_csv("part3_final_performance.csv", index=False)
    
    plot_comparison(res_df, args.model)
    print("\nComparison Complete!")

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    main()