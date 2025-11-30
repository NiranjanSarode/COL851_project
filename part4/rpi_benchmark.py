#!/usr/bin/env python3
"""
Part 4: Raspberry Pi Performance Benchmarking for PM2.5 Forecasting
=====================================================================
Requirements satisfied:
1. Run forecasting with and without covariates on RPi
2. Compare performance metrics with laptop measurements
3. Pin process to 1, 2, 3, 4 cores and measure effect
4. Measure CPU temperature every minute for 30 mins continuous inference
5. Calculate RMSE and compare with/without covariates
"""

import os
import time
import psutil
import pandas as pd
import numpy as np
import torch
import logging
import sys
import gc
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Headless backend for RPi
import matplotlib.pyplot as plt

from chronos import ChronosPipeline

# Optional: CatBoost for covariate modeling (install: pip install catboost)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

# =============================================================================
# CONFIGURATION - Adjust based on your Part 1 best hyperparameters
# =============================================================================
CONFIG = {
    # Data files
    "data_file": "gurgaon.csv",
    "laptop_results_file": "laptop_results.csv",  # From Part 2
    
    # Model settings (use your best from Part 1)
    "model_name": "amazon/chronos-t5-tiny",  # Smallest for RPi 3
    
    # Hyperparameters (REPLACE WITH YOUR PART 1 BEST VALUES)
    # NOTE: Reduced defaults for RPi 3 - increase if you have RPi 4
    "context_length_hours": 48,   # 2 days (heavily reduced for RPi 3 speed)
    "forecast_horizon": 24,       # 24 hours
    "num_samples": 5,             # Minimal samples for faster inference
    
    # Test configuration
    "train_test_split": 0.8,      # 80-20 split for covariate model
    "stress_test_duration_min": 30,
    "temp_log_interval_sec": 60,
    "inference_reps_per_test": 5,  # Reduced from 10 for RPi 3
    
    # Output files
    "results_file": "rpi_results.csv",
    "thermal_log_file": "rpi_thermal_log.csv",
    "rmse_results_file": "rpi_rmse_results.csv",
    "comparison_file": "rpi_vs_laptop_comparison.csv",
    "plots_file": "rpi_part4_plots.png",
}

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rpi_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


class RPiPM25Benchmark:
    """
    Complete Raspberry Pi benchmarking suite for PM2.5 forecasting.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self.pipeline = None
        self.covariate_model = None
        
        # Results storage
        self.performance_results = []
        self.thermal_log = []
        self.rmse_results = []
        
        # Data placeholders
        self.pm_data = None
        self.weather_data = None
        self.timestamps = None
        self.train_idx = None
        self.test_idx = None
        
        # System info
        self.is_rpi = self._detect_rpi()
        self.num_cores = psutil.cpu_count(logical=True)
        
        logger.info(f"System: {'Raspberry Pi' if self.is_rpi else 'Standard Linux/PC'}")
        logger.info(f"Available CPU cores: {self.num_cores}")
    
    def _detect_rpi(self):
        """Detect if running on Raspberry Pi."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read()
                return "Raspberry Pi" in model
        except:
            return os.path.exists("/sys/class/thermal/thermal_zone0/temp")
    
    # =========================================================================
    # SYSTEM METRICS COLLECTION
    # =========================================================================
    
    def get_cpu_temp(self):
        """Read CPU temperature from hardware sensor."""
        try:
            if self.is_rpi:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    return float(f.read().strip()) / 1000.0
            else:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    return temps['coretemp'][0].current
                elif 'cpu_thermal' in temps:
                    return temps['cpu_thermal'][0].current
                return 0.0
        except Exception as e:
            logger.warning(f"Could not read temperature: {e}")
            return 0.0
    
    def get_throttle_status(self):
        """Check for thermal throttling (RPi specific)."""
        if not self.is_rpi:
            return {"throttled": False, "raw": "N/A"}
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"], 
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout.strip()
            # throttled=0x0 means no throttling
            # Any other value indicates throttling occurred
            hex_val = output.split("=")[1] if "=" in output else "0x0"
            is_throttled = hex_val != "0x0"
            return {"throttled": is_throttled, "raw": hex_val}
        except:
            return {"throttled": False, "raw": "Error"}
    
    def collect_system_metrics(self):
        """Collect comprehensive system metrics."""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_percent_per_core": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory_rss_mb": process.memory_info().rss / (1024 ** 2),
            "memory_percent": process.memory_percent(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "temp_c": self.get_cpu_temp(),
            "throttle_status": self.get_throttle_status(),
        }
    
    def set_cpu_affinity(self, n_cores):
        """Pin process to specific CPU cores."""
        try:
            process = psutil.Process()
            available = list(range(min(n_cores, self.num_cores)))
            process.cpu_affinity(available)
            logger.info(f"Process pinned to cores: {available}")
            time.sleep(0.5)  # Let scheduler settle
            return True
        except Exception as e:
            logger.error(f"Failed to set CPU affinity: {e}")
            return False
    
    # =========================================================================
    # DATA LOADING AND PREPROCESSING
    # =========================================================================
    
    def load_data(self):
        """Load and preprocess the PM2.5 and weather data."""
        logger.info(f"Loading data from {self.config['data_file']}...")
        
        try:
            df = pd.read_csv(self.config['data_file'])
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config['data_file']}")
            sys.exit(1)
        
        # Column mapping (based on project description):
        # Col 1: Timestamp, Col 2: RH, Col 3: Temp, Col 4: Wind Speed, Col 5: PM2.5
        self.timestamps = pd.to_datetime(df.iloc[:, 0])
        
        # Extract and clean data
        self.weather_data = df.iloc[:, 1:4].values.astype(np.float32)  # RH, Temp, Wind
        self.pm_data = df.iloc[:, 4].values.astype(np.float32)         # PM2.5
        
        # Handle NaN values
        self.weather_data = np.nan_to_num(self.weather_data, nan=0.0)
        self.pm_data = np.nan_to_num(self.pm_data, nan=np.nanmean(self.pm_data))
        
        # Create train/test split indices
        n_samples = len(self.pm_data)
        split_idx = int(n_samples * self.config['train_test_split'])
        self.train_idx = (0, split_idx)
        self.test_idx = (split_idx, n_samples)
        
        logger.info(f"Data loaded: {n_samples} samples")
        logger.info(f"Train: {self.train_idx[0]}-{self.train_idx[1]}, Test: {self.test_idx[0]}-{self.test_idx[1]}")
        logger.info(f"PM2.5 range: [{self.pm_data.min():.1f}, {self.pm_data.max():.1f}]")
        
        return True
    
    def load_model(self):
        """Load the Chronos forecasting model."""
        logger.info(f"Loading model: {self.config['model_name']}...")
        logger.info("  This may take 5-10 minutes on RPi 3 (first run downloads model)...")
        
        load_start = time.time()
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.config['model_name'],
                device_map=self.device,
                torch_dtype=torch.float32
            )
            load_time = time.time() - load_start
            logger.info(f"Chronos model loaded successfully in {load_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def train_covariate_model(self):
        """Train CatBoost model for covariate-based correction."""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available - using simple linear regression fallback")
            return self._train_simple_covariate_model()
        
        logger.info("Training CatBoost covariate model on 80% training data...")
        
        # Prepare training data
        # Features: weather data + lagged PM values
        context_len = min(24, self.config['context_length_hours'])  # Use last 24h as features
        
        X_train = []
        y_train = []
        
        train_start, train_end = self.train_idx
        
        for i in range(train_start + context_len, train_end):
            # Features: current weather + past PM values
            weather_features = self.weather_data[i]  # RH, Temp, Wind at time i
            pm_lag_features = self.pm_data[i-context_len:i:6]  # Sampled past PM (every 6h)
            
            features = np.concatenate([weather_features, pm_lag_features])
            X_train.append(features)
            y_train.append(self.pm_data[i])
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        # Train CatBoost
        self.covariate_model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            thread_count=self.num_cores
        )
        
        self.covariate_model.fit(X_train, y_train)
        
        # Calculate training RMSE
        train_pred = self.covariate_model.predict(X_train)
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        logger.info(f"CatBoost training RMSE: {train_rmse:.2f}")
        
        return True
    
    def _train_simple_covariate_model(self):
        """Fallback: Simple linear regression for covariates."""
        from sklearn.linear_model import Ridge
        
        logger.info("Training Ridge regression covariate model...")
        
        context_len = min(24, self.config['context_length_hours'])
        X_train, y_train = [], []
        
        train_start, train_end = self.train_idx
        
        for i in range(train_start + context_len, train_end):
            weather_features = self.weather_data[i]
            pm_lag_features = self.pm_data[i-context_len:i:6]
            features = np.concatenate([weather_features, pm_lag_features])
            X_train.append(features)
            y_train.append(self.pm_data[i])
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        
        self.covariate_model = Ridge(alpha=1.0)
        self.covariate_model.fit(X_train, y_train)
        
        return True
    
    # =========================================================================
    # FORECASTING METHODS
    # =========================================================================
    
    def forecast_without_covariates(self, context_start_idx, verbose=True):
        """
        Standard Chronos forecasting using only PM2.5 history.
        Returns: (predictions, ground_truth, inference_time)
        """
        context_len = self.config['context_length_hours']
        horizon = self.config['forecast_horizon']
        
        # Ensure we have enough data
        context_end = context_start_idx
        context_start = max(0, context_end - context_len)
        
        # Get context and ground truth
        context = self.pm_data[context_start:context_end]
        ground_truth = self.pm_data[context_end:context_end + horizon]
        
        if len(ground_truth) < horizon:
            return None, None, 0
        
        # Create tensor
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        if verbose:
            logger.info(f"    [Chronos] Starting inference (context={len(context)}, horizon={horizon}, samples={self.config['num_samples']})...")
        
        # Run inference with timing
        gc.collect()
        t_start = time.perf_counter()
        
        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=horizon,
            num_samples=self.config['num_samples']
        )
        
        t_end = time.perf_counter()
        inference_time = t_end - t_start
        
        if verbose:
            logger.info(f"    [Chronos] Inference complete in {inference_time:.2f}s")
        
        # Extract median prediction
        # Note: torch.median returns (values, indices) namedtuple when dim is specified
        predictions = np.median(forecast.numpy(), axis=1).flatten()
        
        return predictions, ground_truth, inference_time
    
    def forecast_with_covariates(self, context_start_idx, verbose=True):
        """
        Forecasting with meteorological covariates using hybrid approach:
        1. Base forecast from Chronos
        2. Correction from CatBoost using weather data
        Returns: (predictions, ground_truth, inference_time)
        """
        context_len = self.config['context_length_hours']
        horizon = self.config['forecast_horizon']
        
        context_end = context_start_idx
        context_start = max(0, context_end - context_len)
        
        context = self.pm_data[context_start:context_end]
        ground_truth = self.pm_data[context_end:context_end + horizon]
        future_weather = self.weather_data[context_end:context_end + horizon]
        
        if len(ground_truth) < horizon or len(future_weather) < horizon:
            return None, None, 0
        
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        if verbose:
            logger.info(f"    [Chronos+Cov] Starting inference...")
        
        gc.collect()
        t_start = time.perf_counter()
        
        # Step 1: Base Chronos forecast
        base_forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=horizon,
            num_samples=self.config['num_samples']
        )
        # Chronos returns shape (batch, num_samples, horizon) - take median across samples
        base_predictions = np.median(base_forecast.numpy(), axis=1).flatten()
        
        if verbose:
            logger.info(f"    [Chronos+Cov] Base forecast done, applying covariate correction...")
        
        # Step 2: Covariate-based correction
        if self.covariate_model is not None:
            corrected_predictions = []
            feature_context_len = min(24, context_len)
            
            for h in range(horizon):
                # Build features for this forecast step
                weather_feat = future_weather[h]  # "Perfect" weather forecast
                
                # Use predicted PM for lag features
                if h == 0:
                    pm_lags = self.pm_data[context_end-feature_context_len:context_end:6]
                else:
                    recent_pm = list(self.pm_data[context_end-feature_context_len+h:context_end:6])
                    recent_pm.extend(corrected_predictions[-min(h, 4):])
                    pm_lags = np.array(recent_pm[-4:])  # Last 4 values
                
                # Ensure correct feature size
                if len(pm_lags) < 4:
                    pm_lags = np.pad(pm_lags, (4 - len(pm_lags), 0), mode='edge')
                
                features = np.concatenate([weather_feat, pm_lags[:4]]).reshape(1, -1)
                
                # Get covariate prediction
                cov_pred = self.covariate_model.predict(features)[0]
                
                # Blend: weighted average of Chronos and covariate model
                blended = 0.6 * base_predictions[h] + 0.4 * cov_pred
                corrected_predictions.append(blended)
            
            predictions = np.array(corrected_predictions)
        else:
            predictions = base_predictions
        
        t_end = time.perf_counter()
        inference_time = t_end - t_start
        
        return predictions, ground_truth, inference_time
    
    # =========================================================================
    # RMSE CALCULATION
    # =========================================================================
    
    def calculate_rmse(self, predictions, ground_truth):
        """Calculate Root Mean Square Error."""
        if predictions is None or ground_truth is None:
            return np.nan
        return np.sqrt(np.mean((predictions - ground_truth) ** 2))
    
    def evaluate_rmse_on_test_set(self, use_covariates=False):
        """
        Evaluate RMSE across all test windows.
        Returns list of (test_idx, rmse, predictions, ground_truth)
        """
        horizon = self.config['forecast_horizon']
        test_start, test_end = self.test_idx
        
        results = []
        
        # Slide through test set with step = horizon (non-overlapping windows)
        for idx in range(test_start, test_end - horizon, horizon):
            if use_covariates:
                preds, gt, _ = self.forecast_with_covariates(idx)
            else:
                preds, gt, _ = self.forecast_without_covariates(idx)
            
            if preds is not None:
                rmse = self.calculate_rmse(preds, gt)
                results.append({
                    'test_idx': idx,
                    'rmse': rmse,
                    'predictions': preds.tolist(),
                    'ground_truth': gt.tolist(),
                    'use_covariates': use_covariates
                })
        
        return results
    
    # =========================================================================
    # BENCHMARK SUITE
    # =========================================================================
    
    def run_core_scaling_benchmark(self):
        """
        Benchmark performance across different core counts.
        Tests both with and without covariates.
        """
        logger.info("=" * 60)
        logger.info("RUNNING CORE SCALING BENCHMARK")
        logger.info("=" * 60)
        
        horizon = self.config['forecast_horizon']
        test_start, _ = self.test_idx
        test_idx = test_start + 100  # Fixed test point
        
        core_counts = [1, 2, 3, 4] if self.num_cores >= 4 else list(range(1, self.num_cores + 1))
        
        for n_cores in core_counts:
            for use_cov in [False, True]:
                mode = "With Covariates" if use_cov else "No Covariates"
                logger.info(f"\nTesting: {n_cores} cores, {mode}")
                
                self.set_cpu_affinity(n_cores)
                
                # Warmup
                logger.info(f"  Starting warmup inference...")
                warmup_start = time.perf_counter()
                if use_cov:
                    self.forecast_with_covariates(test_idx)
                else:
                    self.forecast_without_covariates(test_idx)
                warmup_time = time.perf_counter() - warmup_start
                logger.info(f"  Warmup complete in {warmup_time:.1f}s")
                
                # Measurement loop
                latencies = []
                metrics_samples = []
                rmse_values = []
                
                num_reps = self.config['inference_reps_per_test']
                for rep in range(num_reps):
                    logger.info(f"  Inference {rep+1}/{num_reps}...")
                    # Collect pre-inference metrics
                    pre_metrics = self.collect_system_metrics()
                    
                    # Run inference
                    if use_cov:
                        preds, gt, latency = self.forecast_with_covariates(test_idx)
                    else:
                        preds, gt, latency = self.forecast_without_covariates(test_idx)
                    
                    # Collect post-inference metrics
                    post_metrics = self.collect_system_metrics()
                    
                    latencies.append(latency)
                    metrics_samples.append(post_metrics)
                    
                    if preds is not None:
                        rmse_values.append(self.calculate_rmse(preds, gt))
                    
                    logger.info(f"    Completed in {latency:.1f}s (Temp: {post_metrics['temp_c']:.1f}°C)")
                
                gc.collect()
                
                # Aggregate results
                result = {
                    "cores": n_cores,
                    "mode": mode,
                    "latency_mean_s": np.mean(latencies),
                    "latency_std_s": np.std(latencies),
                    "throughput_hz": 1.0 / np.mean(latencies),
                    "cpu_percent": np.mean([m['cpu_percent'] for m in metrics_samples]),
                    "memory_mb": np.mean([m['memory_rss_mb'] for m in metrics_samples]),
                    "temp_c": np.mean([m['temp_c'] for m in metrics_samples]),
                    "rmse_mean": np.mean(rmse_values) if rmse_values else np.nan,
                }
                
                self.performance_results.append(result)
                
                logger.info(f"  Latency: {result['latency_mean_s']:.3f}s ± {result['latency_std_s']:.3f}s")
                logger.info(f"  Throughput: {result['throughput_hz']:.2f} inf/s")
                logger.info(f"  CPU: {result['cpu_percent']:.1f}%, Mem: {result['memory_mb']:.1f}MB, Temp: {result['temp_c']:.1f}°C")
                logger.info(f"  RMSE: {result['rmse_mean']:.2f}")
                
                time.sleep(2)  # Cool down between tests
    
    def run_thermal_stress_test(self):
        """
        Run continuous inference for 30 minutes, logging temperature every minute.
        Uses maximum load: all cores + covariates.
        """
        logger.info("=" * 60)
        logger.info(f"RUNNING {self.config['stress_test_duration_min']} MIN THERMAL STRESS TEST")
        logger.info("=" * 60)
        
        # Max load configuration
        self.set_cpu_affinity(self.num_cores)
        
        test_start, _ = self.test_idx
        test_idx = test_start + 100
        
        duration_sec = self.config['stress_test_duration_min'] * 60
        log_interval = self.config['temp_log_interval_sec']
        
        start_time = time.time()
        last_log_time = start_time
        inference_count = 0
        
        logger.info(f"Starting stress test at {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"Initial temperature: {self.get_cpu_temp():.1f}°C")
        
        while (time.time() - start_time) < duration_sec:
            # Run inference (with covariates for max load)
            self.forecast_with_covariates(test_idx)
            inference_count += 1
            
            # Log temperature every minute
            current_time = time.time()
            if (current_time - last_log_time) >= log_interval:
                elapsed_min = (current_time - start_time) / 60.0
                metrics = self.collect_system_metrics()
                
                log_entry = {
                    "time_min": elapsed_min,
                    "temp_c": metrics['temp_c'],
                    "cpu_percent": metrics['cpu_percent'],
                    "memory_mb": metrics['memory_rss_mb'],
                    "throttled": metrics['throttle_status']['throttled'],
                    "throttle_raw": metrics['throttle_status']['raw'],
                    "inference_count": inference_count,
                }
                
                self.thermal_log.append(log_entry)
                
                logger.info(
                    f"  [{elapsed_min:.1f} min] Temp: {metrics['temp_c']:.1f}°C, "
                    f"CPU: {metrics['cpu_percent']:.0f}%, "
                    f"Throttled: {metrics['throttle_status']['throttled']}, "
                    f"Inferences: {inference_count}"
                )
                
                last_log_time = current_time
                gc.collect()
        
        # Final stats
        total_time = time.time() - start_time
        avg_throughput = inference_count / total_time
        
        logger.info(f"\nStress test complete!")
        logger.info(f"Total inferences: {inference_count}")
        logger.info(f"Average throughput: {avg_throughput:.2f} inf/s")
        logger.info(f"Final temperature: {self.get_cpu_temp():.1f}°C")
    
    def run_rmse_evaluation(self):
        """
        Comprehensive RMSE evaluation on test set.
        Compares with and without covariates.
        """
        logger.info("=" * 60)
        logger.info("RUNNING RMSE EVALUATION ON TEST SET")
        logger.info("=" * 60)
        
        # Reset to full cores
        self.set_cpu_affinity(self.num_cores)
        
        # Evaluate without covariates
        logger.info("Evaluating without covariates...")
        results_no_cov = self.evaluate_rmse_on_test_set(use_covariates=False)
        
        # Evaluate with covariates
        logger.info("Evaluating with covariates...")
        results_with_cov = self.evaluate_rmse_on_test_set(use_covariates=True)
        
        self.rmse_results = results_no_cov + results_with_cov
        
        # Summary statistics
        rmse_no_cov = [r['rmse'] for r in results_no_cov if not np.isnan(r['rmse'])]
        rmse_with_cov = [r['rmse'] for r in results_with_cov if not np.isnan(r['rmse'])]
        
        logger.info(f"\nRMSE Summary:")
        logger.info(f"  Without covariates: {np.mean(rmse_no_cov):.2f} ± {np.std(rmse_no_cov):.2f}")
        logger.info(f"  With covariates:    {np.mean(rmse_with_cov):.2f} ± {np.std(rmse_with_cov):.2f}")
        
        improvement = (np.mean(rmse_no_cov) - np.mean(rmse_with_cov)) / np.mean(rmse_no_cov) * 100
        logger.info(f"  Improvement: {improvement:.1f}%")
    
    # =========================================================================
    # COMPARISON WITH LAPTOP
    # =========================================================================
    
    def compare_with_laptop(self):
        """
        Load laptop results and generate comparison.
        """
        logger.info("=" * 60)
        logger.info("COMPARING WITH LAPTOP RESULTS")
        logger.info("=" * 60)
        
        laptop_file = self.config['laptop_results_file']
        
        if not os.path.exists(laptop_file):
            logger.warning(f"Laptop results file not found: {laptop_file}")
            logger.info("Creating placeholder comparison (run this on laptop first)")
            
            # Create comparison data structure anyway
            comparison = {
                "rpi_results": self.performance_results,
                "laptop_results": None,
                "comparison_available": False
            }
        else:
            laptop_df = pd.read_csv(laptop_file)
            logger.info(f"Loaded laptop results: {len(laptop_df)} entries")
            
            # Generate comparison
            comparison = {
                "rpi_results": self.performance_results,
                "laptop_results": laptop_df.to_dict('records'),
                "comparison_available": True
            }
            
            # Print comparison table
            rpi_df = pd.DataFrame(self.performance_results)
            
            logger.info("\nPerformance Comparison (4 cores, No Covariates):")
            
            try:
                rpi_4core = rpi_df[(rpi_df['cores'] == 4) & (rpi_df['mode'] == 'No Covariates')]
                laptop_4core = laptop_df[(laptop_df['cores'] == 4) & (laptop_df['mode'] == 'No Covariates')]
                
                if not rpi_4core.empty and not laptop_4core.empty:
                    rpi_lat = rpi_4core['latency_mean_s'].values[0]
                    laptop_lat = laptop_4core['latency_mean_s'].values[0]
                    
                    logger.info(f"  RPi Latency:    {rpi_lat:.3f}s")
                    logger.info(f"  Laptop Latency: {laptop_lat:.3f}s")
                    logger.info(f"  Slowdown:       {rpi_lat/laptop_lat:.1f}x")
            except Exception as e:
                logger.warning(f"Could not compute comparison: {e}")
        
        # Save comparison
        with open(self.config['comparison_file'].replace('.csv', '.json'), 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        return comparison
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    
    def generate_plots(self):
        """Generate all required plots for Part 4."""
        logger.info("=" * 60)
        logger.info("GENERATING PLOTS")
        logger.info("=" * 60)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        perf_df = pd.DataFrame(self.performance_results)
        thermal_df = pd.DataFrame(self.thermal_log)
        
        # ---------------------------------------------------------------------
        # Plot 1: Latency vs Cores (with/without covariates)
        # ---------------------------------------------------------------------
        ax = axes[0, 0]
        for mode in ['No Covariates', 'With Covariates']:
            data = perf_df[perf_df['mode'] == mode]
            if not data.empty:
                ax.errorbar(
                    data['cores'], data['latency_mean_s'], 
                    yerr=data['latency_std_s'],
                    marker='o', capsize=5, label=mode
                )
        ax.set_xlabel('Number of CPU Cores')
        ax.set_ylabel('Inference Latency (seconds)')
        ax.set_title('Latency vs Core Count (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4])
        
        # ---------------------------------------------------------------------
        # Plot 2: Throughput comparison (bar chart)
        # ---------------------------------------------------------------------
        ax = axes[0, 1]
        width = 0.35
        cores = sorted(perf_df['cores'].unique())
        x = np.arange(len(cores))
        
        no_cov_data = perf_df[perf_df['mode'] == 'No Covariates'].set_index('cores')
        with_cov_data = perf_df[perf_df['mode'] == 'With Covariates'].set_index('cores')
        
        if not no_cov_data.empty:
            ax.bar(x - width/2, [no_cov_data.loc[c, 'throughput_hz'] if c in no_cov_data.index else 0 for c in cores], 
                   width, label='No Covariates', color='steelblue')
        if not with_cov_data.empty:
            ax.bar(x + width/2, [with_cov_data.loc[c, 'throughput_hz'] if c in with_cov_data.index else 0 for c in cores], 
                   width, label='With Covariates', color='darkorange')
        
        ax.set_xlabel('Number of CPU Cores')
        ax.set_ylabel('Throughput (inferences/sec)')
        ax.set_title('Throughput vs Core Count (Higher is Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(cores)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # ---------------------------------------------------------------------
        # Plot 3: Thermal Profile over 30 minutes
        # ---------------------------------------------------------------------
        ax = axes[1, 0]
        if not thermal_df.empty:
            ax.plot(thermal_df['time_min'], thermal_df['temp_c'], 
                    'r-', linewidth=2, marker='o', markersize=4)
            ax.axhline(y=80, color='orange', linestyle='--', label='Throttle Threshold (80°C)')
            ax.fill_between(thermal_df['time_min'], thermal_df['temp_c'], alpha=0.3, color='red')
            
            # Mark throttling events
            throttled = thermal_df[thermal_df['throttled'] == True]
            if not throttled.empty:
                ax.scatter(throttled['time_min'], throttled['temp_c'], 
                          color='black', s=100, marker='x', label='Throttling Event', zorder=5)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('CPU Temperature (°C)')
        ax.set_title(f'Thermal Profile During {self.config["stress_test_duration_min"]} min Stress Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.config['stress_test_duration_min'])
        
        # ---------------------------------------------------------------------
        # Plot 4: CPU & Memory during stress test
        # ---------------------------------------------------------------------
        ax = axes[1, 1]
        if not thermal_df.empty:
            ax2 = ax.twinx()
            ax.plot(thermal_df['time_min'], thermal_df['cpu_percent'], 
                    'b-', linewidth=2, label='CPU %')
            ax2.plot(thermal_df['time_min'], thermal_df['memory_mb'], 
                     'g-', linewidth=2, label='Memory MB')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('CPU Utilization (%)', color='blue')
            ax2.set_ylabel('Memory Usage (MB)', color='green')
            ax.set_title('Resource Usage During Stress Test')
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # ---------------------------------------------------------------------
        # Plot 5: RMSE CDF (Cumulative Distribution)
        # ---------------------------------------------------------------------
        ax = axes[2, 0]
        if self.rmse_results:
            rmse_no_cov = sorted([r['rmse'] for r in self.rmse_results 
                                  if not r['use_covariates'] and not np.isnan(r['rmse'])])
            rmse_with_cov = sorted([r['rmse'] for r in self.rmse_results 
                                    if r['use_covariates'] and not np.isnan(r['rmse'])])
            
            if rmse_no_cov:
                cdf_no_cov = np.arange(1, len(rmse_no_cov) + 1) / len(rmse_no_cov)
                ax.plot(rmse_no_cov, cdf_no_cov, 'b-', linewidth=2, label='No Covariates')
            
            if rmse_with_cov:
                cdf_with_cov = np.arange(1, len(rmse_with_cov) + 1) / len(rmse_with_cov)
                ax.plot(rmse_with_cov, cdf_with_cov, 'r-', linewidth=2, label='With Covariates')
        
        ax.set_xlabel('RMSE')
        ax.set_ylabel('CDF (Cumulative Probability)')
        ax.set_title('CDF of RMSE: With vs Without Covariates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ---------------------------------------------------------------------
        # Plot 6: Covariate overhead (% latency increase)
        # ---------------------------------------------------------------------
        ax = axes[2, 1]
        if not no_cov_data.empty and not with_cov_data.empty:
            overhead = []
            for c in cores:
                if c in no_cov_data.index and c in with_cov_data.index:
                    base = no_cov_data.loc[c, 'latency_mean_s']
                    cov = with_cov_data.loc[c, 'latency_mean_s']
                    overhead.append((cov - base) / base * 100)
                else:
                    overhead.append(0)
            
            colors = ['green' if o < 20 else 'orange' if o < 50 else 'red' for o in overhead]
            ax.bar(cores, overhead, color=colors, edgecolor='black')
            ax.axhline(y=0, color='black', linewidth=0.5)
        
        ax.set_xlabel('Number of CPU Cores')
        ax.set_ylabel('Latency Overhead (%)')
        ax.set_title('Cost of Adding Covariates')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config['plots_file'], dpi=150, bbox_inches='tight')
        logger.info(f"Plots saved to {self.config['plots_file']}")
        
        return fig
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_all_results(self):
        """Save all results to CSV files."""
        logger.info("Saving results...")
        
        # Performance results
        if self.performance_results:
            pd.DataFrame(self.performance_results).to_csv(
                self.config['results_file'], index=False
            )
            logger.info(f"  Performance results: {self.config['results_file']}")
        
        # Thermal log
        if self.thermal_log:
            pd.DataFrame(self.thermal_log).to_csv(
                self.config['thermal_log_file'], index=False
            )
            logger.info(f"  Thermal log: {self.config['thermal_log_file']}")
        
        # RMSE results
        if self.rmse_results:
            # Save summary (not full predictions)
            rmse_summary = [{
                'test_idx': r['test_idx'],
                'rmse': r['rmse'],
                'use_covariates': r['use_covariates']
            } for r in self.rmse_results]
            pd.DataFrame(rmse_summary).to_csv(
                self.config['rmse_results_file'], index=False
            )
            logger.info(f"  RMSE results: {self.config['rmse_results_file']}")
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run_full_benchmark(self):
        """Execute the complete Part 4 benchmark suite."""
        logger.info("=" * 60)
        logger.info("PART 4: RASPBERRY PI BENCHMARK SUITE")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Load Chronos model
            if not self.load_model():
                logger.error("Failed to load model. Exiting.")
                return
            
            # Step 3: Train covariate model
            self.train_covariate_model()
            
            # Step 4: Core scaling benchmark
            self.run_core_scaling_benchmark()
            
            # Step 5: RMSE evaluation
            self.run_rmse_evaluation()
            
            # Step 6: Thermal stress test
            self.run_thermal_stress_test()
            
            # Step 7: Compare with laptop
            self.compare_with_laptop()
            
            # Step 8: Generate plots
            self.generate_plots()
            
            # Step 9: Save results
            self.save_all_results()
            
            logger.info("=" * 60)
            logger.info("BENCHMARK COMPLETE!")
            logger.info("=" * 60)
            
        except KeyboardInterrupt:
            logger.warning("Benchmark interrupted by user")
            self.save_all_results()
            self.generate_plots()
        
        except Exception as e:
            logger.error(f"Benchmark failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.save_all_results()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Check for root (recommended for accurate CPU pinning)
    if os.geteuid() != 0:
        print("WARNING: Running without sudo. CPU affinity and temp reading may fail.")
        print("Recommended: sudo python3 rpi_part4_complete.py\n")
    
    # Parse command line arguments for quick config changes
    import argparse
    parser = argparse.ArgumentParser(description='RPi PM2.5 Forecasting Benchmark')
    parser.add_argument('--data', type=str, default=CONFIG['data_file'], 
                        help='Data file path')
    parser.add_argument('--stress-duration', type=int, default=CONFIG['stress_test_duration_min'],
                        help='Stress test duration in minutes')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (5 min stress, 5 reps)')
    args = parser.parse_args()
    
    # Update config
    CONFIG['data_file'] = args.data
    CONFIG['stress_test_duration_min'] = args.stress_duration
    
    if args.quick:
        CONFIG['stress_test_duration_min'] = 5
        CONFIG['inference_reps_per_test'] = 5
        logger.info("QUICK MODE: 5 min stress test, 5 inference reps")
    
    # Run benchmark
    benchmark = RPiPM25Benchmark(CONFIG)
    benchmark.run_full_benchmark()