"""
Complete Chronos Forecasting Experiments with BASE Model
Run all experiments with the base model variant for best accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the data
print("Loading data...")
df = pd.read_csv('df_patna_covariates.csv')
df['From Date'] = pd.to_datetime(df['From Date'])
df = df.sort_values('From Date')

pm_values = df['calibPM'].values
print(f"Total data points: {len(pm_values)}")
print(f"Date range: {df['From Date'].min()} to {df['From Date'].max()}")

# Handle missing values
if np.isnan(pm_values).any():
    print(f"Found {np.isnan(pm_values).sum()} missing values. Interpolating...")
    pm_series = pd.Series(pm_values)
    pm_series = pm_series.interpolate(method='linear')
    pm_values = pm_series.values

# Use the BASE model variant (best accuracy)
MODEL_NAME = 'base'

print("\n" + "="*80)
print(f"CHRONOS FORECASTING WITH BASE MODEL: chronos-t5-{MODEL_NAME}")
print("="*80)

results = {
    'context_length_exp': {},
    'horizon_exp': {}
}

def evaluate_forecasts(model_name, context_length_hours, prediction_horizon_hours, 
                       pm_values, num_samples=50):
    """Run forecasting evaluation with given parameters"""
    try:
        # Load the model
        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{model_name}",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        
        rmse_scores = []
        test_start_idx = int(len(pm_values) * 0.7)
        max_test_windows = min(20, (len(pm_values) - test_start_idx - context_length_hours) // prediction_horizon_hours)
        
        for i in range(max_test_windows):
            start_idx = test_start_idx + i * prediction_horizon_hours
            end_idx = start_idx + context_length_hours
            
            if end_idx + prediction_horizon_hours > len(pm_values):
                break
            
            context = torch.tensor(pm_values[start_idx:end_idx])
            ground_truth = pm_values[end_idx:end_idx + prediction_horizon_hours]
            
            forecast = pipeline.predict(
                context=context,
                prediction_length=prediction_horizon_hours,
                num_samples=num_samples,
                limit_prediction_length=False
            )
            
            predictions = np.median(forecast[0].numpy(), axis=0)
            rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
            rmse_scores.append(rmse)
        
        avg_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        print(f"  Evaluated {len(rmse_scores)} windows, Avg RMSE: {avg_rmse:.2f} (±{std_rmse:.2f})")
        
        return avg_rmse, std_rmse
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None, None

# Experiment 1: Different context lengths with 24-hour horizon
print("\n" + "-"*80)
print("EXPERIMENT 1: Varying Context Lengths (24-hour forecast horizon)")
print("-"*80)

context_lengths_days = [2, 4, 8, 10, 14]
forecast_horizon_hours = 24

for days in context_lengths_days:
    context_hours = days * 24
    print(f"\nContext Length: {days} days ({context_hours} hours)")
    
    rmse, std = evaluate_forecasts(
        model_name=MODEL_NAME,
        context_length_hours=context_hours,
        prediction_horizon_hours=forecast_horizon_hours,
        pm_values=pm_values,
        num_samples=50
    )
    
    results['context_length_exp'][days] = {'rmse': rmse, 'std': std}

# Experiment 2: Different forecast horizons with 10-day context
print("\n" + "-"*80)
print("EXPERIMENT 2: Varying Forecast Horizons (10-day context length)")
print("-"*80)

context_length_days = 10
forecast_horizons_hours = [4, 8, 12, 24, 48]

for hours in forecast_horizons_hours:
    print(f"\nForecast Horizon: {hours} hours")
    
    rmse, std = evaluate_forecasts(
        model_name=MODEL_NAME,
        context_length_hours=context_length_days * 24,
        prediction_horizon_hours=hours,
        pm_values=pm_values,
        num_samples=50
    )
    
    results['horizon_exp'][hours] = {'rmse': rmse, 'std': std}

# Print Results Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\nModel Used: chronos-t5-{MODEL_NAME}")

print("\nExperiment 1 - RMSE by Context Length (24h horizon):")
for days, metrics in results['context_length_exp'].items():
    if metrics['rmse'] is not None:
        print(f"  {days:2d} days: RMSE = {metrics['rmse']:6.2f} (±{metrics['std']:5.2f})")

print("\nExperiment 2 - RMSE by Forecast Horizon (10-day context):")
for hours, metrics in results['horizon_exp'].items():
    if metrics['rmse'] is not None:
        print(f"  {hours:2d} hours: RMSE = {metrics['rmse']:6.2f} (±{metrics['std']:5.2f})")

# Find best configurations
exp1_best = min([(k, v['rmse']) for k, v in results['context_length_exp'].items() if v['rmse'] is not None], 
                key=lambda x: x[1])
exp2_best = min([(k, v['rmse']) for k, v in results['horizon_exp'].items() if v['rmse'] is not None], 
                key=lambda x: x[1])

print(f"\n{'='*80}")
print(f"OPTIMAL CONFIGURATION")
print(f"{'='*80}")
print(f"Best Context Length: {exp1_best[0]} days (RMSE: {exp1_best[1]:.2f})")
print(f"Best Forecast Horizon: {exp2_best[0]} hours (RMSE: {exp2_best[1]:.2f})")

# Create Enhanced Visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: RMSE vs Context Length with error bars
context_days = list(results['context_length_exp'].keys())
context_rmse = [results['context_length_exp'][d]['rmse'] for d in context_days if results['context_length_exp'][d]['rmse'] is not None]
context_std = [results['context_length_exp'][d]['std'] for d in context_days if results['context_length_exp'][d]['rmse'] is not None]
context_days_filtered = [d for d in context_days if results['context_length_exp'][d]['rmse'] is not None]

axes[0].errorbar(context_days_filtered, context_rmse, yerr=context_std, 
                 marker='o', linewidth=2.5, markersize=10, capsize=5, capthick=2,
                 color='#2E86AB', ecolor='#A23B72', label='RMSE ± Std Dev')
axes[0].set_xlabel('Context Length (days)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Average RMSE (µg/m³)', fontsize=13, fontweight='bold')
axes[0].set_title(f'RMSE vs Context Length\n24-hour Forecast Horizon (Model: {MODEL_NAME})', 
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xticks(context_days_filtered)
axes[0].legend(fontsize=11)

# Plot 2: RMSE vs Forecast Horizon with error bars
horizon_hours = list(results['horizon_exp'].keys())
horizon_rmse = [results['horizon_exp'][h]['rmse'] for h in horizon_hours if results['horizon_exp'][h]['rmse'] is not None]
horizon_std = [results['horizon_exp'][h]['std'] for h in horizon_hours if results['horizon_exp'][h]['rmse'] is not None]
horizon_hours_filtered = [h for h in horizon_hours if results['horizon_exp'][h]['rmse'] is not None]

axes[1].errorbar(horizon_hours_filtered, horizon_rmse, yerr=horizon_std,
                 marker='s', linewidth=2.5, markersize=10, capsize=5, capthick=2,
                 color='#F18F01', ecolor='#C73E1D', label='RMSE ± Std Dev')
axes[1].set_xlabel('Forecast Horizon (hours)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average RMSE (µg/m³)', fontsize=13, fontweight='bold')
axes[1].set_title(f'RMSE vs Forecast Horizon\n10-day Context Length (Model: {MODEL_NAME})', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xticks(horizon_hours_filtered)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(f'chronos_forecasting_{MODEL_NAME}_results.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved plot: chronos_forecasting_{MODEL_NAME}_results.png")

# Save detailed results to CSV
results_data = []
for days, metrics in results['context_length_exp'].items():
    results_data.append({
        'Model': MODEL_NAME,
        'Experiment': 'Context Length',
        'Parameter': days,
        'Unit': 'days',
        'RMSE': metrics['rmse'],
        'Std_Dev': metrics['std']
    })

for hours, metrics in results['horizon_exp'].items():
    results_data.append({
        'Model': MODEL_NAME,
        'Experiment': 'Forecast Horizon',
        'Parameter': hours,
        'Unit': 'hours',
        'RMSE': metrics['rmse'],
        'Std_Dev': metrics['std']
    })

results_df = pd.DataFrame(results_data)
results_df.to_csv(f'chronos_{MODEL_NAME}_results.csv', index=False)
print(f"✓ Saved results: chronos_{MODEL_NAME}_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nFiles generated:")
print(f"  1. chronos_forecasting_{MODEL_NAME}_results.png - Visualization plots")
print(f"  2. chronos_{MODEL_NAME}_results.csv - Detailed numerical results")
print(f"\nModel: chronos-t5-{MODEL_NAME}")
print(f"Best Context Length: {exp1_best[0]} days (RMSE: {exp1_best[1]:.2f})")
print(f"Best Forecast Horizon: {exp2_best[0]} hours (RMSE: {exp2_best[1]:.2f})")

