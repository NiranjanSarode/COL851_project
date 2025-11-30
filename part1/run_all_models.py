"""
Comprehensive Chronos Model Comparison - ALL 5 VARIANTS
Run full experiments with Tiny, Mini, Small, Base, and Large models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
from chronos import ChronosPipeline
import warnings
import time
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

file_name = 'df_patna_covariates.csv'

# Load the data
print("Loading data...")
df = pd.read_csv(file_name)
df['From Date'] = pd.to_datetime(df['From Date'])
df = df.sort_values('From Date')

pm_values = df['calibPM'].values
print(f"Total data points: {len(pm_values)}")

# Handle missing values
if np.isnan(pm_values).any():
    pm_series = pd.Series(pm_values)
    pm_series = pm_series.interpolate(method='linear')
    pm_values = pm_series.values

# All 5 model variants
MODEL_VARIANTS = ['tiny', 'mini', 'small', 'base', 'large']

print("\n" + "="*80)
print("COMPREHENSIVE CHRONOS MODEL COMPARISON - ALL 5 VARIANTS")
print("="*80)

all_results = {
    'context_length_exp': {},  # model -> {days -> rmse}
    'horizon_exp': {}  # model -> {hours -> rmse}
}

model_info = {}

def evaluate_forecasts(model_name, pipeline, context_length_hours, prediction_horizon_hours, 
                       pm_values, num_samples=20, max_windows=20):
    """Run forecasting evaluation"""
    try:
        rmse_scores = []
        test_start_idx = int(len(pm_values) * 0.7)
        max_test_windows = min(max_windows, (len(pm_values) - test_start_idx - context_length_hours) // prediction_horizon_hours)
        
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
        
        return avg_rmse, std_rmse, len(rmse_scores)
        
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None, None, 0

# Process each model variant
for model_idx, model_name in enumerate(MODEL_VARIANTS):
    print(f"\n{'='*80}")
    print(f"MODEL {model_idx+1}/5: chronos-t5-{model_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Load model
        print(f"Loading model...", end='', flush=True)
        start_time = time.time()
        
        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{model_name}",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        
        load_time = time.time() - start_time
        print(f" ✓ Loaded in {load_time:.1f}s")
        
        # Store model info
        model_info[model_name] = {'load_time': load_time, 'status': 'loaded'}
        
        # Initialize results for this model
        all_results['context_length_exp'][model_name] = {}
        all_results['horizon_exp'][model_name] = {}
        
        # Experiment 1: Context Length Analysis (24h horizon)
        print(f"\nExperiment 1: Context Length Analysis (24h forecast)")
        print("-" * 60)
        
        context_lengths_days = [2, 4, 8, 10, 14]
        forecast_horizon_hours = 24
        
        for days in context_lengths_days:
            context_hours = days * 24
            print(f"  {days:2d} days ({context_hours:3d}h)...", end='', flush=True)
            
            rmse, std, n_windows = evaluate_forecasts(
                model_name=model_name,
                pipeline=pipeline,
                context_length_hours=context_hours,
                prediction_horizon_hours=forecast_horizon_hours,
                pm_values=pm_values,
                num_samples=20,
                max_windows=20
            )
            
            all_results['context_length_exp'][model_name][days] = {
                'rmse': rmse, 'std': std, 'n_windows': n_windows
            }
            
            if rmse is not None:
                print(f" RMSE: {rmse:6.2f} (±{std:5.2f}) [{n_windows} windows]")
            else:
                print(f" FAILED")
        
        # Experiment 2: Forecast Horizon Analysis (10-day context)
        print(f"\nExperiment 2: Forecast Horizon Analysis (10-day context)")
        print("-" * 60)
        
        context_length_days = 10
        forecast_horizons_hours = [4, 8, 12, 24, 48]
        
        for hours in forecast_horizons_hours:
            print(f"  {hours:2d} hours...", end='', flush=True)
            
            rmse, std, n_windows = evaluate_forecasts(
                model_name=model_name,
                pipeline=pipeline,
                context_length_hours=context_length_days * 24,
                prediction_horizon_hours=hours,
                pm_values=pm_values,
                num_samples=20,
                max_windows=20
            )
            
            all_results['horizon_exp'][model_name][hours] = {
                'rmse': rmse, 'std': std, 'n_windows': n_windows
            }
            
            if rmse is not None:
                print(f" RMSE: {rmse:6.2f} (±{std:5.2f}) [{n_windows} windows]")
            else:
                print(f" FAILED")
        
        print(f"\n✓ Model {model_name} complete!")
        model_info[model_name]['status'] = 'completed'
        
        # Clean up to free memory
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"\n✗ Model {model_name} FAILED: {str(e)}")
        model_info[model_name] = {'load_time': 0, 'status': 'failed', 'error': str(e)}
        all_results['context_length_exp'][model_name] = {}
        all_results['horizon_exp'][model_name] = {}

# Print comprehensive results
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS - ALL MODELS")
print("="*80)

print("\n📊 EXPERIMENT 1: RMSE by Context Length (24-hour forecast)")
print("-" * 80)
print(f"{'Context':>10} | {'Tiny':>10} | {'Mini':>10} | {'Small':>10} | {'Base':>10} | {'Large':>10}")
print("-" * 80)

context_lengths_days = [2, 4, 8, 10, 14]
for days in context_lengths_days:
    row = f"{days:6d} days |"
    for model in MODEL_VARIANTS:
        if model in all_results['context_length_exp'] and days in all_results['context_length_exp'][model]:
            rmse = all_results['context_length_exp'][model][days]['rmse']
            if rmse is not None:
                row += f" {rmse:10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        else:
            row += f" {'---':>10} |"
    print(row)

print("\n📊 EXPERIMENT 2: RMSE by Forecast Horizon (10-day context)")
print("-" * 80)
print(f"{'Horizon':>10} | {'Tiny':>10} | {'Mini':>10} | {'Small':>10} | {'Base':>10} | {'Large':>10}")
print("-" * 80)

forecast_horizons_hours = [4, 8, 12, 24, 48]
for hours in forecast_horizons_hours:
    row = f"{hours:5d} hours |"
    for model in MODEL_VARIANTS:
        if model in all_results['horizon_exp'] and hours in all_results['horizon_exp'][model]:
            rmse = all_results['horizon_exp'][model][hours]['rmse']
            if rmse is not None:
                row += f" {rmse:10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        else:
            row += f" {'---':>10} |"
    print(row)

# Find best configurations across all models
print("\n" + "="*80)
print("🏆 BEST CONFIGURATIONS")
print("="*80)

best_overall = None
best_rmse = float('inf')

for model in MODEL_VARIANTS:
    if model in all_results['context_length_exp']:
        for days in all_results['context_length_exp'][model]:
            rmse = all_results['context_length_exp'][model][days]['rmse']
            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
                best_overall = (model, days, 24, rmse)

if best_overall:
    print(f"\n🥇 OVERALL BEST:")
    print(f"   Model:   chronos-t5-{best_overall[0]}")
    print(f"   Context: {best_overall[1]} days")
    print(f"   Horizon: {best_overall[2]} hours")
    print(f"   RMSE:    {best_overall[3]:.2f} µg/m³")

print(f"\n📋 BEST BY MODEL (14-day context, 24h horizon):")
for model in MODEL_VARIANTS:
    if model in all_results['context_length_exp'] and 14 in all_results['context_length_exp'][model]:
        rmse = all_results['context_length_exp'][model][14]['rmse']
        if rmse is not None:
            print(f"   {model:6s}: {rmse:6.2f} RMSE")
        else:
            print(f"   {model:6s}: FAILED")
    else:
        print(f"   {model:6s}: NOT RUN")

# Create comprehensive visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

colors = {'tiny': '#1f77b4', 'mini': '#ff7f0e', 'small': '#2ca02c', 
          'base': '#d62728', 'large': '#9467bd'}
markers = {'tiny': 'o', 'mini': 's', 'small': '^', 'base': 'D', 'large': 'v'}

# Plot 1: Context Length Comparison
ax1 = axes[0, 0]
for model in MODEL_VARIANTS:
    if model in all_results['context_length_exp']:
        days_list = []
        rmse_list = []
        for days in sorted(all_results['context_length_exp'][model].keys()):
            rmse = all_results['context_length_exp'][model][days]['rmse']
            if rmse is not None:
                days_list.append(days)
                rmse_list.append(rmse)
        
        if days_list:
            ax1.plot(days_list, rmse_list, marker=markers[model], linewidth=2.5, 
                    markersize=8, label=f'{model.capitalize()}', color=colors[model])

ax1.set_xlabel('Context Length (days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE (µg/m³)', fontsize=12, fontweight='bold')
ax1.set_title('RMSE vs Context Length\n(24-hour Forecast Horizon)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Forecast Horizon Comparison
ax2 = axes[0, 1]
for model in MODEL_VARIANTS:
    if model in all_results['horizon_exp']:
        hours_list = []
        rmse_list = []
        for hours in sorted(all_results['horizon_exp'][model].keys()):
            rmse = all_results['horizon_exp'][model][hours]['rmse']
            if rmse is not None:
                hours_list.append(hours)
                rmse_list.append(rmse)
        
        if hours_list:
            ax2.plot(hours_list, rmse_list, marker=markers[model], linewidth=2.5, 
                    markersize=8, label=f'{model.capitalize()}', color=colors[model])

ax2.set_xlabel('Forecast Horizon (hours)', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE (µg/m³)', fontsize=12, fontweight='bold')
ax2.set_title('RMSE vs Forecast Horizon\n(10-day Context Length)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Best RMSE by Model (14-day context)
ax3 = axes[1, 0]
model_names = []
rmse_values = []
for model in MODEL_VARIANTS:
    if model in all_results['context_length_exp'] and 14 in all_results['context_length_exp'][model]:
        rmse = all_results['context_length_exp'][model][14]['rmse']
        if rmse is not None:
            model_names.append(model.capitalize())
            rmse_values.append(rmse)

if model_names:
    bars = ax3.bar(model_names, rmse_values, color=[colors[m.lower()] for m in model_names], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('RMSE (µg/m³)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Comparison\n(14-day Context, 24h Horizon)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Model Information Table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

table_data = [['Model', 'Status', 'Best RMSE', 'Context']]
for model in MODEL_VARIANTS:
    if model in model_info:
        status = model_info[model].get('status', 'unknown')
        if status == 'completed' and model in all_results['context_length_exp']:
            best_rmse = min([r['rmse'] for r in all_results['context_length_exp'][model].values() 
                           if r['rmse'] is not None], default=None)
            if best_rmse:
                best_context = [d for d, r in all_results['context_length_exp'][model].items() 
                              if r['rmse'] == best_rmse][0]
                table_data.append([model.capitalize(), '✓ Done', f'{best_rmse:.2f}', f'{best_context}d'])
            else:
                table_data.append([model.capitalize(), '✗ Failed', 'N/A', 'N/A'])
        else:
            table_data.append([model.capitalize(), status, 'N/A', 'N/A'])
    else:
        table_data.append([model.capitalize(), '---', 'N/A', 'N/A'])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax4.set_title('Model Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
png_file_name = file_name +'all_models_comparison.png'
plt.savefig(png_file_name, dpi=300, bbox_inches='tight')
print("✓ Saved: all_models_comparison.png")

# Save detailed results to CSV
results_list = []
for model in MODEL_VARIANTS:
    # Context length results
    if model in all_results['context_length_exp']:
        for days, metrics in all_results['context_length_exp'][model].items():
            results_list.append({
                'Model': model,
                'Experiment': 'Context Length',
                'Parameter': days,
                'Unit': 'days',
                'RMSE': metrics['rmse'],
                'Std_Dev': metrics['std'],
                'N_Windows': metrics['n_windows']
            })
    
    # Horizon results
    if model in all_results['horizon_exp']:
        for hours, metrics in all_results['horizon_exp'][model].items():
            results_list.append({
                'Model': model,
                'Experiment': 'Forecast Horizon',
                'Parameter': hours,
                'Unit': 'hours',
                'RMSE': metrics['rmse'],
                'Std_Dev': metrics['std'],
                'N_Windows': metrics['n_windows']
            })

results_df = pd.DataFrame(results_list)
csv_file_name = file_name + 'all_models_results.csv'
results_df.to_csv(csv_file_name, index=False)
print("✓ Saved: all_models_results.csv")

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  1. all_models_comparison.png - Comprehensive visualization")
print("  2. all_models_results.csv - Complete numerical results")
