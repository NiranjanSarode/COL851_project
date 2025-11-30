import requests
import time
import numpy as np
import threading
import random
import matplotlib.pyplot as plt

FIXED_SEED = 1
random.seed(FIXED_SEED)      # Fixes the model variant choices
np.random.seed(FIXED_SEED)   # Fixes the Poisson wait times

SERVER_URL = "http://localhost:8001/predict"
NUM_REQUESTS = 50

# Trace Configuration
VARIANTS = ['tiny', 'small', 'base'] # We mix models to stress the server
CONTEXTS = [2, 8, 14]
HORIZONS = [4, 12, 24]
COVARIATES = [True, False]

def generate_trace(n=50):
    trace = []
    for i in range(n):
        req = {
            "request_id": i,
            "model_variant": random.choice(VARIANTS),
            "context_len": random.choice(CONTEXTS),
            "horizon": random.choice(HORIZONS),
            "use_covariates": random.choice(COVARIATES)
        }
        trace.append(req)
    return trace

def run_experiment(mode="burst"):
    trace = generate_trace(NUM_REQUESTS)
    latencies = []
    start_time = time.time()
    
    print(f"\n--- Starting {mode.upper()} Load Test ({NUM_REQUESTS} reqs) ---")
    
    for req in trace:
        t_req_start = time.time()
        
        try:
            resp = requests.post(SERVER_URL, json=req)
            if resp.status_code == 200:
                data = resp.json()
                latencies.append(data['latency_ms'])
                print(f"Req {req['request_id']} ({req['model_variant']}): {data['latency_ms']:.1f}ms")
            else:
                print(f"Error: {resp.status_code}")
        except Exception as e:
            print(f"Connection Error: {e}")
            
        # Poisson Wait (Inter-arrival time)
        if mode == "poisson":
            # Average arrival rate (lambda) = 2 requests per second
            wait_time = np.random.exponential(scale=0.5) 
            time.sleep(wait_time)
            
    total_time = time.time() - start_time
    throughput = NUM_REQUESTS / total_time
    
    return latencies, throughput

if __name__ == "__main__":
    # 1. Run Burst
    lat_burst, thru_burst = run_experiment("burst")
    
    print("\nCooldown (5s)...")
    time.sleep(5)
    
    # 2. Run Poisson
    lat_poisson, thru_poisson = run_experiment("poisson")
    
    # 3. Simple Plot
    plt.figure(figsize=(10, 5))
    plt.boxplot([lat_burst, lat_poisson], labels=['Burst (Sequential)', 'Poisson (Random Wait)'])
    plt.ylabel('Server Latency (ms)')
    plt.title('Baseline Server Performance')
    plt.savefig('server_baseline_results.png')
    plt.title('Optimized Server Performance (Batching + Caching + INT8)')
    
    print("\nResults:")
    print(f"Burst Throughput:   {thru_burst:.2f} req/s")
    print(f"Poisson Throughput: {thru_poisson:.2f} req/s")