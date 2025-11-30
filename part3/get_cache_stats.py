#!/usr/bin/env python3
"""
get_cache_stats.py
Simple periodic cache statistics using perf stat
Runs perf for 5 seconds to capture real cache behavior
"""

import subprocess
import re
import os

def get_cache_stats(duration=1):
    """
    Get REAL hardware cache statistics by running perf for `duration` seconds.
    
    Args:
        duration: How long to sample cache stats (default 5 seconds)
    
    Returns:
        dict with cache_hit_rate, l1_hit_rate, cache_misses
    """
    
    pid = os.getpid()
    
    try:
        # Run perf stat on the current process for `duration` seconds
        cmd = [
            'perf', 'stat',
            '-e', 'cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses',
            '-p', str(pid),
            'sleep', str(duration)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 2
        )
        
        # Parse output (perf writes to stderr)
        output = result.stderr
        
        # Extract values
        cache_refs = 0
        cache_miss = 0
        l1_loads = 0
        l1_miss = 0
        
        patterns = {
            'cache_references': r'([\d,]+)\s+cache-references',
            'cache_misses': r'([\d,]+)\s+cache-misses',
            'l1_loads': r'([\d,]+)\s+L1-dcache-loads',
            'l1_miss': r'([\d,]+)\s+L1-dcache-load-misses'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    value = int(value_str)
                    if key == 'cache_references':
                        cache_refs = value
                    elif key == 'cache_misses':
                        cache_miss = value
                    elif key == 'l1_loads':
                        l1_loads = value
                    elif key == 'l1_miss':
                        l1_miss = value
                except ValueError:
                    pass
        
        # Calculate hit rates
        if cache_refs > 1000:
            cache_hit_rate = 1.0 - (cache_miss / cache_refs)
        else:
            cache_hit_rate = 0.0
        
        if l1_loads > 1000:
            l1_hit_rate = 1.0 - (l1_miss / l1_loads)
        else:
            l1_hit_rate = 0.0
        
        return {
            'cache_hit_rate': max(0.0, min(1.0, cache_hit_rate)),
            'l1_hit_rate': max(0.0, min(1.0, l1_hit_rate)),
            'cache_misses': cache_miss,
            'cache_references': cache_refs,
            'l1_loads': l1_loads
        }
        
    except Exception as e:
        # Return zeros if perf fails
        return {
            'cache_hit_rate': 0.0,
            'l1_hit_rate': 0.0,
            'cache_misses': 0,
            'cache_references': 0,
            'l1_loads': 0
        }


def reset_cache_stats():
    """No-op for compatibility"""
    pass


# Test
if __name__ == "__main__":
    print("Testing perf cache monitoring")
    print("Sampling for 5 seconds...\n")
    
    # Do some work in background
    import threading
    def work():
        data = []
        for i in range(100000):
            data.append([j*j for j in range(100)])
    
    thread = threading.Thread(target=work)
    thread.start()
    
    # Capture stats
    stats = get_cache_stats(duration=5)
    
    print(f"Cache Hit Rate:  {stats['cache_hit_rate']:.2%}")
    print(f"L1 Hit Rate:     {stats['l1_hit_rate']:.2%}")
    print(f"Cache Misses:    {stats['cache_misses']:,}")
    print(f"Cache Refs:      {stats['cache_references']:,}")
    print(f"L1 Loads:        {stats['l1_loads']:,}")
    
    thread.join()



# CLI interface
if __name__ == "__main__":
    import time
    
    print("Hardware Cache Statistics Monitor")
    print("=" * 60)
    print("Using: perf stat (actual CPU performance counters)")
    print()
    
    # Check if perf is available
    try:
        result = subprocess.run(['perf', '--version'], capture_output=True, check=True)
        print("✓ perf command available")
    except:
        print("✗ perf command not found")
        print("  Install: sudo apt-get install linux-tools-generic linux-tools-$(uname -r)")
        exit(1)
    
    # Check permissions
    try:
        with open('/proc/sys/kernel/perf_event_paranoid', 'r') as f:
            paranoid = int(f.read().strip())
            if paranoid <= 1:
                print(f"✓ perf_event_paranoid = {paranoid} (OK)")
            else:
                print(f"✗ perf_event_paranoid = {paranoid} (needs to be <= 1)")
                print("  Run: sudo sysctl kernel.perf_event_paranoid=1")
    except:
        pass
    
    print("\nMeasuring cache behavior (5 samples)...\n")
    
    for i in range(5):
        stats = get_cache_stats()
        
        if stats.get('_cache_refs', 0) > 0:
            print(f"Sample {i+1}:")
            print(f"  Cache Hit Rate:  {stats['cache_hit_rate']:.2%}")
            print(f"  L1 Hit Rate:     {stats['l1_hit_rate']:.2%}")
            print(f"  Cache Misses:    {stats['cache_misses']:,}")
            print(f"  Cache Refs:      {stats['_cache_refs']:,}")
            print(f"  L1 Loads:        {stats['_l1_loads']:,}")
        else:
            print(f"Sample {i+1}: No data - perf may not support these events on your CPU")
        
        print()
    
    print("=" * 60)
    print("These are REAL hardware cache counters from CPU PMU!")
    print("If you see 'No data', your CPU may not support these perf events.")
