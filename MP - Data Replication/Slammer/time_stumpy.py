# ==============================================================================
# Copyright (c) UNIVERSIDAD AUTÓNOMA DE MADRID
# Francisco Tomás y Valiente, no 1
# Madrid, 28049
# Spain
#
# Óscar Cuevas Martínez
# Evaluating the Performance of BGP Different Anomaly Detection Methods
# All Rights Reserved
# ==============================================================================

import pandas as pd
import stumpy
import numpy as np
import time

# --- CONFIGURATION ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print(f"--- BENCHMARK STUMPY: Slammer (m={WINDOW_SIZE}) ---")

# 1. LOAD DATA (not timed)
print(f"Loading data...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    # Explicit numpy array conversion to avoid pandas overhead during computation
    time_series_values = time_series.values.copy() 
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. WARM-UP
# Numba needs to compile the function on the first call.
# We run a small dummy execution to warm up the JIT compiler
# so it does not affect the real measurement.
print("Warming up JIT compiler (Numba)...")
stumpy.stump(time_series_values[:100], m=10) 

# 3. REAL-TIME MEASUREMENT
print(f"Running stumpy.stump on the entire series...")

start_time = time.time()
# --- START TIMER ---

mp_result = stumpy.stump(time_series_values, m=WINDOW_SIZE)

# --- END TIMER ---
end_time = time.time()

duration = end_time - start_time
print(f"\nBenchmark Results:")
print(f"-------------------------")
print(f"Execution time: {duration:.4f} seconds")
print(f"-------------------------")

# Quick sanity check (sanity check)
mp_vector = mp_result[:, 0]
k1_idx = np.argsort(mp_vector)[-1]
print(f"Sanity check: Highest peak found at index {k1_idx}")