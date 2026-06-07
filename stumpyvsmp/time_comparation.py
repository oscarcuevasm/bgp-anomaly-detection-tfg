import pandas as pd
import numpy as np
import time
import sys

# CONFIGURATION
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print("==================================================")
print(" BENCHMARK: MATRIXPROFILE VS STUMPY")
print(f" Python Version: {sys.version.split()[0]}")
print(f" Dataset: {FILE_NAME} | Window Size (m): {WINDOW_SIZE}")
print("==================================================\n")

# 1. LOAD DATA
print("Loading data...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    # Explicit numpy array conversion to avoid pandas overhead
    time_series_values = time_series.values.copy()
    print("Data loaded successfully.\n")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. MATRIXPROFILE BENCHMARK
try:
    import matrixprofile as mp
    print("--- MATRIXPROFILE LIBRARY ---")
    print("Executing mp.compute on the entire series...")
    
    start_time_mp = time.time()
    profile_mp = mp.compute(time_series_values, WINDOW_SIZE)
    end_time_mp = time.time()
    
    duration_mp = end_time_mp - start_time_mp
    mp_vector_mp = profile_mp['mp']
    k1_idx_mp = np.argmax(mp_vector_mp)
    
    print(f"Execution time: {duration_mp:.4f} seconds")
    print(f"Sanity Check: Highest peak found at index {k1_idx_mp}\n")
    
except ImportError:
    print("--- MATRIXPROFILE LIBRARY ---")
    print("SKIPPED: 'matrixprofile' is not installed in this Python environment.\n")

# 3. STUMPY BENCHMARK
try:
    import stumpy
    print("--- STUMPY LIBRARY ---")
    print("Warming up JIT compiler (Numba)...")
    # Small execution to compile Numba JIT without affecting real measurement
    _ = stumpy.stump(time_series_values[:100], m=10)
    
    print("Executing stumpy.stump on the entire series...")
    start_time_st = time.time()
    mp_result_st = stumpy.stump(time_series_values, m=WINDOW_SIZE)
    end_time_st = time.time()
    
    duration_st = end_time_st - start_time_st
    mp_vector_st = mp_result_st[:, 0]
    k1_idx_st = np.argsort(mp_vector_st)[-1]
    
    print(f"Execution time: {duration_st:.4f} seconds")
    print(f"Sanity Check: Highest peak found at index {k1_idx_st}\n")
    
except ImportError:
    print("--- STUMPY LIBRARY ---")
    print("SKIPPED: 'stumpy' is not installed in this Python environment.\n")

print("==================================================")
print(" BENCHMARK COMPLETED")
print("==================================================")