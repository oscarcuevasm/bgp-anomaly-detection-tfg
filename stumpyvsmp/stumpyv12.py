import pandas as pd
import numpy as np
import time
import sys

# CONFIGURACION
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print("==================================================")
print(" BENCHMARK: STUMPY (PYTHON 3.12)")
print(f" Python Version: {sys.version.split()[0]}")
print(f" Dataset: {FILE_NAME} | Window Size (m): {WINDOW_SIZE}")
print("==================================================\n")

# 1. CARGAR DATOS
print("Loading data...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    # Conversion explicita a numpy array para optimizar rendimiento
    time_series_values = time_series.values.copy()
    print("Data loaded successfully.\n")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit()

# 2. STUMPY BENCHMARK
try:
    import stumpy
    print("--- STUMPY LIBRARY ---")
    print("Warming up JIT compiler (Numba)...")
    # Ejecucion pequena para compilar Numba JIT sin afectar la medicion real
    _ = stumpy.stump(time_series_values[:100], m=10)
    
    print("Executing stumpy.stump on the entire series...")
    start_time_st = time.time()
    mp_result_st = stumpy.stump(time_series_values, m=WINDOW_SIZE)
    end_time_st = time.time()
    
    duration_st = end_time_st - start_time_st
    mp_vector_st = mp_result_st[:, 0]
    
    # Encontrar el indice del valor maximo (discord)
    k1_idx_st = np.argsort(mp_vector_st)[-1]
    
    print(f"Execution time: {duration_st:.4f} seconds")
    print(f"Sanity Check: Highest peak found at index {k1_idx_st}\n")
    
except ImportError:
    print("ERROR: 'stumpy' is not installed in this Python environment.")
    print("Please run: pip install stumpy")

print("==================================================")
print(" BENCHMARK COMPLETED")
print("==================================================")