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
import matrixprofile as mp
import numpy as np
import time

# --- CONFIGURATION ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print(f"--- BENCHMARK MATRIXPROFILE (Library): Slammer (m={WINDOW_SIZE}) ---")

# 1. LOAD DATA
print(f"Loading data...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    time_series_values = time_series.values
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. REAL-TIME MEASUREMENT
# The matrixprofile library has no JIT compilation (warm-up),
# so we measure computation time directly.
print(f"Running mp.compute on the entire series...")

start_time = time.time()
# --- START TIMER ---

# Using the validated positional argument syntax
profile = mp.compute(time_series_values, WINDOW_SIZE)

# --- END TIMER ---
end_time = time.time()

duration = end_time - start_time

print(f"\nBenchmark Results (MatrixProfile Lib):")
print(f"-------------------------------------------")
print(f"Execution time: {duration:.4f} seconds")
print(f"-------------------------------------------")

# Quick sanity check
mp_vector = profile['mp']
k1_idx = np.argmax(mp_vector)
print(f"Sanity check: Highest peak found at index {k1_idx}")