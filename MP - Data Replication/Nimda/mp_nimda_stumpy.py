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
import stumpy  # STUMPY library for Matrix Profile computation
import numpy as np
import matplotlib.pyplot as plt

# --- 0. GLOBAL CONFIGURATION ---
FILE_NAME = "Nimda.csv"
FEATURE_INDEX = 4  # Column 5 (index 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) # Suppress scientific notation when printing MP values

print(f"--- Running Dual Replication: Nimda Incident (using STUMPY) ---")
print(f"Loading data from {FILE_NAME}...")

# --- 1. DATA LOADING (done once) ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    # STUMPY prefers numpy arrays; not strictly required but good practice
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Time series length (N): {N} minutes.")
except Exception as e:
    print(f"ERROR: Could not load {FILE_NAME}. Error: {e}")
    exit()

# --- 2. ANALYSIS 1: m=1300 (Figura 5) ---
print("\n--- ANALYSIS 1 (m=1300) [STUMPY] ---")
WINDOW_SIZE_1 = 1300
K_DISCORDS_1 = 2

print(f"Computing Matrix Profile (MP) with m = {WINDOW_SIZE_1}...")
# COMPUTATION WITH STUMPY
# stumpy.stump returns a matrix. Column 0 is the MP vector.
mp_result_1 = stumpy.stump(time_series_values, m=WINDOW_SIZE_1)
mp_vector_1 = mp_result_1[:, 0]

print(f"Discovering top-{K_DISCORDS_1} discords (with manual exclusion zone)...")
mp_vector_copy_1 = np.copy(mp_vector_1)
discords_indices_1 = []

for k in range(K_DISCORDS_1):
    k_index = np.argmax(mp_vector_copy_1)
    discords_indices_1.append(k_index)
    start_exclusion = max(0, k_index - WINDOW_SIZE_1)
    end_exclusion = min(len(mp_vector_copy_1), k_index + WINDOW_SIZE_1)
    mp_vector_copy_1[start_exclusion:end_exclusion] = np.NINF 

# Sort by MP magnitude
mp_values_1 = mp_vector_1[discords_indices_1]
sort_by_mp_1 = np.argsort(mp_values_1)[::-1]
discords_indices_1_sorted = [discords_indices_1[i] for i in sort_by_mp_1]

print("\n--- RESULTS (m=1300) [STUMPY] ---")
k1_index = discords_indices_1_sorted[0]
k2_index = discords_indices_1_sorted[1]
print(f"k1 (Highest peak): Start at minute {k1_index} (MP value: {mp_vector_1[k1_index]:.2f})")
print(f"k2 (Secondary peak): Start at minute {k2_index} (MP value: {mp_vector_1[k2_index]:.2f})")
print("Validation: Compare with matrixprofile results (Peaks at ~4371 and ~1057).")


# --- 3. ANALYSIS 2: m=144 (Figura 4) ---
print("\n--- ANALYSIS 2 (m=144) [STUMPY] ---")
WINDOW_SIZE_2 = 144
K_DISCORDS_2 = 15 # k=15 to replicate multiple alarms

print(f"Computing Matrix Profile (MP) with m = {WINDOW_SIZE_2}...")
# COMPUTATION WITH STUMPY
mp_result_2 = stumpy.stump(time_series_values, m=WINDOW_SIZE_2)
mp_vector_2 = mp_result_2[:, 0]

print(f"Discovering top-{K_DISCORDS_2} discords (with manual exclusion zone)...")
mp_vector_copy_2 = np.copy(mp_vector_2)
discords_indices_2 = []

for k in range(K_DISCORDS_2):
    k_index = np.argmax(mp_vector_copy_2)
    discords_indices_2.append(k_index)
    start_exclusion = max(0, k_index - WINDOW_SIZE_2)
    end_exclusion = min(len(mp_vector_copy_2), k_index + WINDOW_SIZE_2)
    mp_vector_copy_2[start_exclusion:end_exclusion] = np.NINF 

discords_indices_2.sort()
print(f"\n--- RESULTADOS (m=144) [STUMPY] ---")
print(f"Found {len(discords_indices_2)} peaks (alarms).")
print("Validation: Compare Plot 2 visually with Figure 4 from the paper.")


# --- 4. VISUALIZATION ---
# (Visualization code is identical to the previous version)

# Plot 1 (m=1300)
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Nimda: Analysis m={WINDOW_SIZE_1} (Comparison Fig. 5) [STUMPY]', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Volume (Announcements)', color='blue', alpha=0.6)
colors = ['red', 'purple']
for i, index in enumerate(discords_indices_1_sorted):
    start = index
    end = start + WINDOW_SIZE_1
    plt.plot(np.arange(start, end), time_series.iloc[start:end], c=colors[i], 
             label=f'k{i+1} Discord (Start: {index} min)')
plt.title(f'BGP Time Series')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile (STUMPY)', color='green')
for i, index in enumerate(discords_indices_1_sorted):
    plt.plot(index, mp_vector_1[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Minimum Euclidean Distance')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Plot 2 (m=144)
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Nimda: Analysis m={WINDOW_SIZE_2} (Comparison Fig. 4) [STUMPY]', fontsize=16)

plt.subplot(2, 1, 1) 
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
plt.title(f'Time Series (Data)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP) [STUMPY]', color='C0')
for index in discords_indices_2:
    plt.plot(index, mp_vector_2[index], marker='*', markersize=10, color='red', linestyle='None', label='Discord (Alarm)')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nDisplaying plots...")
plt.show()