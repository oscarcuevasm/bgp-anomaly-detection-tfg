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
import matplotlib.pyplot as plt

# --- 0. GLOBAL CONFIGURATION ---
FILE_NAME = "Code_Red_I.csv" # Ensure this matches your input CSV filename
FEATURE_INDEX = 4 # Column 5 (index 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) # Suppress scientific notation when printing MP values

print(f"--- Running Dual Replication: Code Red I Incident ---")
print(f"Loading data from {FILE_NAME}...")

# --- 1. DATA LOADING (done once) ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Time series length (N): {N} minutes.")
except Exception as e:
    print(f"ERROR: Could not load {FILE_NAME}. Error: {e}")
    exit()

# --- 2. ANALYSIS 1: m=599 (Figura 1) ---
print("\n--- ANÁLISIS 1 (m=599) ---")
WINDOW_SIZE_1 = 599
K_DISCORDS_1 = 2

print(f"Computing Matrix Profile (MP) with m = {WINDOW_SIZE_1}...")
profile_1 = mp.compute(time_series_values, WINDOW_SIZE_1)
mp_vector_1 = profile_1['mp']

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

print("\n--- RESULTS (m=599) ---")
k1_index_m1 = discords_indices_1_sorted[0]
k2_index_m1 = discords_indices_1_sorted[1]
print(f"k1 (Highest peak): Start at minute {k1_index_m1} (MP value: {mp_vector_1[k1_index_m1]:.2f})")
print(f"k2 (Secondary peak): Start at minute {k2_index_m1} (MP value: {mp_vector_1[k2_index_m1]:.2f})")
print("Validation: The paper (Fig. 1) shows peaks at ~3320 and ~3800 min.")


# --- 3. ANALYSIS 2: m=1440 (Figura 2) ---
print("\n--- ANALYSIS 2 (m=1440) ---")
WINDOW_SIZE_2 = 1440
K_DISCORDS_2 = 2 # Searching for 2 as a safety margin; Fig. 2 shows only 1

print(f"Computing Matrix Profile (MP) with m = {WINDOW_SIZE_2}...")
profile_2 = mp.compute(time_series_values, WINDOW_SIZE_2)
mp_vector_2 = profile_2['mp']

print(f"Discovering top-{K_DISCORDS_2} discords (with manual exclusion zone)...")
mp_vector_copy_2 = np.copy(mp_vector_2)
discords_indices_2 = []

for k in range(K_DISCORDS_2):
    k_index = np.argmax(mp_vector_copy_2)
    discords_indices_2.append(k_index)
    start_exclusion = max(0, k_index - WINDOW_SIZE_2)
    end_exclusion = min(len(mp_vector_copy_2), k_index + WINDOW_SIZE_2)
    mp_vector_copy_2[start_exclusion:end_exclusion] = np.NINF 

# Sort by MP magnitude
mp_values_2 = mp_vector_2[discords_indices_2]
sort_by_mp_2 = np.argsort(mp_values_2)[::-1]
discords_indices_2_sorted = [discords_indices_2[i] for i in sort_by_mp_2]

print(f"\n--- RESULTS (m=1440) ---")
k1_index_m2 = discords_indices_2_sorted[0]
print(f"k1 (Highest peak): Start at minute {k1_index_m2} (MP value: {mp_vector_2[k1_index_m2]:.2f})")
print("Validation: The paper (Fig. 2) reports the 1st discord at ~2900 min.")


# --- 4. VISUALIZATION ---

# Plot 1 (m=599)
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Code Red I: Análisis m={WINDOW_SIZE_1} (Comparación Fig. 1)', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
plt.title(f'Time Series (Data)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile (MP)', color='C0')
# Mark k=2 discords
for i, index in enumerate(discords_indices_1_sorted):
    plt.plot(index, mp_vector_1[index], marker='*', markersize=10, color='red', linestyle='None')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Plot 2 (m=1440)
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Code Red I: Análisis m={WINDOW_SIZE_2} (Comparación Fig. 2)', fontsize=16)

plt.subplot(2, 1, 1) 
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
# Mark the subsequence of the 1st discord (k1)
start = discords_indices_2_sorted[0]
end = start + WINDOW_SIZE_2
plt.plot(np.arange(start, end), time_series.iloc[start:end], c='red', 
         label=f'1st Discord (Start: {start} min)')
plt.title(f'Serie Temporal (Datos) y 1ra Discordia')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP)', color='green')
# Mark only k1 discord
plt.plot(discords_indices_2_sorted[0], mp_vector_2[discords_indices_2_sorted[0]], 
         marker='^', markersize=10, color='red', markeredgecolor='black')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Minimum Euclidean Distance')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nDisplaying plots...")
plt.show()