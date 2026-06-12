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
FILE_NAME = "Moscow_blackout.csv" # Ensure this matches your input CSV filename
FEATURE_INDEX = 4  # Column 5 (index 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) 

print(f"--- Running Dual Replication: Moscow Blackout Incident ---")
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

# --- 2. ANALYSIS 1: m=120 (Figura 8) ---
print("\n--- ANÁLISIS 1 (m=120) ---")
WINDOW_SIZE_1 = 120
K_DISCORDS_1 = 12 # k=12 to replicate the multiple alarms in Fig. 8

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

discords_indices_1.sort()
print(f"\n--- RESULTADOS (m=120) ---")
print(f"Se encontraron {len(discords_indices_1)} picos (alarmas).")
print("Validation: Compare Plot 1 visually with Figure 8 from the paper.")


# --- 3. ANALYSIS 2: m=1440 (Figura 9) ---
print("\n--- ANALYSIS 2 (m=1440) ---")
WINDOW_SIZE_2 = 1440
K_DISCORDS_2 = 2 # k=2 for Top-k discord analysis

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
k2_index_m2 = discords_indices_2_sorted[1]
print(f"k1 (Highest peak): Start at minute {k1_index_m2} (MP value: {mp_vector_2[k1_index_m2]:.2f})")
print(f"k2 (Secondary peak): Start at minute {k2_index_m2} (MP value: {mp_vector_2[k2_index_m2]:.2f})")
print("\nValidation: The paper (Fig. 9) reports the 1st discord (k1) at ~2800 min.")


# --- 4. VISUALIZATION ---

# Plot 1 (m=120) - Comparison with Figure 8
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Moscow Blackout: Análisis m={WINDOW_SIZE_1} (Comparación Fig. 8)', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
plt.title(f'Time Series (Data)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile (MP)', color='C0')
# Mark ALL discords (k=12) with red stars
for index in discords_indices_1:
    plt.plot(index, mp_vector_1[index], marker='*', markersize=10, color='red', linestyle='None')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Plot 2 (m=1440) - Comparison with Figure 9
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Moscow Blackout: Análisis m={WINDOW_SIZE_2} (Comparación Fig. 9)', fontsize=16)

plt.subplot(2, 1, 1) 
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
# Mark the subsequence of the 1st discord (k1)
start = discords_indices_2_sorted[0]
end = start + WINDOW_SIZE_2
plt.plot(np.arange(start, end), time_series.iloc[start:end], c='red', 
         label=f'1st Discord (Start: {start} min)')
# Mark the 2nd discord
start_k2 = discords_indices_2_sorted[1]
end_k2 = start_k2 + WINDOW_SIZE_2
plt.plot(np.arange(start_k2, end_k2), time_series.iloc[start_k2:end_k2], c='purple', 
         label=f'2nd Discord (Start: {start_k2} min)')

plt.title(f'Time Series (Data) and Top-k Discords')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP)', color='green')
# Mark k=2 discords
for i, index in enumerate(discords_indices_2_sorted):
    plt.plot(index, mp_vector_2[index], marker='^', markersize=10, color='red', markeredgecolor='black')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Minimum Euclidean Distance')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nDisplaying plots...")
plt.show()