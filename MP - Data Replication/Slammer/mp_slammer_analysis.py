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

# --- 1. ANALYSIS CONFIGURATION (SLAMMER) ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868  # m1 = 868 minutes (reference anomalous period)
K_DISCORDS = 2     # k = 2 (top discords to find)
FEATURE_INDEX = 4  # Column 5 (index 4) = "Number of announcements"

print(f"--- Running Re-Analysis: Slammer Incident (Validated Method) ---")
print(f"Starting analysis of {FILE_NAME} with m={WINDOW_SIZE} using matrixprofile...")

# --- 2. DATA LOADING AND TIME SERIES SELECTION ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Time series length (N): {N} minutes.")
except Exception as e:
    print(f"ERROR: Could not load {FILE_NAME}. Error: {e}")
    exit()

# --- 3. MATRIX PROFILE COMPUTATION ---
print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE}...")
profile = mp.compute(time_series_values, WINDOW_SIZE)
mp_vector = profile['mp']

# --- 4. DISCORD DETECTION (CORRECT MANUAL METHOD) ---
print(f"Descubriendo las top-{K_DISCORDS} discordias (con Zona de Exclusión Manual)...")

mp_vector_copy = np.copy(mp_vector)
discords_indices = []

for k in range(K_DISCORDS):
    # 1. Find the index of the current highest peak
    k_index = np.argmax(mp_vector_copy)
    discords_indices.append(k_index)
    
    # 2. Define the Exclusion Zone
    start_exclusion = max(0, k_index - WINDOW_SIZE)
    end_exclusion = min(len(mp_vector_copy), k_index + WINDOW_SIZE)
    
    # 3. Flatten (set to -inf) the exclusion zone
    mp_vector_copy[start_exclusion:end_exclusion] = np.NINF 

# Sort by MP magnitude
mp_values = mp_vector[discords_indices]
sort_by_mp = np.argsort(mp_values)[::-1]
discords_indices_sorted = [discords_indices[i] for i in sort_by_mp]

print("\n--- OBTAINED RESULTS (Slammer) ---")
k1_index = discords_indices_sorted[0]
k2_index = discords_indices_sorted[1]
print(f"k1 (Highest peak): Start at minute {k1_index} (MP value: {mp_vector[k1_index]:.2f})")
print(f"k2 (Secondary peak): Start at minute {k2_index} (MP value: {mp_vector[k2_index]:.2f})")

# Validate key result reported in the paper (page 6)
# Known anomalous period from RNNs: 3200-4068 min
print(f"\nKEY VALIDATION (k2): Obtained index is {k2_index}.")
print("The paper reported an early alarm (k2) at 3030 min (before the 3200 min mark).")
    

# --- 5. VISUALIZATION (Comparison with Figure 6) ---
plt.figure(figsize=(18, 10))
plt.suptitle(f'Slammer: Análisis m={WINDOW_SIZE} (Comparación Fig. 6)', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Volume (Announcements)', color='blue', alpha=0.6)
colors = ['red', 'purple']
labels = ['1st Discord', '2nd Discord']

for i, index in enumerate(discords_indices_sorted):
    start = index
    end = start + WINDOW_SIZE
    plt.plot(np.arange(start, end), time_series.iloc[start:end], c=colors[i], 
             label=f'{labels[i]} (Inicio: {index} min)')
    plt.plot(start, time_series.iloc[start], marker='*', markersize=10, color=colors[i], markeredgecolor='black')

# Mark the known anomalous period (3200 to 4068 min)
plt.axvspan(3200, 4068, color='gray', alpha=0.2, label='Anomalous Period (RNNs)')
plt.title(f'BGP Time Series')
plt.ylabel('BGP Volume (Announcements)')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector)), mp_vector, label='Matrix Profile', color='green')
for i, index in enumerate(discords_indices_sorted):
    plt.plot(index, mp_vector[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')
plt.title(f'Matrix Profile Vector (MP)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Minimum Euclidean Distance')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("\nDisplaying plots...")
plt.show()