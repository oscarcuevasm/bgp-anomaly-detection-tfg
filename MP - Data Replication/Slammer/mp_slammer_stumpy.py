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
import matplotlib.pyplot as plt

# --- 1. ANALYSIS CONFIGURATION ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868  # m1 = 868 minutes
K_DISCORDS = 2     # k = 2 (top discords to find)
FEATURE_INDEX = 4  # Column 5 = "Number of announcements"

print("--- Running Final Solution (NumPy ArgSort) ---")
print(f"Starting analysis of {FILE_NAME} for Slammer with m={WINDOW_SIZE}...")

# --- 2. DATA LOADING AND TIME SERIES SELECTION ---
df = pd.read_csv(FILE_NAME, header=None)
time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
N = len(time_series)

# --- 3. MATRIX PROFILE COMPUTATION WITH STUMPY ---
# This is the base computation and should always work as it is pure numerical computation.
print(f"Computing Matrix Profile with STUMPY...")
mp_result = stumpy.stump(time_series, m=WINDOW_SIZE)
mp_vector = mp_result[:, 0]  # Extract only the MP vector

# --- 4. DISCORD DETECTION AND VALIDATION (MANUAL WITH NUMPY) ---
print(f"Searching for top-{K_DISCORDS} discords with NumPy...")

# 1. np.argsort sorts indices by magnitude in ascending order.
# 2. [::-1] reverses the order (descending, from largest to smallest).
# 3. [:K_DISCORDS] takes the top K indices (MP peaks).
# NOTE: This method finds the 2 LARGEST anomalies without applying the exclusion zone,
# but provides a close validation of the key time points.
sorted_indices = np.argsort(mp_vector)[::-1]
discords_indices = sorted_indices[:K_DISCORDS]
discords_indices = list(np.sort(discords_indices)) # Sort so that k1 < k2 by time index

# Re-sort based on MP value (k1 must be the largest)
# For the k1/k2 definition, we order them by descending MP value.
mp_values = mp_vector[discords_indices]
# Get the indices that would sort mp_values from largest to smallest
sort_by_mp = np.argsort(mp_values)[::-1]
# Apply the ordering to the discord indices
discords_indices = [discords_indices[i] for i in sort_by_mp]


print("\n--- RESULTADOS OBTENIDOS ---")
for i, index in enumerate(discords_indices):
    discord_rank = i + 1
    print(f"k{discord_rank}: Inicio en el minuto {index}")
    
# Validate the key result reported in the paper (page 6)
if len(discords_indices) >= 2:
    k2_index = discords_indices[1]
    # Known anomalous period from RNNs: 3200-4068 min
    print(f"\nKEY VALIDATION (k2): Obtained index is {k2_index}.")
    print("El artículo predijo un inicio temprano para k2 en ~3030 min (antes de 3200 min).")


# --- 5. VISUALIZATION ---
# ... (Visualization code continues as before, using discords_indices)
plt.figure(figsize=(18, 10))

# Subplot 1: Time series with marked discords
plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Volume (Announcements)', color='blue', alpha=0.6)
colors = ['red', 'purple']
labels = ['1st Discord', '2nd Discord']

for i, index in enumerate(discords_indices):
    start = index
    end = start + WINDOW_SIZE
    plt.plot(np.arange(start, end), time_series.iloc[start:end], c=colors[i], 
             label=f'{labels[i]} (Inicio: {start} min)')
    plt.plot(start, time_series.iloc[start], marker='*', markersize=10, color=colors[i], markeredgecolor='black')

plt.axvspan(3200, 4068, color='gray', alpha=0.2, label='Anomalous Period (RNNs)')
plt.title(f'Slammer: Discord Detection in BGP Volume ($m={WINDOW_SIZE}$) [NumPy ArgSort]')
plt.ylabel('BGP Volume (Announcements)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Subplot 2: Matrix Profile
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector)), mp_vector, label='Matrix Profile', color='green')

for i, index in enumerate(discords_indices):
    plt.plot(index, mp_vector[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')

plt.title(f'Matrix Profile Vector (MP) - Peaks = Discords')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('Minimum Euclidean Distance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()