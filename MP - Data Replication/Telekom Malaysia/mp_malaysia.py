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

# --- 0. CONFIGURATION ---
FILE_NAME = "dataset_malaysian-telecom_513_1_rrc04.csv"
FEATURE_NAME = "announcements"

np.set_printoptions(suppress=True, precision=2) 

print(f"--- Running Replication: Telekom Malaysia (Corrected and Filtered) ---")

# --- 1. DATA LOADING ---
try:
    df = pd.read_csv(FILE_NAME, header=0)
    time_series = df[FEATURE_NAME].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Length: {N} minutes.")
except Exception as e:
    print(f"Loading error: {e}")
    exit()

# --- 2. ANALYSIS ---
WINDOW_SIZE = 120
K_DISCORDS = 8 

print(f"Calculando Matrix Profile (m={WINDOW_SIZE})...")
profile = mp.compute(time_series_values, WINDOW_SIZE)
mp_vector = profile['mp']

# --- NOISE FILTERING AND DIMENSION ALIGNMENT ---
print("Computing noise filter...")
# 1. Compute rolling standard deviation (length 7200)
rolling_std_full = pd.Series(time_series_values).rolling(window=WINDOW_SIZE).std().fillna(0).values

# 2. CRITICAL ALIGNMENT (fix for IndexError)
# The Matrix Profile is shorter than the original series.
# The rolling std must be trimmed to match its length.
rolling_std_aligned = rolling_std_full[WINDOW_SIZE-1:]

# Safety check: trim to equal length if they differ by 1
min_len = min(len(rolling_std_aligned), len(mp_vector))
rolling_std_aligned = rolling_std_aligned[:min_len]
mp_vector = mp_vector[:min_len]

# 3. Apply noise filter
NOISE_THRESHOLD = 5.0
mp_vector_filtered = np.copy(mp_vector)
# Dimensions now match exactly
mp_vector_filtered[rolling_std_aligned < NOISE_THRESHOLD] = 0.0

# --- 3. DISCORD DETECTION ---
print(f"Discovering top-{K_DISCORDS} discords (filtered)...")
mp_vector_copy = np.copy(mp_vector_filtered)
discords_indices = []

for k in range(K_DISCORDS):
    k_index = np.argmax(mp_vector_copy)
    # Only add if it is a genuine anomaly (MP > 0)
    if mp_vector_copy[k_index] > 0:
        discords_indices.append(k_index)
    
    start_exclusion = max(0, k_index - WINDOW_SIZE)
    end_exclusion = min(len(mp_vector_copy), k_index + WINDOW_SIZE)
    mp_vector_copy[start_exclusion:end_exclusion] = np.NINF 

discords_indices.sort()

print(f"\n--- FILTERED RESULTS ---")
print(f"Se encontraron {len(discords_indices)} alarmas relevantes.")
for i, index in enumerate(discords_indices):
    print(f"Alarm {i+1}: Minute {index}")


# --- 4. VISUALIZATION ---
plt.figure(figsize=(18, 10))
plt.suptitle(f'Telekom Malaysia: Filtered Detection (Single Peer)', fontsize=16)

# Subplot 1: Datos
plt.subplot(2, 1, 1) 
plt.plot(time_series, label='BGP Data (Volume)', color='C0')
if discords_indices:
    # Shade the detected event zone
    plt.axvspan(min(discords_indices), max(discords_indices) + WINDOW_SIZE, color='red', alpha=0.1, label='Zona de Evento Detectado')
plt.title(f'Time Series (Data)')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

# Subplot 2: Matrix Profile
plt.subplot(2, 1, 2)
# Adjust X axis for aligned visualization
mp_index_offset = np.arange(len(mp_vector))
plt.plot(mp_index_offset, mp_vector, label='Matrix Profile Original (con ruido)', color='gray', alpha=0.3)
plt.plot(mp_index_offset, mp_vector_filtered, label='Matrix Profile Filtrado', color='green')

# Mark discords
for index in discords_indices:
    plt.plot(index, mp_vector[index], marker='*', markersize=14, color='red', linestyle='None', label='Alarma')

plt.title(f'Matrix Profile Vector (Alarms concentrated around the event)')
plt.xlabel(f'Time Index (minutes)')
plt.ylabel('MP Distance')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()