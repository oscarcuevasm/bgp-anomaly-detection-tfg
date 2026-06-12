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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- 1. CONFIGURATION ---
FILE_BASELINE = "./CodeRed_RRC04_Raw/codered_baseline_12jul_features_1s.csv"
COLUMN_NAME = "edit_distance_dict_1"  
WINDOWS_TO_TEST = range(5, 305, 5)

print(f"--- STARTING PMP: CODE RED BASELINE (12 JULY 2001) ---")

try:
    print(f"Loading {FILE_BASELINE}...")
    df = pd.read_csv(FILE_BASELINE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
    ts_data = df[COLUMN_NAME].resample('1S').sum().fillna(0)
    print(f" -> {len(ts_data)} seconds ready to analyse.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# --- 2. PAN MATRIX PROFILE COMPUTATION ---
print("\nComputing Pan Matrix Profile (m from 5 to 300)...")
pmp_matrix = []
data_values = ts_data.values.astype(float)

for idx, w in enumerate(WINDOWS_TO_TEST, 1):
    if idx % 10 == 0 or idx == 1:
        print(f"  -> Computing window m={w}s...")
    profile = mp.compute(data_values, windows=w)
    mp_values = profile['mp']
    pad_length = len(data_values) - len(mp_values)
    mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
    pmp_matrix.append(mp_padded)

pmp_matrix = np.array(pmp_matrix)

# --- 3. PLOT ---
print("\nGenerating visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

# Panel Superior (Raw Data)
ax1.plot(ts_data.index, ts_data.values, color='tab:blue', linewidth=1)
ax1.set_title('BGP Volume (edit_distance_dict_1) - 12 Jul 2001 (Normal Day)', fontweight='bold')
ax1.set_ylabel('Packets / s')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel Inferior (PMP)
cmap = plt.cm.inferno
# Keep vmax=25 fixed so both plots share the same colour scale
c = ax2.pcolormesh(ts_data.index, list(WINDOWS_TO_TEST), pmp_matrix, cmap=cmap, shading='auto', vmin=0, vmax=25)
ax2.set_title('Pan Matrix Profile - Normal Behaviour', fontweight='bold')
ax2.set_ylabel('Window m (s)')
ax2.set_xlabel('Local Time (UTC)', fontweight='bold')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
fig.colorbar(c, ax=ax2, label='Euclidean Distance (Shared Scale)')

plt.tight_layout()
plt.show()