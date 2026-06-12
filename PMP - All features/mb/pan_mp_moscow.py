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
FILE_BASELINE = "./Moscow_Blackout_RRC04_Raw/moscow_baseline_18may_features_1s.csv"
FILE_ANOMALY = "./Moscow_Blackout_RRC04_Raw/moscow_blackout_25may_features_1s.csv"
COLUMN_NAME = "withdrawals"  
WINDOWS_TO_TEST = range(5, 305, 5)

print("--- STARTING COMPARATIVE PMP (TWO INDEPENDENT FIGURES) ---")

def load_and_prep(filepath):
    print(f"Cargando {filepath}...")
    df = pd.read_csv(filepath)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
    # Resample filling gaps so that the series is continuous
    ts = df[COLUMN_NAME].resample('1S').sum().fillna(0)
    return ts

def calc_pmp(ts, name):
    print(f"\nComputing PMP for {name}...")
    pmp_matrix = []
    data_values = ts.values.astype(float)
    total = len(WINDOWS_TO_TEST)
    
    for idx, w in enumerate(WINDOWS_TO_TEST, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  -> Window {idx}/{total} (m={w}s)")
        profile = mp.compute(data_values, windows=w)
        mp_values = profile['mp']
        # Pad with NaNs at the end to form the matrix
        pad_length = len(data_values) - len(mp_values)
        mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
        pmp_matrix.append(mp_padded)
        
    return np.array(pmp_matrix)

# --- 2. DATA LOADING AND PROCESSING ---
try:
    ts_18 = load_and_prep(FILE_BASELINE)
    ts_25 = load_and_prep(FILE_ANOMALY)
except Exception as e:
    print(f"ERROR: {e}")
    exit()

pmp_18 = calc_pmp(ts_18, "Day 18 (Baseline)")
pmp_25 = calc_pmp(ts_25, "Day 25 (Blackout)")

# --- 3. GLOBAL COLOUR SCALE ---
# Compute the global maximum so both plots share the same colour scale.
# If day 25 reaches distance "20" and day 18 only "5", the day-18 heatmap will appear dark.
global_vmax = np.nanmax([np.nanmax(pmp_18), np.nanmax(pmp_25)])
print(f"\nGenerating plots... (Maximum intensity detected = {global_vmax:.2f})")

cmap = plt.cm.inferno

# ==============================================================================
# --- 4. FIGURE 1: NORMAL DAY (18 MAY) ---
# ==============================================================================
fig1, (ax1_raw, ax1_pmp) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
fig1.canvas.manager.set_window_title('Baseline - 18 May')

ax1_raw.plot(ts_18.index, ts_18.values, color='tab:blue', linewidth=1)
ax1_raw.set_title('BGP Volume (Withdrawals) - 18 May 2005 (Normal Day)', fontweight='bold')
ax1_raw.set_ylabel('Packets / s')
ax1_raw.grid(True, linestyle='--', alpha=0.5)

c1 = ax1_pmp.pcolormesh(ts_18.index, list(WINDOWS_TO_TEST), pmp_18, cmap=cmap, shading='auto', vmin=0, vmax=global_vmax)
ax1_pmp.set_title('Pan Matrix Profile - Normal Behaviour', fontweight='bold')
ax1_pmp.set_ylabel('Ventana m (s)')
ax1_pmp.set_xlabel('Time (UTC)', fontweight='bold')
ax1_pmp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig1.colorbar(c1, ax=ax1_pmp, label='Euclidean Distance (Global Scale)')
fig1.autofmt_xdate()
fig1.tight_layout()

# ==============================================================================
# --- 5. FIGURE 2: BLACKOUT DAY (25 MAY) ---
# ==============================================================================
fig2, (ax2_raw, ax2_pmp) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
fig2.canvas.manager.set_window_title('Anomaly - 25 May')

ax2_raw.plot(ts_25.index, ts_25.values, color='crimson', linewidth=1)
ax2_raw.set_title('BGP Volume (Withdrawals) - 25 May 2005 (Blackout Day)', fontweight='bold')
ax2_raw.set_ylabel('Packets / s')
ax2_raw.grid(True, linestyle='--', alpha=0.5)

c2 = ax2_pmp.pcolormesh(ts_25.index, list(WINDOWS_TO_TEST), pmp_25, cmap=cmap, shading='auto', vmin=0, vmax=global_vmax)
ax2_pmp.set_title('Pan Matrix Profile - Detected Anomaly', fontweight='bold')
ax2_pmp.set_ylabel('Ventana m (s)')
ax2_pmp.set_xlabel('Time (UTC)', fontweight='bold')
ax2_pmp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig2.colorbar(c2, ax=ax2_pmp, label='Euclidean Distance (Global Scale)')
fig2.autofmt_xdate()
fig2.tight_layout()

# --- 6. SHOW BOTH FIGURES ---
print("Opening plot windows...")
plt.show()