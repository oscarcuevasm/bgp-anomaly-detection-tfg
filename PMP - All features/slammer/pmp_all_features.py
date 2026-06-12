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

import os
import pandas as pd
import matrixprofile as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- 1. CONFIGURATION ---
FILE_BASELINE = "./Slammer_RRC04_Raw/slammer_baseline_11jan_features_1s.csv"
OUTPUT_DIR = "./PMP_Resultados_Slammer" # Directory where PNGs will be saved
WINDOWS_TO_TEST = range(5, 305, 5)

# Full list of BGP features to analyse
COLUMNS_TO_ANALYZE = [
    "announcements", "withdrawals", "nlri_ann", "dups", "origin_0", 
    "origin_2", "origin_changes", "as_path_max", "unique_as_path_max", 
    "edit_distance_avg", "edit_distance_max", "edit_distance_dict_0", 
    "edit_distance_dict_1", "edit_distance_dict_2", "edit_distance_dict_3", 
    "edit_distance_dict_4", "edit_distance_dict_5", "edit_distance_dict_6", 
    "edit_distance_unique_dict_0", "edit_distance_unique_dict_1", "imp_wd", 
    "imp_wd_spath", "imp_wd_dpath", "number_rare_ases", "rare_ases_avg", 
    "flaps", "nadas"
]

print(f"--- STARTING MULTI-COLUMN PMP: SLAMMER BASELINE (11 JAN 2003) ---")

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print(f"Loading large dataset: {FILE_BASELINE}...")
    df = pd.read_csv(FILE_BASELINE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
except Exception as e:
    print(f"ERROR loading file: {e}")
    exit()

# --- 2. MAIN LOOP OVER EACH FEATURE COLUMN ---
for idx_col, column_name in enumerate(COLUMNS_TO_ANALYZE, 1):
    print("\n" + "="*60)
    print(f"[{idx_col}/{len(COLUMNS_TO_ANALYZE)}] Analizando: {column_name}")
    print("="*60)
    
    # 2.1 Prepare data for the specific feature column
    if column_name not in df.columns:
        print(f"  -> WARNING: Column {column_name} not found in CSV. Skipping...")
        continue
        
    ts_data = df[column_name].resample('1S').sum().fillna(0)
    data_values = ts_data.values.astype(float)
    
    # If the column contains only zeros, skip computation to save time
    if np.max(data_values) == 0:
        print(f"  -> Column {column_name} contains only zeros. Skipping computation to save time...")
        continue

    # 2.2 Pan Matrix Profile computation
    print(f"  -> Computing Pan Matrix Profile (m from 5 to 300)...")
    pmp_matrix = []
    
    for idx, w in enumerate(WINDOWS_TO_TEST, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"     * Processing window m={w}s...")
            
        profile = mp.compute(data_values, windows=w)
        mp_values = profile['mp']
        pad_length = len(data_values) - len(mp_values)
        mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
        pmp_matrix.append(mp_padded)

    pmp_matrix = np.array(pmp_matrix)

    # 2.3 Plot and save
    print(f"  -> Generating and saving visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

    # Panel Superior (Raw Data)
    ax1.plot(ts_data.index, ts_data.values, color='tab:blue', linewidth=1)
    ax1.set_title(f'BGP Volume ({column_name}) - 11 Jan 2003', fontweight='bold')
    ax1.set_ylabel('Value / s')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel Inferior (PMP)
    cmap = plt.cm.inferno
    # Let vmax auto-adjust to the 99th percentile to highlight each individual metric better
    vmax_auto = np.nanpercentile(pmp_matrix, 99) if not np.isnan(pmp_matrix).all() else 25
    
    c = ax2.pcolormesh(ts_data.index, list(WINDOWS_TO_TEST), pmp_matrix, cmap=cmap, shading='auto', vmin=0, vmax=vmax_auto)
    ax2.set_title(f'Pan Matrix Profile - {column_name}', fontweight='bold')
    ax2.set_ylabel('Window m (s)')
    ax2.set_xlabel('Local Time (UTC)', fontweight='bold')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    fig.colorbar(c, ax=ax2, label='Euclidean Distance')

    plt.tight_layout()
    
    # Save to file and close figure to free RAM
    output_filename = os.path.join(OUTPUT_DIR, f"pmp_11jan_{column_name}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig) 
    
    print(f"  -> ✓ Guardado como: {output_filename}")

print("\n" + "="*60)
print("MULTI-COLUMN PROCESS COMPLETED!")
print(f"All plots saved to: {OUTPUT_DIR}")
print("="*60)