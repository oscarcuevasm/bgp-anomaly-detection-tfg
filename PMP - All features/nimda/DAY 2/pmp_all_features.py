import os
import pandas as pd
import matrixprofile as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- 1. CONFIGURACIÓN ---
FILE_BASELINE = "./Nimda_RRC04_Raw/nimda_baseline_12sept_features_1s.csv"
OUTPUT_DIR = "./PMP_Resultados_Nimda" # Carpeta donde se guardarán los PNGs
WINDOWS_TO_TEST = range(5, 305, 5)

# Lista completa de características a analizar
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

print(f"--- INICIANDO PMP MULTICOLUMNA: LÍNEA BASE NIMDA (12 SEPTIEMBRE 2001) ---")

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    print(f"Cargando dataset gigante: {FILE_BASELINE}...")
    df = pd.read_csv(FILE_BASELINE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
except Exception as e:
    print(f"ERROR al cargar el archivo: {e}")
    exit()

# --- 2. BUCLE PRINCIPAL POR CADA COLUMNA ---
for idx_col, column_name in enumerate(COLUMNS_TO_ANALYZE, 1):
    print("\n" + "="*60)
    print(f"[{idx_col}/{len(COLUMNS_TO_ANALYZE)}] Analizando: {column_name}")
    print("="*60)
    
    # 2.1 Preparar datos de la columna específica
    if column_name not in df.columns:
        print(f"  -> ADVERTENCIA: La columna {column_name} no existe en el CSV. Saltando...")
        continue
        
    ts_data = df[column_name].resample('1S').sum().fillna(0)
    data_values = ts_data.values.astype(float)
    
    # Si la columna está completamente vacía (puros ceros), no perdemos el tiempo calculando
    if np.max(data_values) == 0:
        print(f"  -> La columna {column_name} solo contiene ceros. Saltando cálculo para ahorrar tiempo...")
        continue

    # 2.2 Cálculo de Pan Matrix Profile
    print(f"  -> Calculando Pan Matrix Profile (m de 5 a 300)...")
    pmp_matrix = []
    
    for idx, w in enumerate(WINDOWS_TO_TEST, 1):
        if idx % 20 == 0 or idx == 1:
            print(f"     * Procesando ventana m={w}s...")
            
        profile = mp.compute(data_values, windows=w)
        mp_values = profile['mp']
        pad_length = len(data_values) - len(mp_values)
        mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
        pmp_matrix.append(mp_padded)

    pmp_matrix = np.array(pmp_matrix)

    # 2.3 Graficar y Guardar
    print(f"  -> Generando y guardando visualización...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

    # Panel Superior (Raw Data)
    ax1.plot(ts_data.index, ts_data.values, color='tab:blue', linewidth=1)
    ax1.set_title(f'Volumen BGP ({column_name}) - 12 Septiembre 2001 ', fontweight='bold')
    ax1.set_ylabel('Valor / s')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel Inferior (PMP)
    cmap = plt.cm.inferno
    # Aquí dejamos que vmax se ajuste automáticamente al percentil 99 para resaltar mejor cada métrica individual
    vmax_auto = np.nanpercentile(pmp_matrix, 99) if not np.isnan(pmp_matrix).all() else 25
    
    c = ax2.pcolormesh(ts_data.index, list(WINDOWS_TO_TEST), pmp_matrix, cmap=cmap, shading='auto', vmin=0, vmax=vmax_auto)
    ax2.set_title(f'Pan Matrix Profile - {column_name}', fontweight='bold')
    ax2.set_ylabel('Ventana m (s)')
    ax2.set_xlabel('Hora Local (UTC)', fontweight='bold')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    fig.colorbar(c, ax=ax2, label='Distancia Euclidiana')

    plt.tight_layout()
    
    # Guardar en archivo y cerrar figura para liberar memoria RAM
    output_filename = os.path.join(OUTPUT_DIR, f"pmp_12sept_{column_name}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close(fig) 
    
    print(f"  -> ✓ Guardado como: {output_filename}")

print("\n" + "="*60)
print("¡PROCESO MULTICOLUMNA COMPLETADO!")
print(f"Todas las gráficas están en la carpeta: {OUTPUT_DIR}")
print("="*60)