import pandas as pd
import matrixprofile as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- 1. CONFIGURACIÓN ---
FILE_BASELINE = "./Moscow_Blackout_RRC04_Raw/moscow_baseline_18may_features_1s.csv"
FILE_ANOMALY = "./Moscow_Blackout_RRC04_Raw/moscow_blackout_25may_features_1s.csv"
COLUMN_NAME = "withdrawals"  
WINDOWS_TO_TEST = range(5, 305, 5)

print("--- INICIANDO PMP COMPARATIVO (DOS FIGURAS INDEPENDIENTES) ---")

def load_and_prep(filepath):
    print(f"Cargando {filepath}...")
    df = pd.read_csv(filepath)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
    # Remuestreamos rellenando los vacíos para que la serie sea continua
    ts = df[COLUMN_NAME].resample('1S').sum().fillna(0)
    return ts

def calc_pmp(ts, name):
    print(f"\nCalculando PMP para {name}...")
    pmp_matrix = []
    data_values = ts.values.astype(float)
    total = len(WINDOWS_TO_TEST)
    
    for idx, w in enumerate(WINDOWS_TO_TEST, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  -> Ventana {idx}/{total} (m={w}s)")
        profile = mp.compute(data_values, windows=w)
        mp_values = profile['mp']
        # Añadir NaNs al final para poder crear la matriz
        pad_length = len(data_values) - len(mp_values)
        mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
        pmp_matrix.append(mp_padded)
        
    return np.array(pmp_matrix)

# --- 2. CARGA Y PROCESAMIENTO ---
try:
    ts_18 = load_and_prep(FILE_BASELINE)
    ts_25 = load_and_prep(FILE_ANOMALY)
except Exception as e:
    print(f"ERROR: {e}")
    exit()

pmp_18 = calc_pmp(ts_18, "Día 18 (Baseline)")
pmp_25 = calc_pmp(ts_25, "Día 25 (Apagón)")

# --- 3. ESCALA DE COLORES GLOBAL ---
# Calculamos el máximo general para que ambos gráficos usen la misma escala.
# Si el día 25 llega a "20" de distancia, y el 18 solo a "5", el mapa del 18 se verá oscuro.
global_vmax = np.nanmax([np.nanmax(pmp_18), np.nanmax(pmp_25)])
print(f"\nGenerando gráficas... (Intensidad máxima detectada = {global_vmax:.2f})")

cmap = plt.cm.inferno

# ==============================================================================
# --- 4. FIGURA 1: DÍA NORMAL (18 DE MAYO) ---
# ==============================================================================
fig1, (ax1_raw, ax1_pmp) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
fig1.canvas.manager.set_window_title('Línea Base - 18 Mayo')

ax1_raw.plot(ts_18.index, ts_18.values, color='tab:blue', linewidth=1)
ax1_raw.set_title('Volumen BGP (Withdrawals) - 18 Mayo 2005 (Día Normal)', fontweight='bold')
ax1_raw.set_ylabel('Paquetes / s')
ax1_raw.grid(True, linestyle='--', alpha=0.5)

c1 = ax1_pmp.pcolormesh(ts_18.index, list(WINDOWS_TO_TEST), pmp_18, cmap=cmap, shading='auto', vmin=0, vmax=global_vmax)
ax1_pmp.set_title('Pan Matrix Profile - Comportamiento Normal', fontweight='bold')
ax1_pmp.set_ylabel('Ventana m (s)')
ax1_pmp.set_xlabel('Hora (UTC)', fontweight='bold')
ax1_pmp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig1.colorbar(c1, ax=ax1_pmp, label='Distancia Euclidiana (Escala Global)')
fig1.autofmt_xdate()
fig1.tight_layout()

# ==============================================================================
# --- 5. FIGURA 2: DÍA DEL APAGÓN (25 DE MAYO) ---
# ==============================================================================
fig2, (ax2_raw, ax2_pmp) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})
fig2.canvas.manager.set_window_title('Anomalía - 25 Mayo')

ax2_raw.plot(ts_25.index, ts_25.values, color='crimson', linewidth=1)
ax2_raw.set_title('Volumen BGP (Withdrawals) - 25 Mayo 2005 (Día del Apagón)', fontweight='bold')
ax2_raw.set_ylabel('Paquetes / s')
ax2_raw.grid(True, linestyle='--', alpha=0.5)

c2 = ax2_pmp.pcolormesh(ts_25.index, list(WINDOWS_TO_TEST), pmp_25, cmap=cmap, shading='auto', vmin=0, vmax=global_vmax)
ax2_pmp.set_title('Pan Matrix Profile - Anomalía Detectada', fontweight='bold')
ax2_pmp.set_ylabel('Ventana m (s)')
ax2_pmp.set_xlabel('Hora (UTC)', fontweight='bold')
ax2_pmp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig2.colorbar(c2, ax=ax2_pmp, label='Distancia Euclidiana (Escala Global)')
fig2.autofmt_xdate()
fig2.tight_layout()

# --- 6. MOSTRAR AMBAS FIGURAS ---
print("Abriendo ventanas de las gráficas...")
plt.show()