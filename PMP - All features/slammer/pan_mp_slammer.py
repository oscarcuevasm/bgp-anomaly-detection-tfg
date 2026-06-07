import pandas as pd
import matrixprofile as mp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# --- 1. CONFIGURACIÓN ---
FILE_BASELINE = "./Slammer_RRC04_Raw/slammer_baseline_18jan_features_1s.csv"
COLUMN_NAME = "announcements"  
WINDOWS_TO_TEST = range(5, 305, 5)

print(f"--- INICIANDO PMP: LÍNEA BASE SLAMMER (18 ENERO 2003) ---")

try:
    print(f"Cargando {FILE_BASELINE}...")
    df = pd.read_csv(FILE_BASELINE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df.set_index('window_start', inplace=True)
    ts_data = df[COLUMN_NAME].resample('1S').sum().fillna(0)
    print(f" -> {len(ts_data)} segundos listos para analizar.")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# --- 2. CÁLCULO DE PAN MATRIX PROFILE ---
print("\nCalculando Pan Matrix Profile (m de 5 a 300)...")
pmp_matrix = []
data_values = ts_data.values.astype(float)

for idx, w in enumerate(WINDOWS_TO_TEST, 1):
    if idx % 10 == 0 or idx == 1:
        print(f"  -> Calculando ventana m={w}s...")
    profile = mp.compute(data_values, windows=w)
    mp_values = profile['mp']
    pad_length = len(data_values) - len(mp_values)
    mp_padded = np.pad(mp_values, (0, pad_length), constant_values=np.nan)
    pmp_matrix.append(mp_padded)

pmp_matrix = np.array(pmp_matrix)

# --- 3. GRAFICAR ---
print("\nGenerando visualización...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2.5]})

# Panel Superior (Raw Data)
ax1.plot(ts_data.index, ts_data.values, color='tab:blue', linewidth=1)
ax1.set_title('Volumen BGP (Announcements) - 18 Ene 2003 (Día Normal)', fontweight='bold')
ax1.set_ylabel('Paquetes / s')
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel Inferior (PMP)
cmap = plt.cm.inferno
# Mantenemos el mismo vmax=25 para que ambas gráficas sean directamente comparables
c = ax2.pcolormesh(ts_data.index, list(WINDOWS_TO_TEST), pmp_matrix, cmap=cmap, shading='auto', vmin=0, vmax=25)
ax2.set_title('Pan Matrix Profile - Comportamiento Normal', fontweight='bold')
ax2.set_ylabel('Ventana m (s)')
ax2.set_xlabel('Hora Local (UTC)', fontweight='bold')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
fig.colorbar(c, ax=ax2, label='Distancia Euclidiana (Escala Compartida)')

plt.tight_layout()
plt.show()