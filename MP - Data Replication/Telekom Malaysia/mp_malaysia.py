import pandas as pd
import matrixprofile as mp 
import numpy as np
import matplotlib.pyplot as plt

# --- 0. CONFIGURACIÓN ---
FILE_NAME = "dataset_malaysian-telecom_513_1_rrc04.csv"
FEATURE_NAME = "announcements" 

np.set_printoptions(suppress=True, precision=2) 

print(f"--- Ejecutando Replicación: Telekom Malaysia (Corregido y Filtrado) ---")

# --- 1. CARGA DE DATOS ---
try:
    df = pd.read_csv(FILE_NAME, header=0)
    time_series = df[FEATURE_NAME].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Longitud: {N} minutos.")
except Exception as e:
    print(f"Error carga: {e}")
    exit()

# --- 2. ANÁLISIS ---
WINDOW_SIZE = 120
K_DISCORDS = 8 

print(f"Calculando Matrix Profile (m={WINDOW_SIZE})...")
profile = mp.compute(time_series_values, WINDOW_SIZE)
mp_vector = profile['mp']

# --- CORRECCIÓN DE RUIDO Y DIMENSIONES ---
print("Calculando filtro de ruido...")
# 1. Calcular desviación estándar móvil (Longitud 7200)
rolling_std_full = pd.Series(time_series_values).rolling(window=WINDOW_SIZE).std().fillna(0).values

# 2. ALINEACIÓN CRÍTICA (Fix del IndexError)
# El Matrix Profile es más corto que la serie original. 
# Debemos recortar la desviación estándar para que coincidan.
rolling_std_aligned = rolling_std_full[WINDOW_SIZE-1:]

# Verificación de seguridad y recorte final si difieren por 1
min_len = min(len(rolling_std_aligned), len(mp_vector))
rolling_std_aligned = rolling_std_aligned[:min_len]
mp_vector = mp_vector[:min_len]

# 3. Aplicar Filtro
NOISE_THRESHOLD = 5.0 
mp_vector_filtered = np.copy(mp_vector)
# Ahora las dimensiones coinciden perfectamente
mp_vector_filtered[rolling_std_aligned < NOISE_THRESHOLD] = 0.0

# --- 3. DETECCIÓN DE DISCORDIAS ---
print(f"Descubriendo las top-{K_DISCORDS} discordias (Filtradas)...")
mp_vector_copy = np.copy(mp_vector_filtered)
discords_indices = []

for k in range(K_DISCORDS):
    k_index = np.argmax(mp_vector_copy)
    # Solo añadimos si es una anomalía real (MP > 0)
    if mp_vector_copy[k_index] > 0:
        discords_indices.append(k_index)
    
    start_exclusion = max(0, k_index - WINDOW_SIZE)
    end_exclusion = min(len(mp_vector_copy), k_index + WINDOW_SIZE)
    mp_vector_copy[start_exclusion:end_exclusion] = np.NINF 

discords_indices.sort()

print(f"\n--- RESULTADOS FILTRADOS ---")
print(f"Se encontraron {len(discords_indices)} alarmas relevantes.")
for i, index in enumerate(discords_indices):
    print(f"Alarma {i+1}: Minuto {index}")


# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(18, 10))
plt.suptitle(f'Telekom Malaysia: Detección Filtrada (Peer Único)', fontsize=16)

# Subplot 1: Datos
plt.subplot(2, 1, 1) 
plt.plot(time_series, label='Datos BGP (Volumen)', color='C0')
if discords_indices:
    # Sombrear la zona del evento detectado
    plt.axvspan(min(discords_indices), max(discords_indices) + WINDOW_SIZE, color='red', alpha=0.1, label='Zona de Evento Detectado')
plt.title(f'Serie Temporal (Datos)')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

# Subplot 2: Matrix Profile
plt.subplot(2, 1, 2)
# Ajustar eje X para visualización alineada
mp_index_offset = np.arange(len(mp_vector))
plt.plot(mp_index_offset, mp_vector, label='Matrix Profile Original (con ruido)', color='gray', alpha=0.3)
plt.plot(mp_index_offset, mp_vector_filtered, label='Matrix Profile Filtrado', color='green')

# Marcar las discordias
for index in discords_indices:
    plt.plot(index, mp_vector[index], marker='*', markersize=14, color='red', linestyle='None', label='Alarma')

plt.title(f'Vector Matrix Profile (Alarmas concentradas en el evento)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Distancia MP')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()