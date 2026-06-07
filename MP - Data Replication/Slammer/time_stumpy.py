import pandas as pd
import stumpy
import numpy as np
import time

# --- CONFIGURACIÓN ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print(f"--- BENCHMARK STUMPY: Slammer (m={WINDOW_SIZE}) ---")

# 1. CARGA DE DATOS (No medimos esto)
print(f"Cargando datos...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    # Convertir a array numpy explícito para evitar overhead de pandas durante el cálculo
    time_series_values = time_series.values.copy() 
except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()

# 2. CALENTAMIENTO (WARM-UP)
# Numba necesita compilar la función la primera vez que se ejecuta.
# Ejecutamos una versión muy pequeña para "calentar" el compilador JIT
# y que no afecte a la medición real.
print("Calentando compilador JIT (Numba)...")
stumpy.stump(time_series_values[:100], m=10) 

# 3. MEDICIÓN DEL TIEMPO REAL
print(f"Ejecutando stumpy.stump en toda la serie...")

start_time = time.time()
# --- INICIO CRONÓMETRO ---

mp_result = stumpy.stump(time_series_values, m=WINDOW_SIZE)

# --- FIN CRONÓMETRO ---
end_time = time.time()

duration = end_time - start_time
print(f"\nResultados del Benchmark:")
print(f"-------------------------")
print(f"Tiempo de ejecución: {duration:.4f} segundos")
print(f"-------------------------")

# Verificación rápida (sanity check)
mp_vector = mp_result[:, 0]
k1_idx = np.argsort(mp_vector)[-1]
print(f"Verificación: Pico más alto encontrado en índice {k1_idx}")