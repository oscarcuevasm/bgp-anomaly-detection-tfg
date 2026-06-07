import pandas as pd
import matrixprofile as mp
import numpy as np
import time

# --- CONFIGURACIÓN ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868
FEATURE_INDEX = 4

print(f"--- BENCHMARK MATRIXPROFILE (Librería): Slammer (m={WINDOW_SIZE}) ---")

# 1. CARGA DE DATOS
print(f"Cargando datos...")
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float)
    time_series_values = time_series.values
except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()

# 2. MEDICIÓN DEL TIEMPO REAL
# La librería matrixprofile no tiene compilación JIT (warm-up), 
# así que medimos directamente.
print(f"Ejecutando mp.compute en toda la serie...")

start_time = time.time()
# --- INICIO CRONÓMETRO ---

# Usamos la sintaxis que validamos: argumento posicional
profile = mp.compute(time_series_values, WINDOW_SIZE)

# --- FIN CRONÓMETRO ---
end_time = time.time()

duration = end_time - start_time

print(f"\nResultados del Benchmark (MatrixProfile Lib):")
print(f"-------------------------------------------")
print(f"Tiempo de ejecución: {duration:.4f} segundos")
print(f"-------------------------------------------")

# Verificación rápida
mp_vector = profile['mp']
k1_idx = np.argmax(mp_vector)
print(f"Verificación: Pico más alto encontrado en índice {k1_idx}")