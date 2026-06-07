import pandas as pd
import stumpy # Usaremos STUMPY para el cálculo
import numpy as np
import matplotlib.pyplot as plt

# --- 0. CONFIGURACIÓN GLOBAL ---
FILE_NAME = "Nimda.csv"
FEATURE_INDEX = 4  # Columna 5 (índice 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) # Para imprimir los MP values

print(f"--- Ejecutando Replicación Dual: Incidente Nimda (Usando STUMPY) ---")
print(f"Cargando datos de {FILE_NAME}...")

# --- 1. CARGA DE DATOS (Una sola vez) ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    # STUMPY prefiere valores numpy, no es estrictamente necesario pero es buena práctica
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Longitud de la serie temporal (N): {N} minutos.")
except Exception as e:
    print(f"ERROR: No se pudo cargar {FILE_NAME}. Error: {e}")
    exit()

# --- 2. ANÁLISIS 1: m=1300 (Figura 5) ---
print("\n--- ANÁLISIS 1 (m=1300) [STUMPY] ---")
WINDOW_SIZE_1 = 1300
K_DISCORDS_1 = 2

print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE_1}...")
# CÁLCULO CON STUMPY
# stumpy.stump devuelve una matriz. La columna 0 es el Vector MP.
mp_result_1 = stumpy.stump(time_series_values, m=WINDOW_SIZE_1)
mp_vector_1 = mp_result_1[:, 0] 

print(f"Descubriendo las top-{K_DISCORDS_1} discordias (con Zona de Exclusión Manual)...")
mp_vector_copy_1 = np.copy(mp_vector_1)
discords_indices_1 = []

for k in range(K_DISCORDS_1):
    k_index = np.argmax(mp_vector_copy_1)
    discords_indices_1.append(k_index)
    start_exclusion = max(0, k_index - WINDOW_SIZE_1)
    end_exclusion = min(len(mp_vector_copy_1), k_index + WINDOW_SIZE_1)
    mp_vector_copy_1[start_exclusion:end_exclusion] = np.NINF 

# Ordenar por magnitud de MP
mp_values_1 = mp_vector_1[discords_indices_1]
sort_by_mp_1 = np.argsort(mp_values_1)[::-1]
discords_indices_1_sorted = [discords_indices_1[i] for i in sort_by_mp_1]

print("\n--- RESULTADOS (m=1300) [STUMPY] ---")
k1_index = discords_indices_1_sorted[0]
k2_index = discords_indices_1_sorted[1]
print(f"k1 (Pico más alto): Inicio en el minuto {k1_index} (Valor MP: {mp_vector_1[k1_index]:.2f})")
print(f"k2 (Pico secundario): Inicio en el minuto {k2_index} (Valor MP: {mp_vector_1[k2_index]:.2f})")
print("Validación: Comparar con matrixprofile (Picos en ~4371 y ~1057).")


# --- 3. ANÁLISIS 2: m=144 (Figura 4) ---
print("\n--- ANÁLISIS 2 (m=144) [STUMPY] ---")
WINDOW_SIZE_2 = 144
K_DISCORDS_2 = 15 # k=15 para replicar las múltiples alarmas

print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE_2}...")
# CÁLCULO CON STUMPY
mp_result_2 = stumpy.stump(time_series_values, m=WINDOW_SIZE_2)
mp_vector_2 = mp_result_2[:, 0]

print(f"Descubriendo las top-{K_DISCORDS_2} discordias (con Zona de Exclusión Manual)...")
mp_vector_copy_2 = np.copy(mp_vector_2)
discords_indices_2 = []

for k in range(K_DISCORDS_2):
    k_index = np.argmax(mp_vector_copy_2)
    discords_indices_2.append(k_index)
    start_exclusion = max(0, k_index - WINDOW_SIZE_2)
    end_exclusion = min(len(mp_vector_copy_2), k_index + WINDOW_SIZE_2)
    mp_vector_copy_2[start_exclusion:end_exclusion] = np.NINF 

discords_indices_2.sort()
print(f"\n--- RESULTADOS (m=144) [STUMPY] ---")
print(f"Se encontraron {len(discords_indices_2)} picos (alarmas).")
print("Validación: Compara visualmente la Gráfica 2 con la Figura 4 del PDF.")


# --- 4. VISUALIZACIÓN ---
# (El código de visualización es idéntico al anterior)

# Gráfica 1 (m=1300)
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Nimda: Análisis m={WINDOW_SIZE_1} (Comparación Fig. 5) [STUMPY]', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Volume (Announcements)', color='blue', alpha=0.6)
colors = ['red', 'purple']
for i, index in enumerate(discords_indices_1_sorted):
    start = index
    end = start + WINDOW_SIZE_1
    plt.plot(np.arange(start, end), time_series.iloc[start:end], c=colors[i], 
             label=f'k{i+1} Discord (Inicio: {index} min)')
plt.title(f'Serie Temporal BGP')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile (STUMPY)', color='green')
for i, index in enumerate(discords_indices_1_sorted):
    plt.plot(index, mp_vector_1[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Distancia Euclidiana Mínima')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Gráfica 2 (m=144)
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Nimda: Análisis m={WINDOW_SIZE_2} (Comparación Fig. 4) [STUMPY]', fontsize=16)

plt.subplot(2, 1, 1) 
plt.plot(time_series, label='Datos BGP (Volumen)', color='C0')
plt.title(f'Serie Temporal (Datos)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP) [STUMPY]', color='C0')
for index in discords_indices_2:
    plt.plot(index, mp_vector_2[index], marker='*', markersize=10, color='red', linestyle='None', label='Discord (Alarma)')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nMostrando gráficas...")
plt.show()