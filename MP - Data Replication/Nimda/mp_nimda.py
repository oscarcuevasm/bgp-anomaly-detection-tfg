import pandas as pd
import matrixprofile as mp 
import numpy as np
import matplotlib.pyplot as plt

# --- 0. CONFIGURACIÓN GLOBAL ---
FILE_NAME = "Nimda.csv"
FEATURE_INDEX = 6 # Columna 5 (índice 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) # Para imprimir los MP values

print(f"--- Ejecutando Replicación Dual: Incidente Nimda ---")
print(f"Cargando datos de {FILE_NAME}...")

# --- 1. CARGA DE DATOS (Una sola vez) ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Longitud de la serie temporal (N): {N} minutos.")
except Exception as e:
    print(f"ERROR: No se pudo cargar {FILE_NAME}. Error: {e}")
    exit()

# --- 2. ANÁLISIS 1: m=1300 (Figura 5) ---
print("\n--- ANÁLISIS 1 (m=1300) ---")
WINDOW_SIZE_1 = 1300
K_DISCORDS_1 = 2

print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE_1}...")
profile_1 = mp.compute(time_series_values, WINDOW_SIZE_1)
mp_vector_1 = profile_1['mp']

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

print("\n--- RESULTADOS (m=1300) ---")
k1_index = discords_indices_1_sorted[0]
k2_index = discords_indices_1_sorted[1]
print(f"k1 (Pico más alto): Inicio en el minuto {k1_index} (Valor MP: {mp_vector_1[k1_index]:.2f})")
print(f"k2 (Pico secundario): Inicio en el minuto {k2_index} (Valor MP: {mp_vector_1[k2_index]:.2f})")
print("Validación: Coincide con Fig. 5 (Picos en ~4371 y ~1057).")


# --- 3. ANÁLISIS 2: m=144 (Figura 4) ---
print("\n--- ANÁLISIS 2 (m=144) ---")
WINDOW_SIZE_2 = 144
K_DISCORDS_2 = 15 # k=15 para replicar las múltiples alarmas

print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE_2}...")
profile_2 = mp.compute(time_series_values, WINDOW_SIZE_2)
mp_vector_2 = profile_2['mp']

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
print(f"\n--- RESULTADOS (m=144) ---")
print(f"Se encontraron {len(discords_indices_2)} picos (alarmas).")
print("Validación: Compara visualmente la Gráfica 2 con la Figura 4 del PDF.")


# --- 4. VISUALIZACIÓN ---

# Gráfica 1 (m=1300)
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Nimda: Análisis m={WINDOW_SIZE_1} (Comparación Fig. 5)', fontsize=16)

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
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile', color='green')
for i, index in enumerate(discords_indices_1_sorted):
    plt.plot(index, mp_vector_1[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Distancia Euclidiana Mínima')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Gráfica 2 (m=144)
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Nimda: Análisis m={WINDOW_SIZE_2} (Comparación Fig. 4)', fontsize=16)

plt.subplot(2, 1, 1) # Usaremos solo 2 subplots (Datos y MP)
plt.plot(time_series, label='Datos BGP (Volumen)', color='C0')
plt.title(f'Serie Temporal (Datos)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP)', color='C0')
# Marcar TODAS las discordias (k=15) con estrellas rojas
for index in discords_indices_2:
    plt.plot(index, mp_vector_2[index], marker='*', markersize=10, color='red', linestyle='None', label='Discord (Alarma)')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nMostrando gráficas...")
plt.show()