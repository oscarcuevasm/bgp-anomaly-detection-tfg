import pandas as pd
import matrixprofile as mp 
import numpy as np
import matplotlib.pyplot as plt

# --- 0. CONFIGURACIÓN GLOBAL ---
FILE_NAME = "Code_Red_I.csv" # Asegúrate de que este sea el nombre de tu archivo
FEATURE_INDEX = 4 # Columna 5 (índice 4) = "Number of announcements"
np.set_printoptions(suppress=True, precision=2) # Para imprimir los MP values

print(f"--- Ejecutando Replicación Dual: Incidente Code Red I ---")
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

# --- 2. ANÁLISIS 1: m=599 (Figura 1) ---
print("\n--- ANÁLISIS 1 (m=599) ---")
WINDOW_SIZE_1 = 599
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

print("\n--- RESULTADOS (m=599) ---")
k1_index_m1 = discords_indices_1_sorted[0]
k2_index_m1 = discords_indices_1_sorted[1]
print(f"k1 (Pico más alto): Inicio en el minuto {k1_index_m1} (Valor MP: {mp_vector_1[k1_index_m1]:.2f})")
print(f"k2 (Pico secundario): Inicio en el minuto {k2_index_m1} (Valor MP: {mp_vector_1[k2_index_m1]:.2f})")
print("Validación: El artículo (Fig. 1) muestra picos en ~3320 y ~3800 min.")


# --- 3. ANÁLISIS 2: m=1440 (Figura 2) ---
print("\n--- ANÁLISIS 2 (m=1440) ---")
WINDOW_SIZE_2 = 1440
K_DISCORDS_2 = 2 # Buscamos 2 por si acaso, aunque la Fig 2 solo muestra 1

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

# Ordenar por magnitud de MP
mp_values_2 = mp_vector_2[discords_indices_2]
sort_by_mp_2 = np.argsort(mp_values_2)[::-1]
discords_indices_2_sorted = [discords_indices_2[i] for i in sort_by_mp_2]

print(f"\n--- RESULTADOS (m=1440) ---")
k1_index_m2 = discords_indices_2_sorted[0]
print(f"k1 (Pico más alto): Inicio en el minuto {k1_index_m2} (Valor MP: {mp_vector_2[k1_index_m2]:.2f})")
print("Validación: El artículo (Fig. 2) reporta la 1ra discordia en ~2900 min.")


# --- 4. VISUALIZACIÓN ---

# Gráfica 1 (m=599)
plt.figure(1, figsize=(18, 10))
plt.suptitle(f'Code Red I: Análisis m={WINDOW_SIZE_1} (Comparación Fig. 1)', fontsize=16)

plt.subplot(2, 1, 1) # Usaremos solo 2 subplots
plt.plot(time_series, label='Datos BGP (Volumen)', color='C0')
plt.title(f'Serie Temporal (Datos)')
plt.ylabel('BGP Volume')
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_1)), mp_vector_1, label='Matrix Profile (MP)', color='C0')
# Marcar k=2 discordias
for i, index in enumerate(discords_indices_1_sorted):
    plt.plot(index, mp_vector_1[index], marker='*', markersize=10, color='red', linestyle='None')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Matrix Profile')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Gráfica 2 (m=1440)
plt.figure(2, figsize=(18, 10))
plt.suptitle(f'Code Red I: Análisis m={WINDOW_SIZE_2} (Comparación Fig. 2)', fontsize=16)

plt.subplot(2, 1, 1) 
plt.plot(time_series, label='Datos BGP (Volumen)', color='C0')
# Marcar la subsecuencia de la 1ra discordia (k1)
start = discords_indices_2_sorted[0]
end = start + WINDOW_SIZE_2
plt.plot(np.arange(start, end), time_series.iloc[start:end], c='red', 
         label=f'1st Discord (Inicio: {start} min)')
plt.title(f'Serie Temporal (Datos) y 1ra Discordia')
plt.ylabel('BGP Volume')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector_2)), mp_vector_2, label='Matrix Profile (MP)', color='green')
# Marcar solo la k1
plt.plot(discords_indices_2_sorted[0], mp_vector_2[discords_indices_2_sorted[0]], 
         marker='^', markersize=10, color='red', markeredgecolor='black')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Distancia Euclidiana Mínima')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


print("\nMostrando gráficas...")
plt.show()