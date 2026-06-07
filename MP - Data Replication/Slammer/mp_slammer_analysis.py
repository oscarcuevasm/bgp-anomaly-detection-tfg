import pandas as pd
import matrixprofile as mp 
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DEL ANÁLISIS (SLAMMER) ---
FILE_NAME = "Slammer.csv"
WINDOW_SIZE = 868  # m1 = 868 minutos (Período anómalo de referencia)
K_DISCORDS = 2     # k = 2
FEATURE_INDEX = 4  # Columna 5 (índice 4) = "Number of announcements"

print(f"--- Ejecutando Re-Análisis: Incidente Slammer (Método Validado) ---")
print(f"Iniciando análisis de {FILE_NAME} con m={WINDOW_SIZE} usando matrixprofile...")

# --- 2. CARGA Y SELECCIÓN DE LA SERIE TEMPORAL ---
try:
    df = pd.read_csv(FILE_NAME, header=None)
    time_series = df.iloc[:, FEATURE_INDEX].astype(float) 
    time_series_values = time_series.values
    N = len(time_series)
    print(f"Longitud de la serie temporal (N): {N} minutos.")
except Exception as e:
    print(f"ERROR: No se pudo cargar {FILE_NAME}. Error: {e}")
    exit()

# --- 3. CÁLCULO DEL MATRIX PROFILE (MP) ---
print(f"Calculando Matrix Profile (MP) con m = {WINDOW_SIZE}...")
profile = mp.compute(time_series_values, WINDOW_SIZE)
mp_vector = profile['mp']

# --- 4. DETECCIÓN DE DISCORDIAS (MÉTODO MANUAL CORRECTO) ---
print(f"Descubriendo las top-{K_DISCORDS} discordias (con Zona de Exclusión Manual)...")

mp_vector_copy = np.copy(mp_vector)
discords_indices = []

for k in range(K_DISCORDS):
    # 1. Encontrar el índice del pico más alto actual
    k_index = np.argmax(mp_vector_copy)
    discords_indices.append(k_index)
    
    # 2. Definir la Zona de Exclusión
    start_exclusion = max(0, k_index - WINDOW_SIZE)
    end_exclusion = min(len(mp_vector_copy), k_index + WINDOW_SIZE)
    
    # 3. "Aplanar" (poner a -inf) la zona de exclusión
    mp_vector_copy[start_exclusion:end_exclusion] = np.NINF 

# Ordenar por magnitud de MP
mp_values = mp_vector[discords_indices]
sort_by_mp = np.argsort(mp_values)[::-1]
discords_indices_sorted = [discords_indices[i] for i in sort_by_mp]

print("\n--- RESULTADOS OBTENIDOS (Slammer) ---")
k1_index = discords_indices_sorted[0]
k2_index = discords_indices_sorted[1]
print(f"k1 (Pico más alto): Inicio en el minuto {k1_index} (Valor MP: {mp_vector[k1_index]:.2f})")
print(f"k2 (Pico secundario): Inicio en el minuto {k2_index} (Valor MP: {mp_vector[k2_index]:.2f})")

# Validar el resultado clave reportado en el artículo (página 6)
# Período anómalo conocido por RNNs: 3200-4068 min
print(f"\nVALIDACIÓN CLAVE (k2): El índice obtenido es {k2_index}.")
print("El artículo reportó una alarma temprana (k2) en 3030 min (antes de 3200 min).")
    

# --- 5. VISUALIZACIÓN (Comparación con Figura 6) ---
plt.figure(figsize=(18, 10))
plt.suptitle(f'Slammer: Análisis m={WINDOW_SIZE} (Comparación Fig. 6)', fontsize=16)

plt.subplot(2, 1, 1)
plt.plot(time_series, label='BGP Volume (Announcements)', color='blue', alpha=0.6)
colors = ['red', 'purple']
labels = ['1st Discord', '2nd Discord']

for i, index in enumerate(discords_indices_sorted):
    start = index
    end = start + WINDOW_SIZE
    plt.plot(np.arange(start, end), time_series.iloc[start:end], c=colors[i], 
             label=f'{labels[i]} (Inicio: {index} min)')
    plt.plot(start, time_series.iloc[start], marker='*', markersize=10, color=colors[i], markeredgecolor='black')

# Marcar el período anómalo conocido (3200 a 4068 min)
plt.axvspan(3200, 4068, color='gray', alpha=0.2, label='Período Anómalo (RNNs)')
plt.title(f'Serie Temporal BGP')
plt.ylabel('BGP Volume (Announcements)')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(mp_vector)), mp_vector, label='Matrix Profile', color='green')
for i, index in enumerate(discords_indices_sorted):
    plt.plot(index, mp_vector[index], marker='^', markersize=10, color=colors[i], markeredgecolor='black')
plt.title(f'Vector Matrix Profile (MP)')
plt.xlabel(f'Índice de Tiempo (minutos)')
plt.ylabel('Distancia Euclidiana Mínima')
plt.grid(True, linestyle='--')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("\nMostrando gráficas...")
plt.show()