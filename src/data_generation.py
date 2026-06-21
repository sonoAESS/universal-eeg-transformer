import os
import mne
import numpy as np
import pandas as pd
from mne.datasets import eegbci

# =====================================================================
# 1. CARGA DE DATOS Y COORDENADAS 3D (SISTEMA 10-20)
# =====================================================================
print("1. Cargando datos de EEG y configurando topografía 3D...")
subject = 1
runs = [1]
raw_fnames = eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
eegbci.standardize(raw)

# Configurar el montaje estándar para obtener las posiciones X, Y, Z de cada electrodo
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

nombres_canales = raw.ch_names
Y = raw.get_data()  # Matriz original (canales, tiempo)
num_canales, num_tiempos = Y.shape

# Extraer las posiciones 3D de los canales
posiciones_dict = montage.get_positions()['ch_pos']
pos_canales = np.array([posiciones_dict[ch] for ch in nombres_canales])

# =====================================================================
# 2. GENERACIÓN DEL LEAD FIELD (Modelo de Esfera Concéntrica)
# =====================================================================
print("2. Calculando la Matriz Lead Field (G) basada en la geometría...")
# Crear una simulación de fuentes dipolares en el interior del cerebro
# Generamos N fuentes dipolares distribuidas aleatoriamente dentro de la esfera cerebral
num_fuentes = 500  
np.random.seed(42)
# Radio aproximado de la cabeza en metros (8.5 cm)
radio_cabeza = 0.085 
pos_fuentes = np.random.randn(num_fuentes, 3)
# Normalizar para asegurar que las fuentes estén dentro del cerebro (p.ej. al 70% del radio)
pos_fuentes = (pos_fuentes / np.linalg.norm(pos_fuentes, axis=1, keepdims=True)) * (radio_cabeza * 0.7)

# Calcular la matriz Lead Field G de tamaño (num_canales x num_fuentes)
# Usamos una aproximación física de la ley de Ohm / Coulomb para medios conductores homogéneos
G = np.zeros((num_canales, num_fuentes))
for i, p_ch in enumerate(pos_canales):
    for j, p_src in enumerate(pos_fuentes):
        dist = np.linalg.norm(p_ch - p_src)
        G[i, j] = 1.0 / (4 * np.pi * dist) # Potencial monofásico equivalente

# =====================================================================
# 3. IMPLEMENTACIÓN MATEMÁTICA DE LAS 4 REFERENCIAS
# =====================================================================
print("3. Procesando montajes y estandarización REST...")

# A. Unipolar (Ref: Canal 10)
ref_signal = Y[10, :]
X_unipolar = Y - ref_signal

# B. Bipolar (Diferencial contiguo)
X_bipolar = np.zeros_like(Y)
for i in range(num_canales):
    siguiente = (i + 1) % num_canales
    X_bipolar[i, :] = Y[i, :] - Y[siguiente, :]

# C. Centering / Average Reference (CAR)
X_centering = Y - np.mean(Y, axis=0)

# D. Referencia al Infinito (REST Real usando Lead Field)
# Fórmula de la matriz de estandarización REST: R = G * pinv(Avg(G))
# Primero, aplicamos el operador de promedio (CAR) al Lead Field
W_car = np.eye(num_canales) - (1.0 / num_canales) * np.ones((num_canales, num_canales))
G_car = W_car @ G

# Calculamos la pseudoinversa de la matriz de ganancia promediada
G_car_pinv = np.linalg.pinv(G_car)

# Proyectamos la señal promediada (CAR) al espacio infinito usando el Lead Field original
X_infinito = G @ G_car_pinv @ X_centering

# =====================================================================
# 4. EXPORTACIÓN A ARCHIVOS CSV
# =====================================================================
print("4. Guardando matrices transpuestas en la carpeta 'data'...")
os.makedirs('data', exist_ok=True)

# Transponer a formato (puntos_de_tiempo, canales)
df_ground_truth = pd.DataFrame(Y.T, columns=nombres_canales)
df_unipolar     = pd.DataFrame(X_unipolar.T, columns=nombres_canales)
df_centering    = pd.DataFrame(X_centering.T, columns=nombres_canales)
df_infinito     = pd.DataFrame(X_infinito.T, columns=nombres_canales)

# Formatear nombres para las columnas bipolares (ej: FP1-F7)
nombres_bipolar = [f"{nombres_canales[i]}-{nombres_canales[(i+1)%num_canales]}" for i in range(num_canales)]
df_bipolar      = pd.DataFrame(X_bipolar.T, columns=nombres_bipolar)

# Guardar en disco
df_ground_truth.to_csv(os.path.join('data', 'eeg_ground_truth.csv'), index=False)
df_unipolar.to_csv(os.path.join('data', 'eeg_unipolar_ref10.csv'), index=False)
df_bipolar.to_csv(os.path.join('data', 'eeg_bipolar.csv'), index=False)
df_centering.to_csv(os.path.join('data', 'eeg_centering_car.csv'), index=False)
df_infinito.to_csv(os.path.join('data', 'eeg_infinito_leadfield.csv'), index=False)

print("\n¡Todo listo! Los datos calculados con el Lead Field geométrico se han generado correctamente.")