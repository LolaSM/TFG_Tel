import os
import mne
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def cargar_datos(nombre_archivo, path):
    archivo_senal = os.path.join(path, f"{nombre_archivo}.edf")
    archivo_scoring = os.path.join(path, f"{nombre_archivo}_sleepscoring.edf")

    raw = mne.io.read_raw_edf(archivo_senal, preload=True, verbose=False)
    if nombre_archivo == "SN001":
        print(f"Canales disponibles en {nombre_archivo}: {raw.info['ch_names']}")
    anotaciones = mne.read_annotations(archivo_scoring)
    raw.set_annotations(anotaciones)
    return raw

def filtrar_resamplear(raw):
    #Filtro paso banda que elimina componentes de muy baja frecuencia <0.5 Hz, elimina ruido de ata frecuencia >40 Hz
    raw.filter(0.5, 40., fir_design='firwin')
    #Resampleo a 100 Hz para reducir el tamaño de los datos y acelerar el procesamiento
    raw.resample(100)
    return raw

#fs=100 tras el resampleo
#dur_epoch=30s, cada ventana de datos tiene 3000 muestras (100Hz * 30s)
def procesar_paciente(nombre_archivo, path, fs=100, dur_epoch=30, canales=None):
    raw = cargar_datos(nombre_archivo, path)
    raw = filtrar_resamplear(raw)

    annotations = raw.annotations
    
    #Extrae las anotaciones de sueño y sus etiquetas
    sleep_annotations = [
        (a['onset'], a['duration'], a['description'].strip().split()[-1])
        for a in annotations
        if 'Sleep stage' in a['description']
    ]
    
    if canales is None:
        # Si no se especifican canales, se seleccionan todos los disponibles
        canales = raw.info['ch_names']
    raw_seleccionado = raw.copy().pick_channels(canales)
    datos = raw_seleccionado.get_data()  # shape (n_canales, n_muestras)
    print(f"{nombre_archivo}: {datos.shape[0]} canales, {datos.shape[1]} muestras")
    
    X, y = [], []
    
    for onset, dur, label in sleep_annotations:
        start = int(onset * fs)
        end = start + int(dur_epoch * fs)
        if end > datos.shape[1]:
            continue
        #ventana.shape  # → (n_canales, muestras)
        ventana = datos[:, start:end]
        if ventana.shape[1] == dur_epoch * fs:
            X.append(ventana)
            y.append(label)

    if not X or not y:
        print(f"{nombre_archivo}: No se encontraron ventanas válidas con etiquetas.")
        return None, None, None
    
    #Convierte X en un array 3D (necesario para el modelo) y y en one-hot encoding para clasificación multiclase
    #Conv1D espera entradas de la forma (batch_size-->n_ventanas, time_steps-->muestras, channels-->n_canales)
    #convertir a float32 para evitar problemas de memoria
    X = np.array(X, dtype=np.float32)  # shape (n_ventanas, n_canales, muestras)
    X = np.transpose(X, (0, 2, 1))  # shape (n_ventanas, muestras, n_canales)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    #Me aseguro de que haya 5 clases: W, N1, N2, N3, R (si no hay alguna clase para algún paciente, se asigna 0)
    y_onehot = to_categorical(y_encoded, num_classes=5)

    print(f"{nombre_archivo}: {len(X)} ventanas generadas")

    return X, y_onehot, fs

def guardar_paciente(identificador, X, y, fs, path_destino):
    os.makedirs(path_destino, exist_ok=True)
    #Empaqueta todos los datos en un diccionario y lo guarda
    paciente = {
        "identificador": identificador,
        "X": X,
        "y": y,
        "fs": fs
    }
    np.savez(os.path.join(path_destino, f"{identificador}.npz"), **paciente)
    
def procesamiento_completo(path_origen, path_destino, canales=None):
    pacientes = sorted(set(f.split('_')[0] for f in os.listdir(path_origen) if f.endswith('_sleepscoring.edf')))
    for paciente in pacientes:
        print(f"Procesando paciente: {paciente}")
        X, y, fs = procesar_paciente(paciente, path_origen,canales=canales)
        if X is None or y is None:
            print(f"Error al procesar el paciente {paciente}. Saltando...")
            continue
        if X is not None:
            guardar_paciente(paciente, X, y, fs, path_destino)