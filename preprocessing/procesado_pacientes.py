import os
import mne
import numpy as np
from tensorflow.keras.utils import to_categorical

# -------------------------------------------
# MAPEADO GLOBAL FIJO DE ETIQUETAS (0..4)
# -------------------------------------------
# Asumimos que las anotaciones vienen en forma de 'Sleep stage X',
# donde X ∈ {W, N1, N2, N3, R}.
_GLOBAL_LABEL_MAPPING = {
    'W':  0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'R':  4
}


def cargar_datos(nombre_archivo, path):
    """
    Lee el EDF de señales (nombre_archivo.edf) y su correspondiente
    anotaciones (nombre_archivo_sleepscoring.edf) en `path`.
    Devuelve un objeto Raw de MNE con las anotaciones cargadas.
    """
    archivo_senal   = os.path.join(path, f"{nombre_archivo}.edf")
    archivo_scoring = os.path.join(path, f"{nombre_archivo}_sleepscoring.edf")
    raw = mne.io.read_raw_edf(archivo_senal, preload=True, verbose=False)
    anotaciones = mne.read_annotations(archivo_scoring)
    raw.set_annotations(anotaciones)
    return raw


def filtrar_resamplear(raw, fmin=0.5, fmax=40.0, sfreq_nuevo=100):
    """
    Aplica un filtro pasa-banda [fmin, fmax] y luego resamplea a sfreq_nuevo Hz.
    """
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
    raw.resample(sfreq_nuevo, verbose=False)
    return raw


def procesar_paciente(nombre_archivo, path, fs=100, dur_epoch=30, canales=None):
    """
    - Carga EDF + anotaciones de un paciente.
    - Filtra y resamplea la señal.
    - Extrae ventanas de duración `dur_epoch` segundos.
    - Asigna etiquetas fijas (0..4) según el diccionario global.
    - Devuelve:
        X: np.array float32 con forma (n_ventanas, dur_epoch*fs, n_canales)
        y: np.array one-hot float32 con forma (n_ventanas, 5)
        fs: frecuencia de muestreo (en Hz)
    """
    # 1) Cargar y filtrar + resamplear
    raw = cargar_datos(nombre_archivo, path)
    raw = filtrar_resamplear(raw, fmin=0.5, fmax=40.0, sfreq_nuevo=fs)

    # 2) Extraer solo anotaciones que empiecen por "Sleep stage "
    sleep_annotations = []
    for a in raw.annotations:
        desc = a['description'].strip()
        if desc.startswith('Sleep stage '):
            label_token = desc.split()[-1]  # e.g. 'W', 'N2', 'R'
            if label_token in _GLOBAL_LABEL_MAPPING:
                sleep_annotations.append((a['onset'], a['duration'], label_token))

    # Si no hay anotaciones tipo "Sleep stage", terminamos
    if len(sleep_annotations) == 0:
        print(f"{nombre_archivo}: No se encontraron anotaciones tipo 'Sleep stage X'.")
        return None, None, None

    # 3) Seleccionar canales (si no se indica ninguno, tomamos todos)
    if canales is None:
        canales = raw.info['ch_names']
    # Usamos raw.copy().pick(lista_de_canales) (sin `ch_names=`)
    raw_sel = raw.copy().pick_channels(canales)
    datos = raw_sel.get_data()  # shape: (n_canales, n_muestras)
    n_canales, n_muestras = datos.shape
    print(f"{nombre_archivo}: {n_canales} canales, {n_muestras} muestras")

    # 4) Construir ventanas y etiquetas
    X_ventanas = []
    y_indices  = []
    muestras_por_ventana = int(dur_epoch * fs)  # ej. 30 s * 100 Hz = 3000 muestras

    for onset_sec, dur_sec, label_token in sleep_annotations:
        start = int(onset_sec * fs)
        end   = start + muestras_por_ventana
        # Saltamos si excede la longitud de la grabación
        if end > n_muestras:
            continue

        ventana = datos[:, start:end]  # (n_canales, muestras_por_ventana)
        # Si no tiene exactamente el ancho esperado, saltamos
        if ventana.shape[1] != muestras_por_ventana:
            continue

        # Transponemos a (muestras_por_ventana, n_canales)
        ventana = np.transpose(ventana, (1, 0))  # → (3000, n_canales)

        # Convertir etiqueta a índice fijo
        label_idx = _GLOBAL_LABEL_MAPPING[label_token]

        X_ventanas.append(ventana.astype(np.float32))
        y_indices.append(label_idx)

    # Si al final no creamos ventanas válidas, terminamos
    if len(X_ventanas) == 0:
        print(f"{nombre_archivo}: No se generaron ventanas válidas.")
        return None, None, None

    # 5) Apilar en un array 3D y convertir etiquetas a one-hot
    X = np.stack(X_ventanas, axis=0)            # (n_ventanas, 3000, n_canales)
    y_int = np.array(y_indices, dtype=np.int32)  # (n_ventanas,)
    y_onehot = to_categorical(y_int, num_classes=5).astype(np.float32)  # (n_ventanas, 5)

    print(f"{nombre_archivo}: {len(X)} ventanas generadas.")
    return X, y_onehot, fs


def guardar_paciente(identificador, X, y, fs, path_destino):
    """
    Guarda en path_destino/identificador.npz un diccionario con:
      - 'identificador': string
      - 'X': np.array float32 (n_ventanas, tiempo, canales)
      - 'y': np.array float32 one-hot (n_ventanas, 5)
      - 'fs': int (frecuencia de muestreo)
    """
    os.makedirs(path_destino, exist_ok=True)
    paciente = {
        "identificador": identificador,
        "X": X,
        "y": y,
        "fs": fs
    }
    np.savez(os.path.join(path_destino, f"{identificador}.npz"), **paciente)


def procesamiento_completo(path_origen, path_destino, canales=None):
    """
    Recorre todos los archivos *_sleepscoring.edf en path_origen,
    procesa cada paciente con procesar_paciente() y guarda un .npz en path_destino.
    """
    archivos = [f for f in os.listdir(path_origen) if f.endswith('_sleepscoring.edf')]
    pacientes = sorted({f.split('_')[0] for f in archivos})

    for paciente in pacientes:
        print(f"\nProcesando paciente: {paciente}")
        X, y, fs = procesar_paciente(paciente, path_origen, fs=100, dur_epoch=30, canales=canales)
        if X is None or y is None:
            print(f"{paciente}: No se procesó correctamente. Se omite.")
            continue
        guardar_paciente(paciente, X, y, fs, path_destino)
