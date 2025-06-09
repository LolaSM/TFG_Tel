# check_labeling.py

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# Importa tus funciones desde procesado_pacientes.py
from preprocessing.procesado_pacientes import (
    cargar_datos,
    filtrar_resamplear,
    procesar_paciente,
    _GLOBAL_LABEL_MAPPING
)

# Inversa del mapeado (para etiquetas numéricas → nombre)
INV_LABEL_MAPPING = {v: k for k, v in _GLOBAL_LABEL_MAPPING.items()}

# Parámetros
RECORDINGS_PATH = "recordings"
PACIENTES = sorted(
    f.split("_")[0]
    for f in os.listdir(RECORDINGS_PATH)
    if f.endswith("_sleepscoring.edf")
)[:5]  # toma los primeros 5 pacientes
FS = 100
DUR_EPOCH = 30  # segundos

def plot_hypnogram(raw, ax, title):
    """
    Dibuja en `ax` un hipnograma crudo basado en raw.annotations:
    un step-plot de la etiqueta numérica frente al tiempo.
    """
    # Extraer solo anotaciones 'Sleep stage X'
    ann = [
        (a["onset"], a["duration"], a["description"].strip().split()[-1])
        for a in raw.annotations
        if a["description"].strip().startswith("Sleep stage ")
    ]
    # Convertir a timeline: cada anotación → etiqueta numérica
    times = []
    labels = []
    for onset, dur, token in ann:
        idx = _GLOBAL_LABEL_MAPPING.get(token, None)
        if idx is None:
            continue
        times.extend([onset, onset + dur])
        labels.extend([idx, idx])

    ax.step(times, labels, where="post", label="Crudo")
    ax.set_ylim(-0.5, 4.5)
    ax.set_yticks(list(_GLOBAL_LABEL_MAPPING.values()))
    ax.set_yticklabels(list(_GLOBAL_LABEL_MAPPING.keys()))
    ax.set_xlabel("Tiempo (s)")
    ax.set_title(title)

def plot_windowed_hypnogram(X, y_int, ax):
    """
    Dibuja encima de `ax` las etiquetas windowed como puntos
    en el centro de cada ventana de 30 s.
    """
    n_windows = X.shape[0]
    # centro de cada ventana en segundos
    centers = np.arange(n_windows) * DUR_EPOCH + DUR_EPOCH / 2
    ax.scatter(centers, y_int, color="red", s=20, label="Windowed")
    ax.legend(loc="upper right")

def main():
    plt.figure(figsize=(12, 3 * len(PACIENTES)))

    for i, paciente in enumerate(PACIENTES, start=1):
        # 1) Carga cruda y filtra+resamplea
        raw = cargar_datos(paciente, RECORDINGS_PATH)
        raw = filtrar_resamplear(raw, fmin=0.5, fmax=40.0, sfreq_nuevo=FS)

        # 2) Procesa ventanas + etiquetas
        X, y_onehot, fs = procesar_paciente(
            paciente,
            RECORDINGS_PATH,
            fs=FS,
            dur_epoch=DUR_EPOCH,
            canales=None,
        )
        if X is None:
            print(f"{paciente}: no se encontró ningún segmento, se salta.")
            continue

        # Convertir y_onehot a índice entero
        y_int = np.argmax(y_onehot, axis=1)

        # 3) Dibujo
        ax = plt.subplot(len(PACIENTES), 1, i)
        plot_hypnogram(raw, ax, title=f"Paciente {paciente}")
        plot_windowed_hypnogram(X, y_int, ax)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
