import os
import numpy as np
import mne
import matplotlib.pyplot as plt


def visualizar_ecg(raw, segundos=30):
    raw_ecg = raw.copy().pick(['ECG'])
    datos, tiempos = raw_ecg[:]
    fs = raw_ecg.info['sfreq']
    muestras = int(segundos * fs)

    plt.figure(figsize=(15, 4))
    plt.plot(tiempos[:muestras], datos[0][:muestras])
    plt.title("Señal ECG - primeros 30s")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

def visualizar_ecg_segmento(raw, inicio_segundos=0, duracion_segundos=30, titulo=""):
    raw_ecg = raw.copy().pick(['ECG'])
    fs = raw_ecg.info['sfreq']
    i0 = int(inicio_segundos * fs)
    i1 = int((inicio_segundos + duracion_segundos) * fs)
    datos, tiempos = raw_ecg[:, i0:i1]
    tiempo = np.linspace(0, duracion_segundos, datos.shape[1])

    plt.figure(figsize=(15, 4))
    plt.plot(tiempo, datos[0])
    plt.title(f"{titulo} (desde {inicio_segundos}s hasta {inicio_segundos+duracion_segundos}s)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()


def visualizar_ecg_con_etiquetas(X, y, fs, num_epochs=5):
    tiempo_epoch = X.shape[1] / fs
    for i in range(min(num_epochs, len(X))):
        plt.figure(figsize=(12, 2))
        tiempo = np.linspace(0, tiempo_epoch, X.shape[1])
        plt.plot(tiempo, X[i].squeeze())
        plt.title(f"Epoch {i+1} - Etiqueta: {np.argmax(y[i])}")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Amplitud")
        plt.grid(True)
        plt.show()
