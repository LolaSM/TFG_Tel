from ecg_filtrado import cargar_datos, filtrar_ecg
import matplotlib.pyplot as plt
import numpy as np

nombre = "SN001"
path = "recordings"

raw = cargar_datos_ecg(nombre, path)
raw = filtrar_ecg(raw)

# Visualiza con anotaciones superpuestas
raw.plot(start=0, duration=120, scalings='auto', title=f"Paciente {nombre}", show=True)

# Visualizar anotaciones explícitamente sobre el tiempo
annotations = raw.annotations
fs = raw.info['sfreq']
ecgs = raw.copy().pick(['ECG']).get_data()[0]
times = np.arange(ecgs.shape[0]) / fs

plt.figure(figsize=(15, 4))
plt.plot(times, ecgs, label='ECG')

for annot in annotations:
    onset = annot['onset']
    dur = annot['duration']
    label = annot['description']
    plt.axvspan(onset, onset + dur, color='orange', alpha=0.3)
    plt.text(onset, max(ecgs) * 0.8, label, rotation=90, fontsize=6, color='red')

plt.xlim([0, 120])
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title(f"ECG + Anotaciones para {nombre}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
