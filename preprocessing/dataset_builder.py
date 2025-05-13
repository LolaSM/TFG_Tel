import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def cargar_paciente_npz(path_npz):
    data = np.load(path_npz, allow_pickle=True)
    return data['X'], data['y'], data['identificador']

#Genera las secuencias de longitud L y sus etiquetas correspondientes: convierte una lista de ventanas independientes (cada una de 30s) en secuencias de L ventanas consecutivas, para que puedan ser interpretadas como una serie temporal.
#X: array de datos, y: etiquetas, L: longitud de la secuencia
def construir_secuencias(X, y, L):
    secuencias = []
    etiquetas = []
    for i in range(L, len(X)):
        secuencia = X[i - L:i]  # L ventanas
        etiqueta = y[i]         # etiqueta del último paso
        secuencias.append(secuencia)
        etiquetas.append(etiqueta)
    return np.array(secuencias, dtype=np.float32), np.array(etiquetas)
#mucha memoria RAM empleada, ver si puedo limpiar las variables 
#probar a ejecutar con una sola señal o menos pacientes para ver si funciona

def cargar_datos_secuenciales(path, L=5):
    pacientes = sorted(f for f in os.listdir(path) if f.endswith('.npz'))
    np.random.seed(42)
    np.random.shuffle(pacientes)

    n = len(pacientes)
    # Dividir en 60% entrenamiento, 20% validación y 20% test
    # División por paciente, no aleatoria, por eso no se usa train_test_split
    train_ids = pacientes[:int(0.6 * n)]
    val_ids = pacientes[int(0.6 * n):int(0.8 * n)]
    test_ids = pacientes[int(0.8 * n):]
    print(f"Pacientes: {len(pacientes)}, Entrenamiento: {len(train_ids)}, Validación: {len(val_ids)}, Test: {len(test_ids)}")
    #151 pacientes, 90 de entrenamiento, 30 de validación y 31 de test

    def cargar_lista(ids):
        X_total, y_total = [], []
        for pid in ids:
            X, y, _ = cargar_paciente_npz(os.path.join(path, pid))
            if len(X) <= L:
                continue
            X_seq, y_seq = construir_secuencias(X, y, L)
            X_total.append(X_seq)
            y_total.append(y_seq)
        #Une todos los pacientes en arrays únicos
        return np.concatenate(X_total), np.concatenate(y_total)

    X_train, y_train = cargar_lista(train_ids)
    X_val, y_val = cargar_lista(val_ids)
    X_test, y_test = cargar_lista(test_ids)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

