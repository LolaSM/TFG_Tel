import os
import numpy as np
import tensorflow as tf


def cargar_paciente_npz(path_npz):
    """
    Carga un archivo .npz generado por procesado_pacientes.py
    Devuelve (X, y, identificador).
      - X: np.array float32 con forma (n_ventanas, time_steps, n_canales)
      - y: np.array float32 one-hot con forma (n_ventanas, 5)
      - identificador: string
    """
    data = np.load(path_npz, allow_pickle=True)
    return data['X'], data['y'], data['identificador'].item()

def dividir_pacientes(path_npz_folder, semilla=42):
    """
    Lee todos los archivos .npz en path_npz_folder, los baraja con semilla fija
    y retorna tres listas de rutas: train_ids, val_ids y test_ids.

    “División por paciente pseudoaleatoria” significa:
      1) Toma la lista completa de pacientes (cada uno es un .npz).
      2) La baraja con np.random.shuffle (con semilla) → reparto aleatorio.
      3) Parte en tres grupos 60% / 20% / 20% por lista de pacientes.
    De esta forma, garantizamos que **todas las ventanas de un mismo paciente
    estén en un solo split**, pues trabajamos a nivel de archivos .npz, no a nivel
    de ventanas individuales.
    """
    pacientes = sorted([os.path.join(path_npz_folder, f)
                        for f in os.listdir(path_npz_folder) if f.endswith('.npz')])
    np.random.seed(semilla)
    np.random.shuffle(pacientes)

    n = len(pacientes)
    n_train = int(0.6 * n)
    n_val   = int(0.2 * n)

    train_ids = pacientes[:n_train]
    val_ids   = pacientes[n_train:n_train + n_val]
    test_ids  = pacientes[n_train + n_val:]
    return train_ids, val_ids, test_ids

def generator_de_secuencias(lista_npz, L=5):
    """
    Generador que recorre la lista de archivos .npz de pacientes, carga
    cada paciente, extrae sus ventanas de tamaño (time_steps, n_canales)
    y ensambla secuencias de longitud L.

    - Para cada paciente:
        X_pac: (n_ventanas_pac, time_steps, n_canales)
        y_pac: (n_ventanas_pac, 5)
      Construye subsecuencias:
        para i in [L .. n_ventanas_pac-1]:
          secuencia := X_pac[i-L : i]         → shape: (L, time_steps, n_canales)
          etiqueta  := y_pac[i]               → shape: (5,)
      Y hace `yield (secuencia, etiqueta)` en float32.

    Así, **nunca concatenamos** todos los pacientes en memoria; vamos paciente a paciente.
    """
    for path_npz in lista_npz:
        X_pac, y_pac, _ = cargar_paciente_npz(path_npz)
        n_ventanas_pac = X_pac.shape[0]
        if n_ventanas_pac <= L:
            continue

        # Por cada posible secuencia de largo L en este paciente:
        for i in range(L, n_ventanas_pac):
            secuencia = X_pac[i - L:i]   # shape: (L, time_steps, n_canales)
            etiqueta  = y_pac[i]         # shape: (5,)
            yield secuencia.astype(np.float32), etiqueta.astype(np.float32)


def crear_tf_dataset(path_npz_folder, L=5, batch_size=32, shuffle_buffer=1000):
    """
    Crea objetos tf.data.Dataset para train, val y test, usando generator_de_secuencias:

    Returns: train_ds, val_ds, test_ds
       - Cada uno listo para pasarse a model.fit().
       - Internamente, se usa from_generator para leer dinámicamente cada secuencia.
       - Se aplica shuffle solo en el split de train (con un buffer de tamaño shuffle_buffer).
       - Se repite la iteración indefinidamente (epoch infinito), y se hace batch.
         Keras detendrá la época cuando haya consumido el número de pasos' adecuado
         (ver `steps_per_epoch` en el script de train).

    Nota: para métricas de evaluación posteriores, en test_ds no aplicamos shuffle.
    """
    # 1) Obtener listas de rutas .npz divididas por paciente
    train_ids, val_ids, test_ids = dividir_pacientes(path_npz_folder)

    # 2) Definir una función “callable” para que tf.data pueda usarla
    def gen_train():
        yield from generator_de_secuencias(train_ids, L=L)

    def gen_val():
        yield from generator_de_secuencias(val_ids, L=L)

    def gen_test():
        yield from generator_de_secuencias(test_ids, L=L)

    # 3) Definir los tipos de salida y las formas:
    #    - secuencia:  tf.float32, shape (L, time_steps, n_canales)
    #    - etiqueta:   tf.float32, shape (5,)
    # Para “time_steps” y “n_canales” debemos extraerlos de un paciente de ejemplo.
    #   Abrimos el primer archivo de train_ids para saber shapes:
    ejemplo_X, ejemplo_y, _ = cargar_paciente_npz(train_ids[0])
    time_steps, n_canales = ejemplo_X.shape[1], ejemplo_X.shape[2]  # (3000, num_canales)
    salida_secuencia_shape = (L, time_steps, n_canales)
    salida_etiqueta_shape = (5,)

    # 4) Construir los Dataset
    train_ds = tf.data.Dataset.from_generator(
        gen_train,
        output_signature=(
            tf.TensorSpec(shape=salida_secuencia_shape, dtype=tf.float32),
            tf.TensorSpec(shape=salida_etiqueta_shape, dtype=tf.float32)
        )
    )
    # Shuffle + Batch + Prefetch
    train_ds = (train_ds
                .shuffle(shuffle_buffer)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_generator(
        gen_val,
        output_signature=(
            tf.TensorSpec(shape=salida_secuencia_shape, dtype=tf.float32),
            tf.TensorSpec(shape=salida_etiqueta_shape, dtype=tf.float32)
        )
    )
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        gen_test,
        output_signature=(
            tf.TensorSpec(shape=salida_secuencia_shape, dtype=tf.float32),
            tf.TensorSpec(shape=salida_etiqueta_shape, dtype=tf.float32)
        )
    )
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds