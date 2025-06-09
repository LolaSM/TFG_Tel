import numpy as np
import os
from tensorflow.keras.callbacks import CSVLogger
from collections import Counter
from modelos.cnn_lstm_model import build_cnn_lstm_model, compile_model, get_callbacks
from preprocessing.dataset_builder import crear_tf_dataset
from preprocessing.procesado_pacientes import procesamiento_completo
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
RECORDINGS_PATH = "recordings"
DATASET_PATH = "pacientes"
SEQ_LENGTH = 5
BATCH_SIZE = 64 # 32, 64, 128
EPOCHS = 100 # 100 epochs
MODEL_PATH = "modelos/CNN_LSTM_5_ECG_100epochs.keras"
#canales = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']
#CANALES_INTERES = [
#    'EEG C4-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'ECG'
#]
#probar con solo ECG
CANALES_INTERES = ['ECG']

# ----------------------------
# PREPROCESAR PACIENTES
# ----------------------------
npz_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".npz")]
if not npz_files:
    print("\n--- No se encontraron archivos .npz. Iniciando preprocesamiento ---")
    procesamiento_completo(RECORDINGS_PATH, DATASET_PATH,canales=CANALES_INTERES)
    npz_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".npz")]
else:
    print("\n--- Archivos .npz detectados. Omitiendo preprocesamiento ---")

    print(f"--- {len(npz_files)} pacientes cargados ---")

# ----------------------------
# CREAR DATASETS CON tf.data
# ----------------------------
#Creación dinámica de train_ds, val_ds y test_ds a partir de la carpeta pacientes/, sin concatenar nada en memoria.
#tf.data detecta de forma automática el tamaño de cada dataset (en TF2), así que no es necesario calcular steps_per_epoch manualmente.
#En lugar de concatenar todos los X de todos los pacientes en un solo np.array, el generador va “pidiendo” paciente a paciente
# y solo mantiene en memoria las ventanas necesarias para armar cada batch.

print("\nCreando tf.data.Datasets para train/val/test...")
train_ds, val_ds, test_ds = crear_tf_dataset(path_npz_folder=DATASET_PATH,
                                             L=SEQ_LENGTH,
                                             batch_size=BATCH_SIZE,
                                             shuffle_buffer=1000)

# Determinar cuántos pasos por época (opcional: contar el número de elementos en train_ds)
# Si no queremos calcularlo exacto, podemos pasar steps_per_epoch=None y Keras .
# Lo detectará automáticamente en TF2. En versiones más antiguas, conviene hacer:
#
#   total_train_items = sum(1 for _ in generator_de_secuencias(train_ids, L=SEQ_LENGTH))
#   steps_per_epoch = total_train_items // BATCH_SIZE
#   total_val_items   = sum(1 for _ in generator_de_secuencias(val_ids,   L=SEQ_LENGTH))
#   validation_steps = total_val_items // BATCH_SIZE
#
# Pero TF2 detecta tamaño en dataset que no repite indefinidamente si no se usa .repeat().

# ----------------------------
# CONSTRUIR Y COMPILAR MODELO
# ----------------------------
# Primero necesitamos saber input_shape: (seq_len, time_steps, n_canales)
# Podemos extraerlo directamente de train_ds.element_spec
#   element_spec[0].shape == (None, SEQ_LENGTH, TIME_STEPS, N_CANALES)
input_shape = train_ds.element_spec[0].shape[1:]  # (SEQ_LENGTH, time_steps, n_canales)
print(f"Input shape detectado: {input_shape}")
model = build_cnn_lstm_model(seq_len=SEQ_LENGTH, input_shape=input_shape[1:])
model = compile_model(model)
model.summary()

# ----------------------------
# CALLBACKS
# ----------------------------
csv_logger = CSVLogger("training_log.csv", append=False)
# Opcional: guardamos checkpoint cada 50 épocas
#checkpoint = ModelCheckpoint(
    #filepath="modelos/CNN_LSTM_checkpoint_{epoch:03d}.h5",
    #save_freq='epoch',
    #period=50,
    #save_best_only=False,
    #verbose=1
#)

# ----------------------------
# ENTRENAMIENTO
# ----------------------------
print("\n> Iniciando entrenamiento (200 epochs sin early stopping)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[csv_logger],
    verbose=1
)

# Guardar modelo final
model.save(MODEL_PATH)
print(f"\nModelo completo guardado en: {MODEL_PATH}")

# ----------------------------
# GRAFICAR HISTORIA DE ENTRENAMIENTO
# ----------------------------
history_dict = history.history
epochs_range = range(len(history_dict['loss']))

plt.figure(figsize=(14, 5))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history_dict['loss'], label='Train Loss')
plt.plot(epochs_range, history_dict['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_dict['accuracy'], label='Train Acc')
plt.plot(epochs_range, history_dict['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ----------------------------
# EVALUACIÓN FINAL EN test_ds
# ----------------------------
# Para obtener predicciones y métricas, necesitamos recopilar todo test_ds en memoria
# (en tf2 es sencillo: model.evaluate en un dataset, pero la matriz de confusión
#  requiere predicciones y etiquetas por separado):

y_true = []
y_pred = []

print("\n> Generando predicciones sobre test set...")
for batch_X, batch_y in test_ds:
    preds = model.predict(batch_X)
    y_pred.append(preds)
    y_true.append(batch_y.numpy())

y_pred = np.concatenate(y_pred, axis=0)  # shape total_test x 5
y_true = np.concatenate(y_true, axis=0)  # shape total_test x 5

y_true_labels    = np.argmax(y_true, axis=1)
y_pred_labels    = np.argmax(y_pred, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_true_labels, y_pred_labels)
print("\nConfusion Matrix:")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Cohen's Kappa
kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
print(f"\nCohen's Kappa: {kappa:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=["W", "N1", "N2", "N3", "R"]))

