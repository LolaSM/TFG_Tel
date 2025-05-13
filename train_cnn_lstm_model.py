import numpy as np
import os
from tensorflow.keras.callbacks import CSVLogger
from modelos.cnn_lstm_model import build_cnn_lstm_model, compile_model, get_callbacks
from preprocessing.dataset_builder import cargar_datos_secuenciales
from preprocessing.procesado_pacientes import procesamiento_completo
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
RECORDINGS_PATH = "recordings"
DATASET_PATH = "pacientes"
SEQ_LENGTH = 5
BATCH_SIZE = 100
EPOCHS = 30
MODEL_PATH = "modelo_cnn_lstm_5"
#canales = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']
CANALES_INTERES = [
    'EEG C4-M1', 'EEG C3-M2', 'EMG chin', 'EOG E1-M2', 'ECG'
]

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
# CARGAR DATOS SECUENCIALES
# ----------------------------
(X_train, y_train), (X_val, y_val), (X_test, y_test) = cargar_datos_secuenciales(DATASET_PATH, L=SEQ_LENGTH)
#X_train tiene la forma:(n_samples, sequence_length, time_steps, n_channels)
input_shape = X_train.shape[2:]#será (3000, 8) para multicanal --> secuencias de 30s muestreadas a 100Hz de 8 canales
#me quedo con (time_steps, n_channels) porque el modelo espera un input de la forma (L, time_steps, n_channels)

# ----------------------------
# CONSTRUIR Y ENTRENAR MODELO
# ----------------------------
model = build_cnn_lstm_model(seq_len=SEQ_LENGTH, input_shape=input_shape)
model = compile_model(model)

callbacks = get_callbacks()
csv_logger = CSVLogger("training_log.csv")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks + [csv_logger],
    verbose=1
)

model.save(MODEL_PATH)

# ----------------------------
# GRAFICAR HISTORIA DE ENTRENAMIENTO
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# EVALUACIÓN FINAL
# ----------------------------
#y_test y y_pred son matrices con forma (n_samples, 5), donde cada fila es un vector de 5 probabilidades, una por clase (W, N1, N2, N3, R)
#argmax devuelve el índice de la clase con mayor probabilidad, que es la etiqueta predicha por el modelo
#predicciones sobre el conjunto de test de las etiquetas
y_pred = model.predict(X_test)
#y_true es la matriz de etiquetas reales, y_pred_labels es la matriz de etiquetas predichas
y_true = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

kappa = cohen_kappa_score(y_true, y_pred_labels)
print(f"\nCohen's Kappa: {kappa:.4f}")
