# Sleep Stage Detection with CNN-LSTM
Este proyecto implementa un modelo de aprendizaje profundo basado en una arquitectura **CNN-LSTM** para la detección automática de fases del sueño. Se utilizan señales fisiológicas multicanal extraídas de registros de polisomnografía (PSG), incluyendo:

- EEG C4-M1
- EEG C3-M2
- EMG chin
- EOG E1-M2
- ECG

## Flujo
1º Carga de archivos .edf con señales fisiológicas y su preprocesamiento
2º Extracción de ventanas de 30s con sus respectivas etiquetas de sueño que se guardan en diccionarios .npz
3º Agrupación de las ventanas en secuencias de longitud L=5
4º Entrada al modelo completo de la forma (shape = [n_secuencias, sequence_length(L=5), 3000(30sx100Hz), n_canales(5)])
5º La CNN procesa cada ventana (3000, 5) y extrae un vector de 50 características hecho secuencialmente (TimeDistributed)
6º La LSTM procesa la secuencia de 5 vectores (sequence_length(L=5), 50) y predice la etapa de sueño.

## Arquitectura del modelo

El modelo está compuesto por dos bloques principales:

### CNN (Feature extractor): Extrae un vector de 50 características de cada ventana de las señales (3000,5).
Conv1D espera entradas de la forma (batch_size-->n_ventanas, time_steps-->muestras, channels-->n_canales) es decir 
- 3 bloques operacionales, cada uno con:
  - `Conv1D`: kernel `(1x100)`, padding `'same'`, stride `1`, filtros: `8 → 16 → 32`
  - `ReLU`
  - `BatchNormalization`
  - `AveragePooling1D`: pool size `2`, stride `2`
- `Flatten`
- `Dense(50)`: vector de características de salida

### LSTM (Sequence modeling): Aprende la dinámica temporal a partir de secuencias de 5 vectores de 50
- Entrada: secuencia de vectores de 50 características (`L` longitudes --5)
- `LSTM(100)` unidades
- `Dense(5, activation='softmax')`: salida con 5 clases para clasificar en fases W, N1, N2, N3, R

### Preprocesamiento
- Resampleo: 100 Hz para reducir coste computacional
- Filtro paso banda: 0.5-40 Hz
- Ventanas de duración de 30s

### División datos (subject-wise)
- Entrenamiento: 60% --> 151 pacientes
- Validación: 20% --> 30 pacientes
- Test: 20% --> 31 pacientes

### Entrenamiento
- **Optimizer**: SGD (lr=0.001, momentum=0.9)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 100 (el modelo procesa 100 secuencias a la vez)
- **Epochs**: 30
- **EarlyStopping**: monitor `val_loss`, paciencia 10
- **ReduceLROnPlateau**: factor 0.1, min_lr=1e-6
- **Metric**: Accuracy + F1_score + Cohen's Kappa

### Visualización
- Loss entrenamiento VS validación por epoch
- Matriz de confusión

## Estructura del proyecto

sleep-stage-detection/
│
├── recordings/ # Archivos .edf originales (no se suben al repo)
├── pacientes/ # Archivos procesados .npz (no se suben al repo)
│
├── modelos/ # Definición del modelo CNN-LSTM
│ ├── cnn_lstm_model.py
│
├── preprocessing/ # Procesamiento de datos y anotaciones
│ ├── procesado_pacientes.py
│ ├── dataset_builder.py
│
├── train_cnn_lstm_model.py # Script principal de entrenamiento y evaluación
├── .gitignore
└── README.md

## Requisitos

- Python 3.10+
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn, MNE
- Scikit-learn

## Ejecución del proyecto

python train_cnn_lstm_model.py

