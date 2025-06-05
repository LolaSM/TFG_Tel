import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, AveragePooling1D,
    ReLU, Dense, GlobalAveragePooling1D, LSTM, TimeDistributed
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from modelos.metrics import F1Score 

def build_cnn_encoder(input_shape):
    """
    CNN feature extractor with 3 operational blocks:
    Conv1D(kernel_size=100) + ReLU + BatchNorm + AvgPool1D(pool=2)
    Accepts multichannel input, e.g., (3000, 5) for 5 physiological signals 2xEEG,1xEMG,1xEOG,1xECG.
    """
    #HACER UNA CAPA PARA CADA SEÑAL Y ASÍ EXTRAER CARACTERISTICAS DE CADA UNA
    inputs = Input(shape=input_shape)  
    x = inputs
    filters = 8
    #probar con kernel de 50, añadir droput tras la capa Dense(50)
    for _ in range(3):
        x = Conv1D(filters=filters, kernel_size=100, padding='same', strides=1)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size=2, strides=2)(x)
        filters *= 2  # double filters each block
        
    # Ahora x tiene shape (time_steps // 8, filters_final)
    x = GlobalAveragePooling1D()(x)  # shape → (filters_final,)
    x = Dense(50, activation='relu')(x)  # feature vector de dimensión 50
    #x = Dropout(0.5)(x)  # dropout para regularización

    model = Model(inputs=inputs, outputs=x, name="CNN_Encoder")
    return model

#test seq_len = 3,5,7
#5 channels
def build_cnn_lstm_model(seq_len=5, input_shape=(3000, 1)):
    """
    Modelo completo CNN-LSTM:
      - Cada ventana de 30s pasa por build_cnn_encoder → vector de 50
      - TimeDistributed sobre seq_len ventanas → tensor (seq_len, 50)
      - Dense final 5 clases softmax
    
    The value 3000 refers to the number of time samples in each 30-second segment.
    For example, if your data is sampled at 100 Hz, then 30 seconds * 100 Hz = 3000 samples.
    Adjust this value according to your signal's sampling frequency and segment length.
    """
    cnn_encoder = build_cnn_encoder(input_shape)

    # Input is a sequence of L segments (each of shape input_shape)
    seq_input = Input(shape=(seq_len, *input_shape))  # (L, 3000, n_canales)

    # Apply CNN encoder to each time step
    x = TimeDistributed(cnn_encoder)(seq_input)  # → (L, 50)

    # LSTM block
    x = LSTM(100)(x)  # output shape → (100,)

    # Final classification layer
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=seq_input, outputs=output, name="CNN_LSTM_5")
    return model

def compile_model(model):
    """
    Compiles model with SGD and learning rate scheduler.
    """
    #luego probar a cambiar learning_rate u optimizador adam
    optimizer = SGD(learning_rate=0.001, momentum=0.9)
    #lr=1e-3
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', F1Score()]
    )
    return model


def get_callbacks():
    """
    Returns callbacks: EarlyStopping and ReduceLROnPlateau
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stopping, lr_schedule]

#utilizar tarjeta grafica en entorno distribuido

