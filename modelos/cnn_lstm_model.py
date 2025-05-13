import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, AveragePooling1D,
    ReLU, Dense, Flatten, LSTM, TimeDistributed
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_cnn_encoder(input_shape):
    """
    CNN feature extractor with 3 operational blocks:
    Conv1D(kernel_size=100) + ReLU + BatchNorm + AvgPool1D(pool=2)
    Accepts multichannel input, e.g., (3000, 7) for 4 physiological signals 3xEEG,1xEMG,2xEOG,1xECG.
    """
    #HACER UNA CAPA PARA CADA SEÑAL Y ASÍ EXTRAER CARACTERISTICAS DE CADA UNA
    inputs = Input(shape=input_shape)  
    x = inputs
    filters = 8
    for _ in range(3):
        x = Conv1D(filters=filters, kernel_size=100, padding='same', strides=1)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(pool_size=2, strides=2)(x)
        filters *= 2  # double filters each block

    x = Flatten()(x)
    x = Dense(50)(x)  # output feature vector of length 50

    model = Model(inputs=inputs, outputs=x, name="CNN_Encoder")
    return model

#test seq_len = 3,5,7
#5 channels
def build_cnn_lstm_model(seq_len=5, input_shape=(3000, 5)):
    """
    Builds the full CNN-LSTM model as described in the article.
    - CNN feature extractor applied on each 30s segment
    - LSTM takes sequences of feature vectors of length L
    - Final Dense layer with softmax for 5-class classification
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
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy','f1_score','kappa_score']
    )
    return model


def get_callbacks():
    """
    Returns callbacks: EarlyStopping and ReduceLROnPlateau
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stopping, lr_schedule]
