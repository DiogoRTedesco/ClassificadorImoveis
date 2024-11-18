import numpy as np
from tensorflow.keras.models import load_model

def make_prediction(input_data, model_path='model.h5', scaler_X=None, scaler_y=None):
    model = load_model(model_path, compile=False)

    # Pré-processar os dados de entrada
    input_data_scaled = scaler_X.transform(input_data)

    # Fazer a predição
    predictions_scaled = model.predict(input_data_scaled)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    return predictions
