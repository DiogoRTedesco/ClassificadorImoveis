from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Blueprint para organizar as rotas
predictions_bp = Blueprint('predictions', __name__)

# Carregar o modelo e os escaladores uma vez quando o servidor iniciar
model = load_model('model.h5')  # Caminho para o modelo treinado
scaler_X = joblib.load('scaler_X.pkl')  # Caminho para o escalador X
scaler_y = joblib.load('scaler_y.pkl')  # Caminho para o escalador y

@predictions_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados da requisição
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validar entrada
        required_fields = ["size", "bedrooms", "location", "age"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": f"Missing fields: {required_fields}"}), 400

        # Extrair os valores
        size = data["size"]
        bedrooms = data["bedrooms"]
        location = data["location"]
        age = data["age"]

        # Converter para array
        input_array = np.array([[size, bedrooms, location, age]])

        # Normalizar os dados de entrada
        X_scaled = scaler_X.transform(input_array)

        # Realizar a predição
        prediction_scaled = model.predict(X_scaled)

        # Desnormalizar o valor predito
        prediction = scaler_y.inverse_transform(prediction_scaled)

        # Convertendo para um valor de tipo nativo Python (float)
        predicted_price = float(prediction[0][0])

        # Retornar o preço predito como resposta
        return jsonify({"predicted_price": predicted_price})

       
    except Exception as e:
        return jsonify({"error": str(e)}), 500
