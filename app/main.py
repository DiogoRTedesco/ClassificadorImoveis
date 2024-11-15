from flask import Flask, request, jsonify
from routes.predictions import predictions_bp

# Criar a aplicação Flask
app = Flask(__name__)

# Registrar o blueprint
app.register_blueprint(predictions_bp, url_prefix='/api')

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "API para Previsão de Preços de Casas"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
