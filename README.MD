# ROTAS
    * / = rota default

    * api/predict  - rota para predição de valores de imvóveis
        - inputs:
        {
            "size": 1500, // tamanho
            "bedrooms": 3, // Banheiros
            "location": 1, // 1 urbano 0 interior
            "age": 10 // Idade do imóvel
        }

# OBS
    * Se precisar treinar o modelo novamente
        - python app/model/train.py

# Executar API
    * python app/main.py

# Com docker-compose 
    - docker-compose up --build (primeira vez)
    - docker-compose up