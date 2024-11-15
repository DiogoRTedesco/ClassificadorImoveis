import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib  # Para salvar os escaladores

def train_model(data_path='housing_data.csv', save_path='model.h5'):
    # Carregar os dados
    data = pd.read_csv(data_path)
    X = data[['size', 'bedrooms', 'location', 'age']].values
    y = data['price'].values

    # Pré-processar os dados
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Criar e treinar o modelo
    model = Sequential([
        Dense(30, input_dim=X_train.shape[1], activation='relu'),
        Dense(30, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print(f"Modelo salvo em: {save_path}")
    print("Escaladores salvos como 'scaler_X.pkl' e 'scaler_y.pkl'")
    model.save(save_path)


if __name__ == "__main__":
    train_model()
