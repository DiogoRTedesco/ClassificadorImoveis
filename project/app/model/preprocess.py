import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib 
def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    X = data[['size', 'bedrooms', 'location', 'age']].values
    y = data['price'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Salvar os escaladores ajustados
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    
    return X_scaled, y_scaled, scaler_X, scaler_y
