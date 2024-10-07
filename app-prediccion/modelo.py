import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('data.csv')

# Tratamiento de missing
data['Water_Thickness_GRACE'].fillna(data['Water_Thickness_GRACE'].median(), inplace=True)

# Variable Objetivo es la precipitación (lluvia)
X = data[['Temperature_A', 'Temperature_D', 'Humidity_A', 'Humidity_D', 'Water_Thickness_GRACE']]
y = data['Precipitation']

# Se divide el conjunto de datos en entrenamiento y prueba (con un 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se utiliza Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Se guarda el modelo
joblib.dump(model, 'precipitation_model.pkl')

# Convertir unidades para un mejor entendimiento de la precipitación
def convert_to_mm_h(kg_m2_s):
    """
    Convierte precipitación de kg m-2 s-1 a mm/h.
    
    Args:
        kg_m2_s (float): Precipitación en kg m-2 s-1.
        
    Returns:
        float: Precipitación en mm/h.
    """
    return kg_m2_s * 3600


def predict_precipitation(features):
    """
    Realiza una predicción de precipitación y convierte la unidad a mm/h.
    
    Args:
        features (pd.DataFrame): Características para la predicción.
        
    Returns:
        float: Precipitación predicha en mm/h.
    """
    prediction_kg_m2_s = model.predict(features)[0]  # Realiza la predicción
    prediction_mm_h = convert_to_mm_h(prediction_kg_m2_s)  # Convierte a mm/h
    return prediction_mm_h

# Evaluar el modelo 
def evaluate_model(X, y):
    """
    Evalúa el modelo y calcula el RMSE y R2.
    
    Args:
        X (pd.DataFrame): Características para la evaluación.
        y (pd.Series): Variable objetivo para la evaluación.
        
    Returns:
        tuple: RMSE y R2 del modelo.
    """
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return rmse, r2

# RMSE y R2 con el conjunto de entrenamiento
train_rmse, train_r2 = evaluate_model(X_train, y_train)
print("Entrenamiento - RMSE:", train_rmse)
print("Entrenamiento - R^2:", train_r2)

# RMSE y R2 con el conjunto de prueba
test_rmse, test_r2 = evaluate_model(X_test, y_test)
print("Prueba - RMSE:", test_rmse)
print("Prueba - R^2:", test_r2)




