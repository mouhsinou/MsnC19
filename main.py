from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Charger le modèle Keras
model = load_model('model.keras')

# Charger le scaler utilisé pour la normalisation
scaler = joblib.load('scaler.joblib')

app = FastAPI()

# Définir le modèle de données d'entrée
class InputData(BaseModel):
    Min_Sec: float
    D1: float
    D2: float
    D3: float
    D4: float
    D5: float
    D6: float
    D7: float
    D8: float
    D9: float
    D10: float
    D11: float
    D12: float
    D13: float
    D14: float
    D15: float
    D16: float
    D17: float
    D18: float
    D19: float
    D20: float
    D21: float
    D22: float
    D23: float
    D24: float
    D25: float
    D26: float
    D27: float
    D28: float
    D29: float
    D30: float
    D31: float
    D32: float
    D33: float
    D34: float
    D35: float
    D36: float
    D37: float
    D38: float
    D39: float
    D40: float
    D41: float
    D42: float
    D43: float
    D44: float
    D45: float
    D46: float
    D47: float
    D48: float
    D49: float
    D50: float
    D51: float
    D52: float
    D53: float
    D54: float
    D55: float
    D56: float
    D57: float
    D58: float
    D59: float
    D60: float
    D61: float
    D62: float
    D63: float
    D64: float

@app.post("/predict/")
async def predict(data: InputData):
    # Convertir les données d'entrée en DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Prétraiter les données (mise à l'échelle)
    input_scaled = scaler.transform(input_df)
    
    # Faire la prédiction
    prediction_prob = model.predict(input_scaled)
    prediction = prediction_prob.argmax(axis=1)[0]  # Choisir la classe avec la probabilité maximale
    
    # Convertir la prédiction en label
    label = ' TEST POSITIVE' if prediction == 1 else 'TEST NEGATIVE'
    
    return {"prediction": label}

@app.post("/predict_batch/")
async def predict_batch(data: list[InputData]):
    # Convertir les données d'entrée en DataFrame
    input_df = pd.DataFrame([item.dict() for item in data])
    
    # Prétraiter les données (mise à l'échelle)
    input_scaled = scaler.transform(input_df)
    
    # Faire les prédictions
    predictions_prob = model.predict(input_scaled)
    predictions = predictions_prob.argmax(axis=1)
    
    # Convertir les prédictions en labels
    labels = ['POSITIVE' if pred == 1 else 'NEGATIVE' for pred in predictions]
    
    return {"predictions": labels}
