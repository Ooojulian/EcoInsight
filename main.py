import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. INICIALIZACIÓN DE LA APP
app = FastAPI(
    title="EcoInsight API 🌍",
    description="API de Deep Learning para predicción de calidad del aire en Bogotá (LSTM)",
    version="1.0.0"
)

# Variables globales para cargar los artefactos
model = None
scaler = None

# 2. CARGA DE ARTEFACTOS AL INICIO
# Esto se ejecuta una sola vez cuando prendes el servidor
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        # Cargar modelo LSTM
        model = tf.keras.models.load_model("eco_insight_lstm.keras")
        print("✅ Modelo cargado correctamente.")
        
        # Cargar Scaler
        scaler = joblib.load("scaler_maestro.pkl")
        print("✅ Scaler cargado correctamente.")
    except Exception as e:
        print(f"❌ Error cargando artefactos: {e}")
        print("Asegúrate de que los archivos .keras y .pkl están en la misma carpeta.")

# 3. DEFINICIÓN DE LA ESTRUCTURA DE DATOS (Input Schema)
# El usuario debe enviar una lista de listas (Matriz de 24 horas x N variables)
class PredictionInput(BaseModel):
    # Ejemplo: [[0.1, 0.2...], [0.1, 0.3...] ... 24 veces]
    data: List[List[float]] 

# 4. ENDPOINT DE PREDICCIÓN
@app.post("/predict")
def predict_air_quality(payload: PredictionInput):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        # A. Convertir JSON a Numpy Array
        input_data = np.array(payload.data)
        
        # Validación básica de forma
        # Esperamos (24, N_Variables). Si envían menos horas, fallará.
        if input_data.shape[0] != 24:
             raise HTTPException(status_code=400, detail=f"Se requieren exactamente 24 horas de datos. Recibido: {input_data.shape[0]}")

        # B. Normalización (Usando el scaler entrenado)
        # IMPORTANTE: El scaler espera 2D (24, Features)
        input_scaled = scaler.transform(input_data)
        
        # C. Reshape para la LSTM (1, 24, Features)
        # La red espera un bloque 3D: (1 muestra, 24 tiempos, N variables)
        input_reshaped = input_scaled.reshape(1, 24, input_scaled.shape[1])
        
        # D. Predicción (Inferencia)
        prediction_scaled = model.predict(input_reshaped)
        
        # E. Des-normalización (Volver a µg/m³)
        prediction_real = scaler.inverse_transform(prediction_scaled)
        
        # F. Respuesta
        return {
            "status": "success",
            "prediction_raw": prediction_real.tolist()[0], # Convertir a lista simple
            "message": "Predicción generada exitosamente."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint de prueba para ver si vive
@app.get("/")
def home():
    return {"message": "EcoInsight API está en línea 🚀"}
