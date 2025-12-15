# 🌍 EcoInsight: Sistema de Predicción de Calidad del Aire con LSTM

![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![TensorFlow](https://img.shields.io/badge/AI-TensorFlow%20Keras-orange)

## 📖 Descripción
EcoInsight es un sistema *End-to-End* de Ingeniería de Datos e Inteligencia Artificial diseñado para predecir la concentración de material particulado (PM2.5) en las estaciones de monitoreo de Bogotá.

El sistema utiliza una arquitectura de **Red Neuronal Recurrente (LSTM)** entrenada con datos históricos (2017-2025), implementando ingeniería de características cíclicas (Trigonometría temporal) y vectores de viento.

## 🚀 Arquitectura del Proyecto

1.  **ETL Pipeline:** Procesamiento de +160 archivos Excel (RMCAB), limpieza dinámica e imputación de datos faltantes con **MICE (Iterative Imputer)**.
2.  **Modelo AI:** LSTM (Long Short-Term Memory) optimizada con **Keras Tuner**.
    * *Métricas:* R² Score: 0.61 (Rendimiento Competitivo).
3.  **Backend:** API REST construida con **FastAPI** para inferencia en tiempo real.

## 🛠️ Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/EcoInsight.git](https://github.com/TU_USUARIO/EcoInsight.git)
cd EcoInsight
