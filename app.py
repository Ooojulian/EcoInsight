import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="EcoInsight Dashboard",
    page_icon="🌍",
    layout="wide"
)

# TÍTULO Y DESCRIPCIÓN
st.title("🌍 EcoInsight: Monitor de Calidad del Aire")
st.markdown("""
Este sistema utiliza **Inteligencia Artificial (LSTM)** para predecir la contaminación (PM2.5) 
en Bogotá basándose en patrones atmosféricos de las últimas 24 horas.
""")

# COLUMNAS PARA ORGANIZAR LA VISTA
col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Panel de Control")
    st.info("Como no tenemos sensores reales conectados en este momento, simularemos los datos de entrada.")
    
    # Botón para simular
    if st.button("📡 Simular Datos de Sensores (24h)", use_container_width=True):
        st.session_state['simulando'] = True

# LÓGICA DE PREDICCIÓN
if st.session_state.get('simulando'):
    with st.spinner('Consultando API de Inferencia...'):
        try:
            # 1. GENERAR DATOS FALSOS (Simulación de lo que enviaría el hardware)
            # Deben ser 24 horas x 79 variables (según tu modelo)
            N_VARS = 79 
            fake_data = np.random.rand(24, N_VARS).tolist()
            
            # 2. LLAMAR A TU API (Backend FastAPI)
            # Asegúrate de que uvicorn esté corriendo en el puerto 8000
            api_url = "http://127.0.0.1:8000/predict"
            payload = {"data": fake_data}
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediccion_valor = result["prediction_raw"][0] # Asumimos que predice la primera columna (ej: Sevillana)
                
                # MOSTRAR RESULTADOS EN LA COLUMNA 2
                with col2:
                    st.success("✅ Predicción Exitosa")
                    
                    # Métrica Grande
                    st.metric(
                        label="Pronóstico PM2.5 (Próxima Hora)", 
                        value=f"{prediccion_valor:.2f} µg/m³",
                        delta="-1.5 µg/m³ vs hora anterior" # Simulado
                    )
                    
                    # GRÁFICA INTERACTIVA
                    # Simulamos datos históricos para que la gráfica se vea bonita
                    historia = np.random.uniform(10, 35, 24)
                    futuro = np.append(historia, prediccion_valor)
                    
                    fig = go.Figure()
                    
                    # Línea de historia
                    fig.add_trace(go.Scatter(
                        y=historia, 
                        mode='lines+markers', 
                        name='Últimas 24h',
                        line=dict(color='#00d2be')
                    ))
                    
                    # Punto de predicción
                    fig.add_trace(go.Scatter(
                        x=[24], 
                        y=[prediccion_valor], 
                        mode='markers', 
                        name='Predicción IA',
                        marker=dict(color='red', size=12, symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="Tendencia de Contaminación (Real vs Predicción)",
                        xaxis_title="Horas",
                        yaxis_title="PM2.5 (µg/m³)",
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.json(result) # Mostrar el JSON crudo para depuración
            else:
                st.error(f"Error en la API: {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"No se pudo conectar con el Backend. ¿Está encendido? \nError: {e}")
