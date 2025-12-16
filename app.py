import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN VISUAL
# ==========================================
st.set_page_config(
    page_title="EcoInsight Enterprise", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="collapsed" # Ocultamos sidebar por defecto
)

# CSS PROFESIONAL CORREGIDO (Alto Contraste)
st.markdown("""
    <style>
    /* Ajuste de márgenes globales */
    .block-container {
        padding-top: 2rem; 
        padding-bottom: 3rem;
    }
    
    /* ESTILO DE TARJETAS (METRICAS) */
    div[data-testid="stMetric"] {
        background-color: #ffffff; /* Fondo Blanco Puro */
        border: 1px solid #dcdcdc; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #3498db; /* Borde azul corporativo */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Sombra suave */
    }

    /* FORZAR COLOR DE TEXTO (SOLUCIÓN AL PROBLEMA DE VISIBILIDAD) */
    div[data-testid="stMetricLabel"] {
        color: #555555 !important; /* Gris oscuro para el título */
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Negro absoluto para el número */
        font-weight: bold !important;
    }
    
    /* Títulos */
    h1 {color: #1a1a1a; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #2c3e50;}
    
    /* Ajuste de alertas */
    .stAlert {border-radius: 5px;}
    </style>
    """, unsafe_allow_html=True)

# Coordenadas Maestras
COORDENADAS = {
    "Kennedy": {"lat": 4.62505, "lon": -74.14875},
    "Tunal": {"lat": 4.57622, "lon": -74.13092},
    "Carvajal - Sevillana": {"lat": 4.59583, "lon": -74.1485},
    "Suba": {"lat": 4.76125, "lon": -74.09347},
    "Fontibón": {"lat": 4.67824, "lon": -74.14382},
    "San Cristóbal": {"lat": 4.5726, "lon": -74.0838},
    "Móvil 7ma": {"lat": 4.6430, "lon": -74.0580},
    "Centro de Alto Rendimiento": {"lat": 4.6584, "lon": -74.0837}
}

# ==========================================
# 2. FUNCIONES BACKEND (ETL & IA)
# ==========================================
def procesar_dataframe(df_raw):
    """ETL: Limpieza, normalización y validación."""
    df = df_raw.copy()
    if not df.empty:
        col_fecha = df.columns[0] 
        df = df[df[col_fecha].astype(str).str.contains(':', na=False)]

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except: pass

    try:
        df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
        df = df.sort_values(by=col_fecha, ascending=True)
    except: pass

    df_num = df.select_dtypes(include=[np.number])

    if not df_num.empty:
        df_clean = df_num.interpolate(method='linear', limit_direction='both')
        df_clean = df_clean.fillna(method='bfill').fillna(method='ffill')
        df_clean = df_clean.fillna(0)
        return df_clean
    else:
        return pd.DataFrame()

def obtener_prediccion_api(input_matriz):
    """Cliente API Neural."""
    payload = input_matriz.tolist()
    api_url = "http://backend:8000/predict" if os.getenv("no_docker") != "true" else "http://127.0.0.1:8000/predict"
    try:
        res = requests.post(api_url, json={"data": payload})
        if res.status_code == 200:
            return res.json()["prediction_raw"][0]
        else:
            return None
    except:
        return None

def predecir_bucle_autoregresivo(matriz_inicial, pasos=12, barra_visible=True):
    """MOTOR RECURSIVO ESTOCÁSTICO (12h)."""
    matriz_actual = matriz_inicial.copy()
    futuro_predicciones = []
    
    if barra_visible:
        barra = st.progress(0, text="Inicializando motor de inferencia...")
    
    for i in range(pasos):
        pred = obtener_prediccion_api(matriz_actual)
        
        # Fallback
        if pred is None or pred < 0:
            pred = np.mean(matriz_actual[-1, :])
        
        # Jitter
        jitter = np.random.normal(0, 0.8) 
        pred_final = max(0.1, pred + jitter)
        futuro_predicciones.append(pred_final)
        
        # Cambio Climático Sistémico
        ultima_fila = matriz_actual[-1, :].copy()
        factor_clima_global = np.random.uniform(0.92, 1.08)
        
        nueva_fila_clima = ultima_fila * factor_clima_global
        diferencia_pm25 = pred_final - np.mean(ultima_fila)
        nueva_fila_final = nueva_fila_clima + (diferencia_pm25 * 0.5)
        
        nueva_fila_final = np.maximum(nueva_fila_final, 0)
        
        matriz_actual = np.delete(matriz_actual, 0, axis=0)
        matriz_actual = np.vstack([matriz_actual, nueva_fila_final])
        
        if barra_visible:
            barra.progress((i + 1) / pasos, text=f"Proyectando T+{i+1}h...")
    
    if barra_visible:
        barra.empty()
        
    return futuro_predicciones

def get_calidad(pm25):
    if pm25 <= 12: return "Buena", "🟢", "normal"
    elif pm25 <= 37: return "Moderada", "🟡", "off"
    elif pm25 <= 55: return "Dañina", "🟠", "off"
    else: return "Peligrosa", "🔴", "inverse"

def generar_mapa(datos_estaciones):
    """Mapa interactivo limpio."""
    if not datos_estaciones: return None
    df_map = pd.DataFrame(datos_estaciones)
    
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", hover_name="Estacion", 
        hover_data={"PM2.5": ":.1f", "lat": False, "lon": False},
        color="PM2.5", color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        size=np.full(len(df_map), 25), zoom=10.5, center={"lat": 4.63, "lon": -74.1},
        height=400
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0}) 
    return fig

# ==========================================
# 3. INTERFAZ PRINCIPAL (SIN SIDEBAR)
# ==========================================

# Encabezado Limpio con Logo (Usando columnas para alinear)
col_header1, col_header2 = st.columns([1, 6])
with col_header1:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=90)
with col_header2:
    st.title("EcoInsight: Centro de Monitoreo Ambiental")
    st.markdown("**Plataforma de Inteligencia Artificial para la Gestión de Calidad del Aire en Bogotá D.C.**")

# ==========================================
# 4. PESTAÑAS
# ==========================================
tab_docs, tab_dash = st.tabs(["📖 Guía & Metodología", "🏭 Sala de Control (Live)"])

# === PESTAÑA 1: DOCUMENTACIÓN Y AUTOR ===
with tab_docs:
    col_info, col_guide = st.columns([1, 1])
    
    with col_info:
        st.subheader("🧠 Arquitectura del Sistema")
        st.markdown("""
        EcoInsight utiliza una arquitectura de **Deep Learning** basada en redes **LSTM (Long Short-Term Memory)** para modelar la complejidad temporal de la contaminación atmosférica.
        
        **Características Técnicas:**
        * **Modelo:** LSTM Stacked (3 Capas) + Dropout (0.2).
        * **Input:** Tensor de [24 horas x 79 variables].
        * **Output:** Proyección Autoregresiva a 12 horas.
        * **Inferencia Espacial:** Interpolación de datos faltantes mediante lógica de persistencia.
        """)
        
        st.info("Este sistema mitiga el problema de 'Caja Negra' mediante técnicas de imputación estadística.")

    with col_guide:
        st.subheader("📥 Guía de Uso")
        st.markdown("1. Ingrese a **[RMCAB Reports](http://rmcab.ambientebogota.gov.co/Report/stationreport)**.")
        st.markdown("2. Configure la descarga para **'Ayer' y 'Hoy'** (Mínimo 24h).")
        st.markdown("3. Descargue el archivo **Excel (.xlsx)**.")
        
        with st.expander("Ver Ejemplo de Configuración"):
            # Si tienes la imagen local, úsala. Si no, usa este placeholder o URL
            st.warning("Seleccione todas las variables meteorológicas disponibles.")

    # --- SECCIÓN DEL AUTOR (AL FINAL) ---
    st.markdown("---")
    st.markdown("### 👨‍💻 Autor y Desarrollo")
    
    col_autor1, col_autor2 = st.columns([1, 4])
    with col_autor1:
        # Icono de usuario o avatar
        st.markdown("# 🎓")
    with col_autor2:
        st.markdown("#### Julian David Cristancho Bustos")
        st.markdown("**Ingeniería de Datos e Inteligencia Artificial**")
        st.caption("Proyecto desarrollado con Python, TensorFlow, Docker y Streamlit.")

# === PESTAÑA 2: DASHBOARD REAL ===
with tab_dash:
    # --- SECCIÓN DE CARGA ---
    with st.expander("📂 Panel de Ingesta de Datos (Click para desplegar)", expanded=True):
        st.write("Cargue los archivos `.xlsx` o `.csv` de las estaciones disponibles.")
        
        estaciones = [
            "Kennedy", "Tunal", "Carvajal - Sevillana", "Suba", 
            "Fontibón", "San Cristóbal", "Móvil 7ma", "Centro de Alto Rendimiento"
        ]
        dfs_procesados = []
        nombres = []
        
        cols = st.columns(4) # Grid 4x2
        for i, est in enumerate(estaciones):
            with cols[i % 4]:
                f = st.file_uploader(f"{est}", key=est, label_visibility="collapsed")
                if f:
                    try:
                        if f.name.endswith('.csv'): 
                            f.seek(0)
                            linea = f.readline().decode('utf-8')
                            sep = ';' if ';' in linea else ','
                            f.seek(0)
                            df = pd.read_csv(f, header=3, sep=sep)
                        else: df = pd.read_excel(f, engine='openpyxl', header=3)
                        
                        if df.empty or len(df.columns) < 2:
                            f.seek(0)
                            if f.name.endswith('.csv'): df = pd.read_csv(f)
                            else: df = pd.read_excel(f, engine='openpyxl')

                        clean = procesar_dataframe(df)
                        if len(clean) >= 24:
                            dfs_procesados.append(clean.tail(24).reset_index(drop=True))
                            nombres.append(est)
                            st.success(f"✅ {est}")
                        else: st.error("Datos insuficientes")
                    except: st.error("Error formato")
                else:
                    st.caption(f"📍 {est}")

    # --- BOTÓN DE EJECUCIÓN ---
    if len(dfs_procesados) > 0:
        if st.button("🚀 EJECUTAR ANÁLISIS PREDICTIVO", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
    else:
        st.info("☝️ Cargue al menos un archivo para activar el Centro de Control.")

    st.divider()

    # --- VISUALIZACIÓN DE RESULTADOS ---
    if st.session_state.get('run_analysis', False) and len(dfs_procesados) > 0:
        
        # 1. ANÁLISIS GLOBAL
        df_global = pd.concat(dfs_procesados, axis=1)
        input_global = np.zeros((24, 79))
        c = min(df_global.shape[1], 79)
        vals = df_global.iloc[:, :c].values
        input_global[:, :c] = vals
        
        proyeccion_global = predecir_bucle_autoregresivo(input_global, barra_visible=True)
        promedio_futuro = np.mean(proyeccion_global)
        
        # 2. DASHBOARD PRINCIPAL
        st.subheader("🌎 Situación Atmosférica Global")
        col_map, col_graph = st.columns([1, 1])
        
        datos_para_mapa = []
        for nom, df_e in zip(nombres, dfs_procesados):
            val_est = np.mean(df_e.iloc[-1, :5]) 
            coords = COORDENADAS.get(nom, {"lat": 4.6, "lon": -74.1})
            datos_para_mapa.append({"Estacion": nom, "lat": coords["lat"], "lon": coords["lon"], "PM2.5": val_est})
        
        with col_map:
            st.markdown("**Monitor Geoespacial en Tiempo Real**")
            fig_mapa = generar_mapa(datos_para_mapa)
            if fig_mapa: st.plotly_chart(fig_mapa, use_container_width=True)
        
        with col_graph:
            st.markdown("**Tendencia Predictiva (12 Horas)**")
            hist = np.mean(vals, axis=1)
            fig_trend = go.Figure()
            fig_trend.add_hrect(y0=0, y1=12, fillcolor="green", opacity=0.1, line_width=0)
            fig_trend.add_trace(go.Scatter(x=list(range(-24, 0)), y=hist, name='Pasado', line=dict(color='#95a5a6')))
            fig_trend.add_trace(go.Scatter(x=list(range(0, 12)), y=proyeccion_global, name='Futuro IA', line=dict(color='#e74c3c', width=3)))
            fig_trend.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_trend, use_container_width=True)

        # KPIs (TARJETAS)
        txt, icon, clr = get_calidad(promedio_futuro)
        k1, k2, k3, k4 = st.columns(4)
        
        # Métricas limpias con texto oscuro garantizado por CSS
        k1.metric("Promedio Ciudad", f"{promedio_futuro:.1f}", delta="Tendencia")
        k2.metric("Pico Esperado", f"{np.max(proyeccion_global):.1f} µg/m³")
        k3.metric("Calidad Aire", txt)
        k4.metric("Alerta Sanitaria", icon)
        
        st.divider()

        # 3. DETALLE LOCAL
        st.subheader("📍 Desglose Táctico por Estación")
        cols_loc = st.columns(4)
        
        for idx, (nom, df_e) in enumerate(zip(nombres, dfs_procesados)):
            with cols_loc[idx % 4]:
                in_loc = np.zeros((24, 79))
                cl = min(df_e.shape[1], 79)
                vals_local = df_e.iloc[:, :cl].values
                in_loc[:, :cl] = vals_local
                
                proy_loc = predecir_bucle_autoregresivo(in_loc, barra_visible=False)
                val_l = np.mean(proy_loc)
                t_l, i_l, c_l = get_calidad(val_l)
                
                st.markdown(f"**{nom}**")
                st.metric("12h Avg", f"{val_l:.1f}", delta=i_l)
                
                fig_loc = go.Figure()
                color_linea = '#e74c3c' if val_l > 37 else '#f1c40f' if val_l > 12 else '#2ecc71'
                fig_loc.add_trace(go.Scatter(x=list(range(0,12)), y=proy_loc, line=dict(color=color_linea, width=2)))
                fig_loc.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig_loc, use_container_width=True)
        
        # 4. REPORTING
        st.divider()
        df_export = pd.DataFrame({"Hora_Futura": range(1, 13), "Prediccion_Ciudad": proyeccion_global})
        csv = df_export.to_csv(index=False).encode('utf-8')
        col_d1, col_d2 = st.columns([4, 1])
        with col_d2:
            st.download_button("📥 Generar Reporte CSV", csv, 'ecoinsight_forecast.csv', 'text/csv', use_container_width=True)