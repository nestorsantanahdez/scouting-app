import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# --- 1. ESTÉTICA DE ALTO CONTRASTE (OPTIMIZADA PARA VISIBILIDAD MÓVIL) ---
st.set_page_config(page_title="Scouting Pro 🏀", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    /* Fondo claro y limpio */
    .main { background-color: #FFFFFF; }
    
    /* Tarjetas con borde definido y texto negro para máxima lectura */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    
    /* Títulos en NEGRO INTENSO */
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 800 !important;
        color: #000000 !important; /* Negro puro */
        text-transform: uppercase;
    }
    
    /* Valores en Rojo Deportivo para que resalten */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #E11D48 !important; /* Rojo vibrante */
        font-weight: bold;
    }

    /* Botón con contraste invertido */
    .stButton>button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        height: 4em !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def cargar_recursos():
    try:
        with open('modelo_final_8_stats.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

data = cargar_recursos()
if not data:
    st.error("Archivo no encontrado en GitHub.")
    st.stop()

modelo, scaler, mae_por_stat, stats_nombres = data['modelo'], data['scaler'], data['mae'], data['stats']

# --- 3. INTERFAZ ---
st.title("🏀 SCOUTING EBA ➔ U22")

# --- 4. ESCÁNER ---
with st.expander("📝 PEGAR DATOS EXCEL/WEB", expanded=False):
    raw_data = st.text_input("Pega la línea aquí:")
    if raw_data:
        clean_text = raw_data.replace('%', '').replace(',', '.')
        parts = clean_text.split()
        if len(parts) >= 18:
            try:
                def f(x):
                    if ":" in str(x):
                        p = str(x).split(":")
                        return float(p[0]) + float(p[1])/60
                    return float(x)
                # Mapeo corregido
                st.session_state.inputs = [f(parts[1]), f(parts[2]), f(parts[9]), f(parts[10]), f(parts[11]), f(parts[12]), f(parts[13]), f(parts[17])]
                st.success("✅ Datos cargados correctamente.")
            except: st.error("Error en formato.")

# --- 5. PANEL LATERAL ---
st.sidebar.header("📊 DATOS EBA")
final_inputs = []
for i, stat in enumerate(stats_nombres):
    default_val = st.session_state.inputs[i] if 'inputs' in st.session_state else 5.0
    val = st.sidebar.number_input(f"{stat}", value=float(default_val), step=0.1, key=f"s_{stat}")
    final_inputs.append(val)

# --- 6. CÁLCULO ---
if st.button("🚀 GENERAR INFORME", use_container_width=True):
    X_input = np.array([final_inputs])
    if scaler: X_input = scaler.transform(X_input)
    
    pred = modelo.predict(X_input)[0] 

    st.subheader("🎯 PROYECCIÓN U22")
    
    cols = st.columns(2) 
    for i, stat in enumerate(stats_nombres):
        val_p = max(0, pred[i])
        err = mae_por_stat[i]
        with cols[i % 2]:
            # Abreviaturas para que no se amontone el texto
            label_móvil = stat.replace("Rebotes total", "REBOTES").replace("Faltas recibidas", "F. RECIBIDAS").replace("Tapones favor", "TAPONES")
            st.metric(label=label_móvil, value=f"{val_p:.1f}")
            st.caption(f"Margen: ±{err:.1f}")

    # GRÁFICO
    st.subheader("📈 COMPARATIVA")
    chart_df = pd.DataFrame({
        'Stat': stats_nombres,
        'EBA': final_inputs,
        'U22': pred
    }).set_index('Stat')
    st.bar_chart(chart_df, color=["#000000", "#E11D48"])
