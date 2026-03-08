import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# --- 1. ESTÉTICA ELEGANTE Y LEGIBLE (AZUL Y GRIS SUAVE) ---
st.set_page_config(page_title="Scouting Pro 🏀", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    /* Fondo principal limpio */
    .main { background-color: #F1F5F9; }
    
    /* Tarjetas de resultados: Fondo blanco, borde azul suave */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-top: 5px solid #1E40AF; /* Azul elegante */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* TÍTULOS: Azul muy oscuro para que se lea bien en móvil */
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #0F172A !important; /* Azul casi negro */
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* VALORES: Azul profesional resaltado */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #1E40AF !important; 
        font-weight: 800;
    }

    /* Botón azul con bordes redondeados */
    .stButton>button {
        background-color: #1E40AF !important;
        color: #FFFFFF !important;
        height: 3.5em !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        transform: scale(1.02);
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
    st.error("Archivo 'modelo_final_8_stats.pkl' no encontrado.")
    st.stop()

modelo, scaler, mae_por_stat, stats_nombres = data['modelo'], data['scaler'], data['mae'], data['stats']

# --- 3. INTERFAZ ---
st.title("🏀 Scouting EBA ➔ U22")
st.write("Análisis de proyección basado en datos históricos.")

# --- 4. ESCÁNER ---
with st.expander("📝 PEGAR DATOS (PC / EXCEL)", expanded=False):
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
                # Mapeo: 1:MIN, 2:PT, 9:RT, 10:AS, 11:BR, 12:BP, 13:TF, 17:FR
                st.session_state.inputs = [f(parts[1]), f(parts[2]), f(parts[9]), f(parts[10]), 
                                          f(parts[11]), f(parts[12]), f(parts[13]), f(parts[17])]
                st.success("✅ Datos importados correctamente.")
            except: st.error("Formato no reconocido.")

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
    
    pred_raw = modelo.predict(X_input)
    pred = pred_raw[0]

    st.subheader("🎯 PROYECCIÓN EN U22")
    
    cols = st.columns(2) 
    for i, stat in enumerate(stats_nombres):
        val_p = max(0, pred[i])
        err = mae_por_stat[i]
        with cols[i % 2]:
            # Abreviaturas elegantes
            label_display = stat.replace("Rebotes total", "REBOTES").replace("Faltas recibidas", "F. RECIBIDAS").replace("Tapones favor", "TAPONES")
            st.metric(label=label_display, value=f"{val_p:.1f}")
            st.caption(f"Confianza: ±{err:.1f}")

    # GRÁFICO
    st.subheader("📈 COMPARATIVA EBA vs U22")
    chart_df = pd.DataFrame({
        'Stat': stats_nombres,
        'EBA': final_inputs,
        'U22': pred
    }).set_index('Stat')
    st.bar_chart(chart_df, color=["#94A3B8", "#1E40AF"]) # Gris vs Azul
