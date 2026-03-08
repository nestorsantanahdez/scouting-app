import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# --- 1. ESTÉTICA PROFESIONAL (RESPONSIVE PARA MÓVIL Y PC) ---
st.set_page_config(page_title="Scouting Pro 🏀", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border-top: 4px solid #1e3a8a; }
    [data-testid="stMetricValue"] { color: #1e3a8a; font-size: 2.2rem !important; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #1e3a8a; color: white; font-weight: bold; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DEL CEREBRO (MODELO ENTRENADO) ---
@st.cache_resource
def cargar_recursos():
    try:
        # Usamos el nombre del último archivo que generamos con 8 variables
        with open('modelo_final_8_stats.pkl', 'rb') as f:
            return pickle.load(f)
    except: return None

data = cargar_recursos()
if not data:
    st.error("⚠️ Archivo 'modelo_final_8_stats.pkl' no detectado. Asegúrate de que el nombre coincide en GitHub.")
    st.stop()

modelo, scaler, mae_por_stat, stats_nombres = data['modelo'], data['scaler'], data['mae'], data['stats']

# --- 3. INTERFAZ PRINCIPAL ---
st.title("🏀 Scouting Inteligente: Proyección EBA ➔ U22")
st.write("Predicción de impacto mediante **Random Forest** (200 árboles).")

# --- 4. ESCÁNER DE DATOS (ORDEN: Part, MIN, PT, T2, T3, TC, TL, RO, RD, RT, AS, BR, BP, TF, TC, MT, FC, FR, VA) ---
with st.container():
    st.info("✨ **MODO ESCÁNER:** Pega la línea de estadísticas de Excel aquí abajo.")
    raw_data = st.text_input("Pegar línea estadística:", placeholder="Ej: 21 19:58 7,9 47,7% ...")
    
    if raw_data:
        # Limpieza de comas y porcentajes
        clean_text = raw_data.replace('%', '').replace(',', '.')
        parts = clean_text.split()
        
        if len(parts) >= 18:
            try:
                def f(x):
                    if ":" in str(x):
                        p = str(x).split(":")
                        return float(p[0]) + float(p[1])/60
                    return float(x)

                # MAPEADO SEGÚN TU CABECERA (Índices exactos):
                # 1:MIN, 2:PT, 9:RT, 10:AS, 11:BR, 12:BP, 13:TF, 17:FR
                mapeo_auto = [
                    f(parts[1]),  # Minutos
                    f(parts[2]),  # Puntos
                    f(parts[9]),  # Rebotes Total
                    f(parts[10]), # Asistencias
                    f(parts[11]), # Recuperaciones (BR)
                    f(parts[12]), # Perdidas (BP)
                    f(parts[13]), # Tapones Favor (TF)
                    f(parts[17])  # Faltas Recibidas (FR)
                ]
                st.session_state.inputs = mapeo_auto
                st.success("✅ ¡Línea detectada! Datos cargados en el panel de control.")
            except Exception as e:
                st.error(f"Error al procesar: {e}")

# --- 5. PANEL DE AJUSTE (MÓVIL / SIDEBAR) ---
st.sidebar.header("⚙️ Panel de Control")
final_inputs = []
for i, stat in enumerate(stats_nombres):
    val_defecto = st.session_state.inputs[i] if 'inputs' in st.session_state else 5.0
    val = st.sidebar.number_input(f"{stat}", value=float(val_defecto), step=0.1, key=f"side_{stat}")
    final_inputs.append(val)

# --- 6. CÁLCULO Y RESULTADOS ---
if st.button("🚀 GENERAR INFORME DE PROYECCIÓN"):
    X_in = np.array([final_inputs])
    if scaler: X_in = scaler.transform(X_in)
    
    # PREDICCIÓN CON CORRECCIÓN DE ÍNDICE [0]
    pred_raw = modelo.predict(X_in)
    pred = pred_raw[0] # IMPORTANTE: Extraemos la primera fila para evitar el ValueError

    # MÉTRICAS EN TARJETAS
    st.divider()
    st.subheader("🎯 Proyección de Rendimiento en U22")
    cols = st.columns(4)
    for i, stat in enumerate(stats_nombres):
        val_p = max(0, pred[i])
        err = mae_por_stat[i]
        with cols[i % 4]:
            st.metric(label=stat.upper(), value=f"{val_p:.1f}")
            st.caption(f"Confianza: **[{max(0, val_p-err):.1f} a {val_p+err:.1f}]**")
            st.divider()

    # GRÁFICO COMPARATIVO
    st.subheader("📈 Comparativa Visual: EBA vs U22")
    chart_df = pd.DataFrame({
        'Estadística': stats_nombres,
        'EBA (Entrada)': final_inputs,
        'U22 (Proyectado)': pred
    }).set_index('Estadística')
    st.bar_chart(chart_df)

    # RESUMEN DE PERFIL
    st.subheader("🔍 Análisis de Perfil Proyectado")
    if pred[1] > 12: st.write("- **Perfil:** Gran Anotador en U22.")
    if pred[2] > 6: st.write("- **Perfil:** Dominio del Rebote.")
    if pred[3] > 3: st.write("- **Perfil:** Generador de juego (Playmaker).")
