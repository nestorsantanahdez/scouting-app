import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# --- 1. ESTÉTICA ELEGANTE Y RESPONSIVA ---
st.set_page_config(page_title="Scouting Pro 🏀", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #F1F5F9; }
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px; border-radius: 12px; border: 1px solid #E2E8F0;
        border-top: 5px solid #1E40AF; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important; font-weight: 700 !important; color: #0F172A !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; color: #1E40AF !important; font-weight: 800;
    }
    .stButton>button {
        background-color: #1E40AF !important; color: #FFFFFF !important;
        height: 3.5em !important; font-weight: bold !important; border-radius: 12px !important;
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

# --- 3. DICCIONARIO DE ROLES (LEYENDA) ---
diccionario_roles = {
    "⭐ Referente Ofensivo": "Jugador con alta carga de minutos y puntos. Capaz de liderar el ataque en U22.",
    "🎯 Microondas / 6º Hombre": "Alta eficiencia anotadora en periodos cortos. Ideal para revolucionar partidos.",
    "💪 Especialista en Pintura": "Dominio del rebote total. Asegura segundas oportunidades y control defensivo.",
    "🧠 Generador de Juego": "Base o escolta con alta visión. Facilita el juego y las canastas de sus compañeros.",
    "🛡️ Especialista Defensivo": "Capacidad para robar balones y presionar. Clave para romper el ritmo rival.",
    "⏳ Rotación Profunda": "Jugador para minutos específicos. Aportación de energía en ventanas cortas.",
    "🔄 Jugador de Complemento": "Perfil equilibrado que aporta solidez a la segunda unidad sin errores graves."
}

# --- 4. INTERFAZ Y ESCÁNER ---
st.title("🏀 Inteligencia de Scouting: EBA ➔ U22")

with st.expander("📝 MODO ESCÁNER (PC / EXCEL)", expanded=False):
    raw_data = st.text_input("Pega la línea estadística aquí:")
    if raw_data:
        clean_text = raw_data.replace('%', '').replace(',', '.')
        parts = clean_text.split()
        if len(parts) >= 18:
            try:
                def f(x):
                    if ":" in str(x):
                        p = str(x).split(":"); return float(p[0]) + float(p[1])/60
                    return float(x)
                # Mapeo: MIN(1), PT(2), RT(9), AS(10), BR(11), BP(12), TF(13), FR(17)
                st.session_state.inputs = [f(parts[1]), f(parts[2]), f(parts[9]), f(parts[10]), 
                                          f(parts[11]), f(parts[12]), f(parts[13]), f(parts[17])]
                st.success("✅ Datos cargados correctamente.")
            except: st.error("Error en formato.")

# --- 5. PANEL LATERAL ---
st.sidebar.header("📊 DATOS EBA")
final_inputs = []
for i, stat in enumerate(stats_nombres):
    default_val = st.session_state.inputs[i] if 'inputs' in st.session_state else 5.0
    val = st.sidebar.number_input(f"{stat}", value=float(default_val), step=0.1, key=f"s_{stat}")
    final_inputs.append(val)

# --- 6. CÁLCULO Y RESULTADOS ---
if st.button("🚀 GENERAR INFORME COMPLETO", use_container_width=True):
    X_input = np.array([final_inputs])
    if scaler: X_input = scaler.transform(X_input)
    
    pred = modelo.predict(X_input)[0]

    st.subheader("🎯 PROYECCIÓN EN U22")
    cols = st.columns(2) 
    for i, stat in enumerate(stats_nombres):
        val_p = max(0, pred[i])
        err = mae_por_stat[i]
        with cols[i % 2]:
            label_display = stat.replace("Rebotes total", "REBOTES").replace("Faltas recibidas", "F. RECIBIDAS").replace("Tapones favor", "TAPONES")
            st.metric(label=label_display, value=f"{val_p:.1f}")
            st.caption(f"Confianza: ±{err:.1f}")

    # GRÁFICO
    st.subheader("📈 COMPARATIVA EBA vs U22")
    chart_df = pd.DataFrame({'Stat': stats_nombres, 'EBA': final_inputs, 'U22': pred}).set_index('Stat')
    st.bar_chart(chart_df, color=["#94A3B8", "#1E40AF"])

    # --- 7. ANÁLISIS DE ROL ---
    st.divider()
    st.subheader("📋 Rol Sugerido")
    
    # Lógica de detección (basada en el array 'pred')
    # 0:Min, 1:Pts, 2:Reb, 3:Ast, 4:Rec, 5:Per, 6:Tap, 7:F.Rec
    roles_detectados = []
    if pred[1] > 12 and pred[0] > 20: roles_detectados.append("⭐ Referente Ofensivo")
    elif pred[1] > 8: roles_detectados.append("🎯 Microondas / 6º Hombre")
    
    if pred[2] > 6: roles_detectados.append("💪 Especialista en Pintura")
    if pred[3] > 3.5: roles_detectados.append("🧠 Generador de Juego")
    if pred[4] > 1.2: roles_detectados.append("🛡️ Especialista Defensivo")
    if pred[0] < 10: roles_detectados.append("⏳ Rotación Profunda")
    
    if not roles_detectados: roles_detectados.append("🔄 Jugador de Complemento")

    # Mostrar botones de rol con tooltip (ayuda al pinchar/pasar ratón)
    c1, c2 = st.columns(2)
    for i, rol in enumerate(roles_detectados):
        with [c1, c2][i % 2]:
            st.button(f"{rol}", help=diccionario_roles[rol], disabled=False, use_container_width=True)
            st.caption("*(Pincha o pasa el ratón para ver descripción)*")

# --- 8. LEYENDA GENERAL (AL FINAL) ---
st.write("")
with st.expander("📖 VER LEYENDA DE TODOS LOS ROLES"):
    for r, d in diccionario_roles.items():
        st.markdown(f"**{r}:** {d}")
