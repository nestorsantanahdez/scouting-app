import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- CARGA DEL MODELO REAL ---
@st.cache_resource # Para que no cargue el modelo cada vez que mueves un botón
def cargar_cerebro():
    with open('modelo_basket.pkl', 'rb') as f:
        return pickle.load(f)

data = cargar_cerebro()
modelo = data['modelo']
scaler = data['scaler']
mae_por_stat = data['mae']
stats_nombres = data['stats']

# --- DISEÑO DE LA WEB ---
st.set_page_config(page_title="Scouting Pro", page_icon="🏀")
st.title("🏀 Sistema de Proyección EBA ➔ U22")
st.info("Predicciones basadas en Random Forest Regressor entrenado con tus datos históricos.")

# --- ENTRADA DE DATOS ---
st.sidebar.header("📊 Estadísticas EBA")
inputs = []
for stat in stats_nombres:
    val = st.sidebar.number_input(f"{stat} (Media EBA)", value=10.0, step=0.1)
    inputs.append(val)

# --- PROCESAMIENTO ---
if st.sidebar.button("CALCULAR PROYECCIÓN"):
    X_input = np.array([inputs])
    
    # Aplicar escalado si el modelo original lo usaba
    if scaler:
        X_input = scaler.transform(X_input)
    
    # Predicción real
    prediccion = modelo.predict(X_input)[0]

    # --- MOSTRAR RESULTADOS ---
    st.subheader("🎯 Resultados Proyectados en U22")
    cols = st.columns(2)
    
    for i, stat in enumerate(stats_nombres):
        val_pred = max(0, prediccion[i])
        error = mae_por_stat[i]
        
        with cols[i % 2]:
            st.metric(label=stat, value=f"{val_pred:.2f}")
            st.write(f"Confianza: **[{max(0, val_pred-error):.1f} a {val_pred+error:.1f}]**")
            st.progress(min(1.0, val_pred / (val_pred + error + 0.1))) # Barra visual de progreso

    st.success("Informe generado con éxito.")
else:
    st.warning("Introduce los datos en la barra lateral y pulsa el botón para empezar.")
