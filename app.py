import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- CARGA DEL MODELO REAL ---
@st.cache_resource
def cargar_cerebro():
    with open('modelo_basket.pkl', 'rb') as f:
        return pickle.load(f)

data = cargar_cerebro()
modelo = data['modelo']
scaler = data['scaler']
mae_por_stat = data['mae']
# Usamos las estadísticas con las que el modelo fue entrenado realmente
stats_entrenamiento = data['stats'] 

# --- DISEÑO DE LA WEB ---
st.set_page_config(page_title="Scouting Pro", page_icon="🏀")
st.title("🏀 Sistema de Proyección EBA ➔ U22")

# --- ENTRADA DE DATOS DINÁMICA ---
st.sidebar.header("📊 Estadísticas EBA")
inputs = []

# IMPORTANTE: Solo pedimos las variables que el modelo conoce
for stat in stats_entrenamiento:
    val = st.sidebar.number_input(f"{stat} (Media EBA)", value=10.0, step=0.1)
    inputs.append(val)

# --- PROCESAMIENTO ---
if st.sidebar.button("CALCULAR PROYECCIÓN"):
    # Convertimos a array de 2D
    X_input = np.array([inputs])
    
    # 1. Ajuste automático de dimensiones por seguridad
    if X_input.shape[1] != modelo.n_features_in_:
        st.error(f"Error de dimensiones: El modelo espera {modelo.n_features_in_} variables pero recibió {X_input.shape[1]}. Re-exporta el modelo desde Colab.")
    else:
        # 2. Aplicar escalado si existe
        if scaler is not None:
            X_input = scaler.transform(X_input)
        
        # 3. Predicción real (Multisalida)
        prediccion = modelo.predict(X_input)[0] # Extraemos la primera fila de resultados

        # --- MOSTRAR RESULTADOS ---
        st.subheader("🎯 Resultados Proyectados en U22")
        cols = st.columns(2)
        
        # Iteramos sobre las variables de SALIDA (U22)
        for i, stat in enumerate(stats_entrenamiento):
            val_pred = max(0, prediccion[i])
            error = mae_por_stat[i]
            
            with cols[i % 2]:
                st.metric(label=f"Proyección {stat}", value=f"{val_pred:.2f}")
                st.caption(f"Confianza: **[{max(0, val_pred-error):.1f} a {val_pred+error:.1f}]**")
                st.divider()
