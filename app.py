import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Scouting Pro 🏀", layout="wide")

# --- 2. CARGA DEL MODELO (EL "CEREBRO" DE COLAB) ---
@st.cache_resource
def cargar_recursos():
    try:
        with open('modelo_basket.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

data = cargar_recursos()

if data is None:
    st.error("⚠️ No se encontró el archivo 'modelo_basket.pkl'. Súbelo a tu repositorio de GitHub.")
    st.stop()

# Extraer los objetos guardados
modelo = data['modelo']
scaler = data['scaler']
mae_por_stat = data['mae']
stats_entrenamiento = data['stats']

# --- 3. DISEÑO DE LA INTERFAZ ---
st.title("🏀 Sistema de Proyección: EBA ➔ U22")
st.markdown(f"Este modelo utiliza **Random Forest** entrenado con {len(stats_entrenamiento)} variables clave.")

# --- 4. BARRA LATERAL (ENTRADA DE DATOS) ---
st.sidebar.header("📊 Estadísticas del Jugador (EBA)")
st.sidebar.write("Introduce las medias por partido:")

inputs_usuario = []
for stat in stats_entrenamiento:
    # Creamos un campo de entrada para cada estadística que el modelo conoce
    val = st.sidebar.number_input(f"{stat}", min_value=0.0, value=5.0, step=0.1)
    inputs_usuario.append(val)

# --- 5. PROCESAMIENTO Y PREDICCIÓN ---
if st.sidebar.button("🚀 GENERAR PROYECCIÓN"):
    # Convertimos los inputs a un formato que el modelo entienda (Array 2D)
    X_input = np.array([inputs_usuario])
    
    # Verificación de seguridad antes de escalar
    n_esperado = scaler.n_features_in_ if scaler else modelo.n_features_in_
    
    if X_input.shape[1] != n_esperado:
        st.error(f"Error de dimensiones: El modelo espera {n_esperado} datos, pero recibió {X_input.shape[1]}.")
    else:
        # Aplicamos el escalador de Colab
        if scaler is not None:
            X_input = scaler.transform(X_input)
        
        # Realizamos la predicción real
        prediccion = modelo.predict(X_input)[0]

        # --- 6. VISUALIZACIÓN DE RESULTADOS ---
        st.subheader("🎯 Proyección de Rendimiento en U22")
        
        # Creamos 4 columnas para que queden 2 arriba y 2 abajo (o según el número de stats)
        cols = st.columns(4)
        
        for i, stat in enumerate(stats_entrenamiento):
            val_pred = max(0, prediccion[i]) # Evitamos negativos deportivos
            error = mae_por_stat[i]
            
            with cols[i % 4]:
                st.metric(label=stat, value=f"{val_pred:.2f}")
                st.markdown(f"**Confianza (±MAE):**")
                st.caption(f"[{max(0, val_pred-error):.1f} a {val_pred+error:.1f}]")
                st.divider()
        
        st.success("✅ Proyección calculada con éxito basándose en el histórico de jugadores.")
else:
    st.info("👈 Introduce los datos en la barra lateral y pulsa el botón para ver la proyección.")

# --- 7. PIE DE PÁGINA ---
st.sidebar.divider()
st.sidebar.caption("Modelo: Random Forest Regressor")
st.sidebar.caption("Validación: Leave-One-Out (LOOCV)")
