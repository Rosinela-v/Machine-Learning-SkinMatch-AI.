# app_streamlit/app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import io

# ============================================================
#       CONFIGURACIÓN INICIAL — SkinMatch AI
# ============================================================
st.set_page_config(page_title="SkinMatch AI", page_icon="🌸", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#D16BA5; font-weight:800; margin-bottom:2px;'>
        🌸 SkinMatch AI
    </h1>
    <p style='text-align:center; font-size:16px; color:#555; margin-top:0;'>
        Tu asistente tierno para analizar cosméticos según tipo de piel
    </p>
""", unsafe_allow_html=True)

# ============================================================
#   RUTA DEL MODELO MULTILABEL (un solo RF multioutput)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
RUTA_MODELO_MULTI = os.path.join(MODELS_DIR, "pipeline_mejor_multilabel.pkl")

@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load(RUTA_MODELO_MULTI)
    except Exception as e:
        st.error(f"No pude cargar el modelo multilabel en {RUTA_MODELO_MULTI}. Error: {e}")
        return None

modelo_multilabel = cargar_modelo()

# Orden correcto de tus etiquetas multilabel
TIPOS_PIEL = ["Sensitive", "Oily", "Dry", "Combination", "Normal"]

# ============================================================
#   LIMPIEZA + FEATURES EXACTAS DEL NOTEBOOK 02
#   (LAS FUNCIONES ESTÁN DECLARADAS AQUÍ PARA EVITAR NAMEERROR)
# ============================================================
def limpiar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záéíóúñ0-9,()\- ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def traducir(texto):
    t = limpiar(texto)
    dicc = {
        r"\bagua\b": "water",
        r"\bglicerina\b": "glycerin",
        r"\bácido hialurónico\b": "hyaluronic acid",
        r"\bacido hialuronico\b": "hyaluronic acid",
        r"\bniacinamida\b": "niacinamide",
        r"\baloe vera\b": "aloe barbadensis leaf extract",
        r"\bfragancia\b": "fragrance",
        r"\bperfume\b": "fragrance"
    }
    for pat, rep in dicc.items():
        t = re.sub(pat, rep, t)
    return t

# Listas dermatológicas
irritantes = ["salicylic", "glycolic", "benzoyl", "alcohol", "menthol", "eucalyptus", "retinol"]
emolientes = ["glycerin", "squalane", "petrolatum", "caprylic", "cetyl", "stearyl", "shea", "lanolin"]
aceites = [" oil ", "olea europaea", "jojoba", "coconut", "ricinus"]
calmantes = ["aloe", "panthenol", "bisabolol", "centella", "niacinamide", "hyaluronic"]

def crear_features(texto):
    """
    Genera las 12 columnas que espera el pipeline (igual que en Notebook 02).
    Devuelve un DataFrame de 1 fila para compatibilidad directa con model.predict.
    """
    texto = traducir(texto)
    cantidad = len(texto.split(",")) if texto.strip() != "" else 0
    longitud = len(texto)

    cnt_irrit = sum([texto.count(w) for w in irritantes])
    cnt_emol = sum([texto.count(w) for w in emolientes])
    cnt_acei = sum([texto.count(w) for w in aceites])
    cnt_calm = sum([texto.count(w) for w in calmantes])

    ratio_irrit = cnt_irrit / (cantidad + 1)
    ratio_calm = cnt_calm / (cantidad + 1)

    return pd.DataFrame({
        "ingredientes_limpios": [texto],
        "cantidad_ingredientes": [cantidad],
        "longitud_texto": [longitud],
        "contiene_alcohol": [1 if "alcohol" in texto else 0],
        "contiene_fragancia": [1 if "fragrance" in texto else 0],
        "contiene_acido": [1 if "acid" in texto else 0],
        "cnt_irritantes": [cnt_irrit],
        "cnt_emolientes": [cnt_emol],
        "cnt_aceites": [cnt_acei],
        "cnt_calmantes": [cnt_calm],
        "ratio_irritantes": [ratio_irrit],
        "ratio_calmantes": [ratio_calm]
    })

# ============================================================
#    PESTAÑAS: Analizar / Subir archivo / Guía / Acerca
# ============================================================
tabs = st.tabs(["Analizar (manual)", "Subir archivo (CSV)", "Guía rápida", "Acerca"])

# -------------------------
# PESTAÑA 1: Analizar manual
# -------------------------
with tabs[0]:
    st.subheader("🧴 Ingresa los ingredientes (manual)")
    ingredientes = st.text_area(
        "Pégalos aquí (pueden estar en español o inglés):",
        placeholder="Ej: Aqua, Glycerin, Niacinamide, Hyaluronic Acid...",
        height=180
    )

    col1, col2 = st.columns([1,1])
    with col1:
        boton_analizar = st.button("✨ Analizar con SkinMatch AI")
    with col2:
        mostrar_prob = st.checkbox("Mostrar probabilidades detalladas", value=True)

    if boton_analizar:
        if ingredientes.strip() == "":
            st.warning("Por favor ingresa ingredientes.")
        else:
            entrada = crear_features(ingredientes)

            # ------------------------------
            #   PREDICCIÓN MULTILABEL
            # ------------------------------
            if modelo_multilabel is None:
                st.error("El modelo multilabel no está disponible. Comprueba models/pipeline_mejor_multilabel.pkl")
            else:
                try:
                    proba_multi = modelo_multilabel.predict_proba(entrada)
                    # manejar ambos formatos:
                    if isinstance(proba_multi, list):
                        prob_arr = np.array([p[:,1] for p in proba_multi]).reshape(len(proba_multi),)
                    else:
                        # predict_proba puede devolver (n_samples, n_labels) o (n_samples, n_labels, 2) según versión
                        if proba_multi.ndim == 2 and proba_multi.shape[0] == 1 and proba_multi.shape[1] == len(TIPOS_PIEL):
                            prob_arr = proba_multi[0]
                        else:
                            # forma habitual: (n_labels, n_samples, 2) no esperada; intentar convertir
                            prob_arr = np.array([p[0,1] if p.ndim==2 else p for p in proba_multi]).reshape(len(proba_multi),)

                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    prob_arr = None

                if prob_arr is not None:
                    df_res = pd.DataFrame({
                        "Tipo de piel": TIPOS_PIEL,
                        "Probabilidad": prob_arr
                    }).sort_values("Probabilidad", ascending=False)

                    st.subheader("📊 Probabilidades estimadas por SkinMatch AI")
                    if mostrar_prob:
                        st.dataframe(df_res.reset_index(drop=True))
                    else:
                        st.write("Probabilidades ocultas por solicitud del usuario.")

                    mejor = df_res.iloc[0]["Tipo de piel"]
                    st.success(f"🌟 Este producto es más adecuado para piel **{mejor}**.")

# -------------------------
# PESTAÑA 2: Subir archivo CSV
# -------------------------
with tabs[1]:
    st.subheader("📁 Subir CSV con columna 'Ingredients'")
    st.write("Sube un CSV con una columna llamada `Ingredients`. El sistema procesará cada fila y devolverá probabilidades y un CSV descargable.")

    uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])
    usar_default = st.checkbox("Usar CSV procesado del proyecto (data/processed/cosmetics_processed.csv) si existe", value=True)

    df_input = None
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"No pude leer el CSV subido: {e}")
            df_input = None
    else:
        # intentar ruta por defecto
        ruta_def = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "cosmetics_processed.csv")
        if usar_default and os.path.exists(ruta_def):
            try:
                df_input = pd.read_csv(ruta_def)
                st.info(f"Usando archivo por defecto: {ruta_def}")
            except Exception as e:
                st.error(f"No pude leer el CSV por defecto: {e}")
                df_input = None

    if df_input is not None:
        st.write("Vista previa del CSV:")
        st.dataframe(df_input.head())

        if "Ingredients" not in df_input.columns:
            st.error("El CSV debe tener una columna llamada 'Ingredients'.")
        else:
            if st.button("Procesar todo el CSV"):
                # Construir features para cada fila (lista de dicts -> DataFrame)
                textos = df_input["Ingredients"].astype(str).tolist()

                # crear lista de filas de features (uso list comprehension)
                filas = [crear_features(t).iloc[0].to_dict() for t in textos]
                X_batch = pd.DataFrame(filas)

                if modelo_multilabel is None:
                    st.error("El modelo multilabel no está disponible. No se puede predecir.")
                else:
                    try:
                        proba_multi = modelo_multilabel.predict_proba(X_batch)
                        # Normalizar salida proba a (n_samples, n_labels)
                        if isinstance(proba_multi, list):
                            # cada elemento p es (n_samples, 2) -> p[:,1]
                            proba_arr = np.vstack([p[:,1] for p in proba_multi]).T
                        else:
                            # Si ya viene en forma (n_samples, n_labels) lo tomamos
                            # También hay casos (n_samples, n_labels, 2) -> entonces coger [:,:,1]
                            if proba_multi.ndim == 3 and proba_multi.shape[2] == 2:
                                proba_arr = proba_multi[:,:,1]
                            else:
                                proba_arr = proba_multi

                        # Crear DataFrame de salida
                        df_probs = pd.DataFrame(proba_arr, columns=TIPOS_PIEL)
                        df_out = pd.concat([df_input.reset_index(drop=True), df_probs.reset_index(drop=True)], axis=1)

                        st.success("Predicción completada. Aquí las primeras filas:")
                        st.dataframe(df_out.head())

                        # Botón para descargar CSV resultante
                        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Descargar resultados (CSV)", data=csv_bytes, file_name="predicciones_cosmetics.csv", mime="text/csv")

                    except Exception as e:
                        st.error(f"Ocurrió un error al predecir en lote: {e}")

# -------------------------
# PESTAÑA 3: Guía rápida
# -------------------------
with tabs[2]:
    st.subheader("📘 Guía rápida: ¿Qué significa cada tipo de piel?")
    st.markdown("""
    ### 🩷 Sensitive
    Reacciona fácilmente a fragancias, alcoholes y ácidos. Necesita ingredientes calmantes.

    ### 💧 Dry
    Le falta hidratación. Prefiere emolientes y humectantes.

    ### ✨ Normal
    Equilibrada, tolera la mayoría de ingredientes.

    ### 🌿 Combination
    Zonas secas + zonas grasas. Requiere fórmulas equilibradas.

    ### 🛢 Oily
    Produce más sebo. Prefiere geles ligeros y evita aceites pesados.
    """)

    st.markdown("---")
    st.subheader("🩺 Consejos dermatológicos rápidos (educativo)")
    st.write("- Haz siempre prueba de parche si tu piel es sensible.")
    st.write("- Evita productos con alcohol denat y fragancias si tienes sensibilidad.")
    st.write("- Para piel seca busca 'glycerin', 'hyaluronic', 'panthenol' en la lista de ingredientes.")

# -------------------------
# PESTAÑA 4: Acerca
# -------------------------
with tabs[3]:
    st.subheader("ℹ️ Acerca de SkinMatch AI")
    st.write("Esta aplicación usa un pipeline entrenado sobre datos de ingredientes para estimar qué tipos de piel son más compatibles con un producto.")
    st.write("Versiones del proyecto, modelos y CSV de ejemplo se esperan en la carpeta `models/` y `data/processed/` del repo.")
    st.write("🔒 Disclaimer: Es una herramienta educativa, no sustituye una consulta dermatológica profesional.")
    st.markdown("**Contacto / notas:** Si tienes problemas con el modelo (fichero faltante), coloca `pipeline_mejor_multilabel.pkl` en la carpeta `models` junto al repo.")

# ============================================================
# PIE / AYUDAS
# ============================================================
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("SkinMatch AI — App para soporte académico. No es diagnóstico médico.")
st.markdown("**Desarrollado por:** Rosinela Vega")