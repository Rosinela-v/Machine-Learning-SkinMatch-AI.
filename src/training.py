"""
training.py — Entrenamiento del modelo multilabel SkinSoft AI
--------------------------------------------------------------

Este script:
1. Carga el dataset procesado desde /data/processed/
2. Aplica el Feature Engineering dermatológico
3. Entrena un modelo RandomForest Multilabel (MultiOutputClassifier)
4. Guarda el pipeline completo en /models/pipeline_mejor_multilabel.pkl

NO utiliza split train/test.
El objetivo es entrenar un modelo final para producción.
"""

import os
import pandas as pd
import numpy as np
import re
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------------------------------------------
# RUTAS
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "cosmetics_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------------------------------------
# CARGAR DATOS
# -----------------------------------------------------------
print("Cargando dataset procesado...")
df = pd.read_csv(DATA_PROCESSED)

# Etiquetas multilabel
columnas_target = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

y = df[columnas_target]
X = df.drop(columns=columnas_target)

# -----------------------------------------------------------
# FEATURE ENGINEERING (idéntico a Notebook 02)
# -----------------------------------------------------------

def limpiar(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z0-9,()\- ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

# Listas dermatológicas
irritantes = ["salicylic", "glycolic", "benzoyl", "alcohol", "menthol", "eucalyptus", "retinol"]
emolientes = ["glycerin", "squalane", "petrolatum", "caprylic", "cetyl", "stearyl", "shea", "lanolin"]
aceites = [" oil ", "olea europaea", "jojoba", "coconut", "ricinus"]
calmantes = ["aloe", "panthenol", "bisabolol", "centella", "niacinamide", "hyaluronic"]

def crear_features_batch(df):
    df = df.copy()
    df["ingredientes_limpios"] = df["ingredientes_limpios"].astype(str).apply(limpiar)

    df["cnt_irritantes"] = df["ingredientes_limpios"].apply(lambda x: sum([x.count(w) for w in irritantes]))
    df["cnt_emolientes"] = df["ingredientes_limpios"].apply(lambda x: sum([x.count(w) for w in emolientes]))
    df["cnt_aceites"] = df["ingredientes_limpios"].apply(lambda x: sum([x.count(w) for w in aceites]))
    df["cnt_calmantes"] = df["ingredientes_limpios"].apply(lambda x: sum([x.count(w) for w in calmantes]))

    df["ratio_irritantes"] = df["cnt_irritantes"] / (df["cantidad_ingredientes"] + 1)
    df["ratio_calmantes"] = df["cnt_calmantes"] / (df["cantidad_ingredientes"] + 1)

    return df

print("Aplicando feature engineering...")
X = crear_features_batch(X)

# -----------------------------------------------------------
# PREPROCESAMIENTO (TFIDF + SCALER)
# -----------------------------------------------------------
columnas_numericas = [
    "cantidad_ingredientes", "longitud_texto",
    "contiene_alcohol", "contiene_fragancia", "contiene_acido",
    "cnt_irritantes", "cnt_emolientes", "cnt_aceites", "cnt_calmantes",
    "ratio_irritantes", "ratio_calmantes"
]

preprocesador = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words="english"), "ingredientes_limpios"),
        ("num", StandardScaler(), columnas_numericas),
    ],
    remainder="drop"
)

# -----------------------------------------------------------
# MODELO MULTILABEL RF
# -----------------------------------------------------------
print("Entrenando modelo RandomForest Multilabel...")

modelo_rf_multi = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
)

pipeline = Pipeline([
    ("preprocesador", preprocesador),
    ("modelo", modelo_rf_multi)
])

pipeline.fit(X, y)

# -----------------------------------------------------------
# GUARDAR MODELO
# -----------------------------------------------------------
ruta_guardado = os.path.join(MODELS_DIR, "pipeline_mejor_multilabel.pkl")
joblib.dump(pipeline, ruta_guardado)
