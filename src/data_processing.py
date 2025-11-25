# src/data_processing.py

import pandas as pd
import numpy as np
import os
import re

# ================================
#   FUNCIONES DE LIMPIEZA
# ================================

def limpiar(texto):
    texto = texto.lower()
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


# Listas para feature engineering
irritantes = ["salicylic", "glycolic", "benzoyl", "alcohol",
              "menthol", "eucalyptus", "retinol"]
emolientes = ["glycerin", "squalane", "petrolatum",
              "caprylic", "cetyl", "stearyl", "shea", "lanolin"]
aceites = [" oil ", "olea europaea", "jojoba", "coconut", "ricinus"]
calmantes = ["aloe", "panthenol", "bisabolol", "centella",
             "niacinamide", "hyaluronic"]


# ================================
#   CREACIÓN DE FEATURES
# ================================
def crear_features(texto):
    texto = traducir(texto)

    cantidad = len(texto.split(","))
    longitud = len(texto)

    cnt_irrit = sum([texto.count(w) for w in irritantes])
    cnt_emol = sum([texto.count(w) for w in emolientes])
    cnt_acei = sum([texto.count(w) for w in aceites])
    cnt_calm = sum([texto.count(w) for w in calmantes])

    ratio_irrit = cnt_irrit / (cantidad + 1)
    ratio_calm = cnt_calm / (cantidad + 1)

    return pd.Series({
        "ingredientes_limpios": texto,
        "cantidad_ingredientes": cantidad,
        "longitud_texto": longitud,
        "contiene_alcohol": 1 if "alcohol" in texto else 0,
        "contiene_fragancia": 1 if "fragrance" in texto else 0,
        "contiene_acido": 1 if "acid" in texto else 0,
        "cnt_irritantes": cnt_irrit,
        "cnt_emolientes": cnt_emol,
        "cnt_aceites": cnt_acei,
        "cnt_calmantes": cnt_calm,
        "ratio_irritantes": ratio_irrit,
        "ratio_calmantes": ratio_calm
    })


# ================================
#   PROCESAMIENTO PRINCIPAL
# ================================
def procesar_datos():
    raw_path = "data/raw/cosmetics.csv"
    save_path = "data/processed/cosmetics_processed.csv"

    df = pd.read_csv(raw_path)

    df["ingredientes_limpios"] = df["Ingredients"].astype(str).apply(traducir)

    features_df = df["ingredientes_limpios"].apply(crear_features)
    df = pd.concat([df, features_df], axis=1)

    df.to_csv(save_path, index=False)
    print(f"Archivo procesado guardado en: {save_path}")


if __name__ == "__main__":
    procesar_datos()
