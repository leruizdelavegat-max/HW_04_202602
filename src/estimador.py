"""Estimador en vivo (pestaña 4 de la app): archivo subido → imagen → OCR → limpieza → predicción de cada modelo.
Separado de Streamlit para poder probarlo sin la interfaz."""
import io
import json
import tempfile
from pathlib import Path

import numpy as np

from . import config, modelos, ocr, reglas, texto


def a_imagen(datos, nombre):
    """Guarda lo subido como PNG. Un PDF se convierte rasterizando su primera página a ~200 dpi."""
    destino = Path(tempfile.mkdtemp()) / "cv.png"
    if nombre.lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(datos)
        pdf[0].render(scale=200 / 72).to_pil().convert("RGB").save(destino)
    else:
        from PIL import Image
        Image.open(io.BytesIO(datos)).convert("RGB").save(destino)
    return destino


def cargar_predictores(tope, carpeta=config.MODELOS):
    """{modelo: función(lista_de_textos) → años}. Solo incluye los modelos cuyos archivos existen."""
    import joblib
    out = {}
    if (carpeta / "mediana.json").exists():
        v = json.loads((carpeta / "mediana.json").read_text())["valor"]
        out["mediana"] = lambda ts: np.full(len(ts), v)
    if (carpeta / "tfidf_ridge.joblib").exists():
        tf = joblib.load(carpeta / "tfidf_ridge.joblib")
        out["tfidf_ridge"] = lambda ts: np.clip(tf.predict(ts), 0, tope)
    if (carpeta / "embeddings_ridge.joblib").exists():
        from sentence_transformers import SentenceTransformer
        e = joblib.load(carpeta / "embeddings_ridge.joblib")
        enc = SentenceTransformer(e["encoder"], device="cpu")
        out["embeddings_ridge"] = lambda ts: np.clip(e["ridge"].predict(modelos.embeber(ts, enc)), 0, tope)
    if (carpeta / "regla.json").exists():
        anio = json.loads((carpeta / "regla.json").read_text())["anio_actual"]
        out["regla_fechas"] = lambda ts: np.array([reglas.estimar(t, anio, tope) for t in ts])
    if (carpeta / "bert_regresor" / "config.json").exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(carpeta / "bert_regresor")
        bert = AutoModelForSequenceClassification.from_pretrained(carpeta / "bert_regresor")
        out["bert"] = lambda ts: modelos.predecir_bert(bert, tok, ts, tope, device="cpu")
    return out


def leer(motor, ruta, umbral=0.0):
    """OCR + orden de lectura + la misma limpieza del entrenamiento. El nombre de un CV subido no se conoce:
    si la primera línea parece un nombre (2–4 palabras en mayúsculas, sin dígitos), se quita como en el
    entrenamiento, donde se borraba con first_name/last_name."""
    lineas, _, ancho = ocr.leer_imagen(motor, ruta)
    ordenadas, _ = ocr.ordenar_lectura(lineas, ancho)
    limpio, conteo = texto.limpiar(ordenadas, umbral)
    primera, _, resto = limpio.partition("\n")
    if primera.isupper() and 2 <= len(primera.split()) <= 4 and not any(c.isdigit() for c in primera) \
            and primera.strip() not in ("WORK HISTORY", "EXPERIENCE", "EDUCATION"):
        limpio = resto
    return ordenadas, limpio, conteo


def predecir(predictores, texto_limpio):
    return {m: float(f([texto_limpio])[0]) for m, f in predictores.items()}
