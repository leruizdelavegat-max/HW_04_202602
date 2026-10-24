"""Estimador en vivo (pestaña 4 de la app): archivo subido → imagen → OCR → anonimización → limpieza → predicción.
Separado de Streamlit para poder probarlo sin la interfaz.

Privacidad: la imagen original solo vive en memoria y en un directorio temporal que se borra al terminar el OCR.
Nada del CV se escribe en logs ni en mensajes de error."""
import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import anonimizar, config, modelos, ocr, reglas, texto

LADO_MAX = 2200                 # px: las fotos de móvil se reducen (PopResume se renderiza a ~1275 × 1650)
PIXELES_MAX = 60_000_000        # más que esto se rechaza (protección frente a imágenes gigantes)


class ArchivoInvalido(ValueError):
    """El archivo subido no se puede convertir en una imagen de CV. El mensaje es apto para el usuario."""


def cargar_imagen(datos, nombre):
    """Bytes subidos → imagen RGB de PIL. Un PDF se convierte rasterizando su primera página a ~200 dpi.
    Corrige la orientación EXIF de las fotos y limita el tamaño."""
    from PIL import Image, ImageOps
    if not datos:
        raise ArchivoInvalido("El archivo está vacío.")
    if nombre.lower().endswith(".pdf"):
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(datos)
            if len(pdf) == 0:
                raise ArchivoInvalido("El PDF no tiene páginas.")
            img = pdf[0].render(scale=200 / 72).to_pil()
        except ArchivoInvalido:
            raise
        except Exception:
            raise ArchivoInvalido("No se pudo abrir el PDF (¿está dañado o protegido con contraseña?).") from None
    else:
        try:
            img = Image.open(io.BytesIO(datos))
            if img.width * img.height > PIXELES_MAX:
                raise ArchivoInvalido("La imagen es demasiado grande. Prueba con una de menos de 60 megapíxeles.")
            img = ImageOps.exif_transpose(img)
        except ArchivoInvalido:
            raise
        except Exception:
            raise ArchivoInvalido("El archivo no es una imagen válida (usa PNG, JPG o PDF).") from None
    img = img.convert("RGB")
    if max(img.size) > LADO_MAX:
        img.thumbnail((LADO_MAX, LADO_MAX))
    return img


def a_imagen(datos, nombre):
    """Compatibilidad: guarda lo subido como PNG en un directorio temporal y devuelve la ruta."""
    destino = Path(tempfile.mkdtemp()) / "cv.png"
    cargar_imagen(datos, nombre).save(destino)
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


def archivos_de_modelos(carpeta=config.MODELOS):
    """{modelo: ¿existe su archivo?}, para avisar qué falta sin cargar nada."""
    return {"mediana": (carpeta / "mediana.json").exists(),
            "tfidf_ridge": (carpeta / "tfidf_ridge.joblib").exists(),
            "embeddings_ridge": (carpeta / "embeddings_ridge.joblib").exists(),
            "regla_fechas": (carpeta / "regla.json").exists(),
            "bert": (carpeta / "bert_regresor" / "model.safetensors").exists()}


@dataclass
class Lectura:
    lineas: list                    # líneas de OCR en orden de lectura (texto, confianza, caja)
    personales: list                # True = la línea es un dato personal
    texto_modelo: str               # lo que ven los modelos: sin líneas personales, limpio y enmascarado
    texto_visible: str              # para mostrar: líneas personales sustituidas por una etiqueta
    imagen: object                  # imagen protegida (PIL): solo cajas no personales
    dos_columnas: bool
    mascaras: dict = field(default_factory=dict)

    @property
    def n_personales(self):
        return int(sum(self.personales))


def procesar(motor, datos, nombre, umbral=0.0):
    """Pipeline completo de un CV subido. Lanza ArchivoInvalido si el archivo no sirve."""
    img = cargar_imagen(datos, nombre)
    with tempfile.TemporaryDirectory() as tmp:          # el PNG temporal se borra al salir del bloque
        ruta = Path(tmp) / "cv.png"
        img.save(ruta)
        lineas, _, ancho = ocr.leer_imagen(motor, ruta)
    ordenadas, dos_col = ocr.ordenar_lectura(lineas, ancho)
    personales = anonimizar.clasificar(ordenadas)
    visibles = [l for l, p in zip(ordenadas, personales) if not p]
    limpio, conteo = texto.limpiar(visibles, umbral)
    return Lectura(lineas=ordenadas, personales=personales, texto_modelo=limpio,
                   texto_visible=anonimizar.texto_visible(ordenadas, personales),
                   imagen=anonimizar.imagen_protegida(img, ordenadas, personales),
                   dos_columnas=dos_col, mascaras=conteo)


def leer(motor, ruta, umbral=0.0):
    """Compatibilidad: OCR + orden de lectura + limpieza sin las líneas personales. Devuelve
    (líneas ordenadas, texto limpio, conteo de máscaras)."""
    lineas, _, ancho = ocr.leer_imagen(motor, ruta)
    ordenadas, _ = ocr.ordenar_lectura(lineas, ancho)
    personales = anonimizar.clasificar(ordenadas)
    limpio, conteo = texto.limpiar([l for l, p in zip(ordenadas, personales) if not p], umbral)
    return ordenadas, limpio, conteo


def predecir(predictores, texto_limpio):
    """{modelo: años}. Si un modelo falla, se omite y se informa en la clave '_errores' (sin texto del CV)."""
    out, errores = {}, []
    for m, f in predictores.items():
        try:
            out[m] = float(f([texto_limpio])[0])
        except Exception as e:
            errores.append(f"{m}: {type(e).__name__}")
    if errores:
        out["_errores"] = errores
    return out
