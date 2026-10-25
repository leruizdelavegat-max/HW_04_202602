"""Etapa 1 · Muestreo de PopResume y descarga de las imágenes sin foto.

Punto de entrada con el nombre que pide el enunciado; la implementación está en `data.py` y es la misma que
ejecuta `notebooks/01_muestreo_ocr.ipynb`.

Uso (requiere aceptar las condiciones de PopResume en Hugging Face y un token de lectura):
    set HF_TOKEN=...            (o huggingface-cli login)
    python -m src.download_data
"""
import os

from . import config
from .data import cargar_metadatos, descargar_imagenes, muestrear, shards_de_imagenes

IMAGENES = config.DATA / "raw" / "images"


def main(token=None):
    token = token or os.environ.get("HF_TOKEN")
    meta = cargar_metadatos()
    muestra = muestrear(meta)                                   # 800 por ocupación, semilla 42
    muestra[["id", "job"]].to_csv(config.DATA / "sample_ids.csv", index=False)
    # El texto de referencia solo sirve para medir el OCR (CER/WER): se guarda aparte y fuera del repo
    (config.DATA / "raw").mkdir(parents=True, exist_ok=True)
    muestra[["id", "text"]].to_parquet(config.DATA / "raw" / "reference_text.parquet", index=False)
    descargar_imagenes(muestra["id"], IMAGENES, token=token)   # solo has_photo == False
    print("PNG en disco:", len(list(IMAGENES.glob("*.png"))))


__all__ = ["IMAGENES", "cargar_metadatos", "descargar_imagenes", "main", "muestrear", "shards_de_imagenes"]

if __name__ == "__main__":
    main()
