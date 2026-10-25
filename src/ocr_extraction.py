"""Etapa 1 · OCR con PaddleOCR 3.x y orden de lectura (una o dos columnas).

Punto de entrada con el nombre que pide el enunciado; la implementación está en `ocr.py` y es la misma que
ejecuta `notebooks/01_muestreo_ocr.ipynb` (en Colab con GPU T4).

Uso:
    python -m src.ocr_extraction            (GPU si hay; si no, CPU)
Lee `data/raw/images/*.png` y escribe un JSON por CV en `data/ocr/` (texto, confianza y caja por línea, ya en
orden de lectura). Es reanudable: los JSON existentes no se vuelven a procesar.
"""
from . import config
from .ocr import cargar_json, crear_ocr, detectar_canal, leer_imagen, ordenar_lectura, procesar_lote, texto_de

IMAGENES = config.DATA / "raw" / "images"
OCR_JSON = config.DATA / "ocr"


def main(device=None):
    if device is None:
        try:
            import paddle
            device = "gpu:0" if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() else "cpu"
        except Exception:
            device = "cpu"
    rutas = sorted(IMAGENES.glob("*.png"))
    hechos, fallidos = procesar_lote(crear_ocr(device=device), rutas, OCR_JSON)
    print(f"{len(rutas)} imágenes · nuevos {hechos} · fallidos {fallidos} · JSON en {OCR_JSON}")


__all__ = ["IMAGENES", "OCR_JSON", "cargar_json", "crear_ocr", "detectar_canal", "leer_imagen", "main",
           "ordenar_lectura", "procesar_lote", "texto_de"]

if __name__ == "__main__":
    main()
