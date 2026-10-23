"""OCR con PaddleOCR 3.x + reconstrucción del orden de lectura (incluye CVs a dos columnas)."""
import json
import time
from pathlib import Path

import numpy as np

from . import config


_MOTORES = {}


def crear_ocr(device=None):
    """PaddleOCR con motor PaddlePaddle (el que pide el issue). `device`: 'gpu:0' en Colab, 'cpu' local.

    PaddleX solo se inicializa una vez por proceso: el motor se guarda y se reutiliza, así volver a
    ejecutar la celda no falla con «PDX has already been initialized»."""
    if device in _MOTORES:
        return _MOTORES[device]
    import logging
    import os
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR
    for nombre in ("paddlex", "paddleocr", "ppocr"):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    extra = {}
    if device == "cpu":
        # PaddlePaddle 3.3 + oneDNN falla en CPU ("ConvertPirAttribute2RuntimeAttribute not support")
        extra["enable_mkldnn"] = False
    try:
        motor = PaddleOCR(
            text_detection_model_name=config.OCR_DET,
            text_recognition_model_name=config.OCR_REC,
            device=device,
            **config.OCR_OPCIONES,
            **extra,
        )
    except RuntimeError as e:
        if "already been initialized" in str(e):
            raise RuntimeError(
                "Un intento anterior de crear PaddleOCR falló en esta misma sesión y PaddleX no se puede "
                "reinicializar. Reinicia la sesión (Entorno de ejecución → Reiniciar sesión) y vuelve a "
                "ejecutar todo: lo ya descargado se reutiliza. Si falla otra vez, el error que aparezca "
                "primero es el verdadero.") from e
        raise
    _MOTORES[device] = motor
    return motor


def leer_imagen(ocr, ruta):
    """Devuelve (lineas, alto, ancho). Cada línea: texto, confianza y caja [x0, y0, x1, y1]."""
    r = ocr.predict(str(ruta))[0]
    cajas = np.asarray(r["rec_boxes"]).reshape(-1, 4).tolist()
    lineas = [
        {"text": t, "conf": round(float(c), 4), "box": [int(v) for v in b]}
        for t, c, b in zip(r["rec_texts"], r["rec_scores"], cajas)
    ]
    alto, ancho = _tamano(ruta)
    return lineas, alto, ancho


def _tamano(ruta):
    from PIL import Image
    with Image.open(ruta) as im:
        return im.height, im.width


# ── Orden de lectura ────────────────────────────────────────────────────────────

def detectar_canal(lineas, ancho, min_lado=0.2, max_cruce=0.05):
    """Busca un canal vertical (gutter) entre 25 % y 75 % del ancho que casi ninguna caja cruce
    y que tenga suficientes líneas a cada lado. Devuelve su x o None si la página es de una columna."""
    if len(lineas) < 8:
        return None
    cajas = np.array([l["box"] for l in lineas], dtype=float)
    mejor, mejor_cruce = None, 1.0
    for x in np.linspace(0.25 * ancho, 0.75 * ancho, 101):
        cruce = ((cajas[:, 0] < x) & (cajas[:, 2] > x)).mean()
        izq = (cajas[:, 2] <= x).mean()
        der = (cajas[:, 0] >= x).mean()
        if izq >= min_lado and der >= min_lado and cruce < mejor_cruce:
            mejor, mejor_cruce = x, cruce
    return mejor if mejor is not None and mejor_cruce <= max_cruce else None


def _filas(lineas):
    """Agrupa en filas las líneas cuyos centros verticales están a menos de media altura
    mediana; dentro de cada fila, izquierda → derecha."""
    if not lineas:
        return []
    alto_med = np.median([l["box"][3] - l["box"][1] for l in lineas]) or 10
    orden = sorted(lineas, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    filas, actual, y_ref = [], [orden[0]], (orden[0]["box"][1] + orden[0]["box"][3]) / 2
    for l in orden[1:]:
        yc = (l["box"][1] + l["box"][3]) / 2
        if abs(yc - y_ref) <= 0.5 * alto_med:
            actual.append(l)
        else:
            filas.append(sorted(actual, key=lambda z: z["box"][0]))
            actual, y_ref = [l], yc
    filas.append(sorted(actual, key=lambda z: z["box"][0]))
    return [l for fila in filas for l in fila]


def ordenar_lectura(lineas, ancho):
    """Una columna: arriba→abajo, izquierda→derecha. Dos columnas: las líneas que cruzan el
    canal (títulos a todo el ancho) parten la página en franjas; en cada franja se lee la
    columna izquierda completa y después la derecha."""
    canal = detectar_canal(lineas, ancho)
    if canal is None:
        return _filas(lineas), False
    resultado, franja = [], []

    def vaciar():
        izq = [l for l in franja if (l["box"][0] + l["box"][2]) / 2 < canal]
        der = [l for l in franja if (l["box"][0] + l["box"][2]) / 2 >= canal]
        resultado.extend(_filas(izq) + _filas(der))
        franja.clear()

    for l in sorted(lineas, key=lambda z: z["box"][1]):
        if l["box"][0] < canal < l["box"][2]:     # línea a todo el ancho
            vaciar()
            resultado.append(l)
        else:
            franja.append(l)
    vaciar()
    return resultado, True


def texto_de(lineas, umbral=0.0):
    """Une las líneas (ya ordenadas) con saltos de línea, descartando las de baja confianza."""
    return "\n".join(l["text"] for l in lineas if l["conf"] >= umbral)


# ── Lote reanudable ─────────────────────────────────────────────────────────────

def procesar_lote(ocr, rutas, destino_json, log=print, cada=100):
    """OCR de muchas imágenes. Escribe un JSON por CV (texto, confianza y caja por línea, ya en
    orden de lectura) y salta los que ya existen, así una sesión de Colab cortada se retoma."""
    destino_json = Path(destino_json)
    destino_json.mkdir(parents=True, exist_ok=True)
    hechos = fallidos = 0
    t_total = time.perf_counter()
    for k, ruta in enumerate(rutas, 1):
        ruta = Path(ruta)
        salida = destino_json / f"{ruta.stem}.json"
        if salida.exists():
            continue
        t0 = time.perf_counter()
        try:
            lineas, alto, ancho = leer_imagen(ocr, ruta)
            ordenadas, dos_col = ordenar_lectura(lineas, ancho)
            registro = {"id": ruta.stem, "width": ancho, "height": alto, "two_columns": dos_col,
                        "seconds": round(time.perf_counter() - t0, 3), "error": None,
                        "lines": ordenadas}
            hechos += 1
        except Exception as e:                       # una imagen rota no detiene el lote
            registro = {"id": ruta.stem, "error": repr(e), "lines": [], "seconds": None}
            fallidos += 1
        salida.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")
        if k % cada == 0:
            seg = time.perf_counter() - t_total
            log(f"{k}/{len(rutas)} · nuevos {hechos} · fallidos {fallidos} · {seg / max(hechos, 1):.2f} s/CV")
    return hechos, fallidos


def cargar_json(destino_json):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(destino_json).glob("*.json"))]
