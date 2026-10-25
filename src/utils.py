"""Utilidades comunes: rutas, semilla, tope del target y bandas de experiencia.

Punto de entrada con el nombre que pide el enunciado; las constantes viven en `config.py` y el recorte del
target en `data.py`."""
import os
import random

import numpy as np

from .config import (BANDAS, DATA, FIGURAS, MODELOS, NOMBRES_BANDAS, OUTPUT, PROCESADOS, PROHIBIDAS, RAIZ, SEED,
                     SPLITS, TABLAS, TARGET, TOPE_BASE, banda)
from .data import elegir_tope, recortar_target

METRICAS = OUTPUT / "metrics"
OCR_JSON = DATA / "ocr"


def fijar_semilla(seed=SEED):
    """Misma semilla en Python, NumPy y (si está instalado) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


__all__ = ["BANDAS", "DATA", "FIGURAS", "METRICAS", "MODELOS", "NOMBRES_BANDAS", "OCR_JSON", "OUTPUT", "PROCESADOS",
           "PROHIBIDAS", "RAIZ", "SEED", "SPLITS", "TABLAS", "TARGET", "TOPE_BASE", "banda", "elegir_tope",
           "fijar_semilla", "recortar_target"]
