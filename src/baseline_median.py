"""Baseline 0 · Mediana: predice siempre la mediana de `y` en train. Todo otro modelo debe superarla.

Implementación en `modelos.py`; se entrena y evalúa en `notebooks/03_baselines.ipynb`."""
import json

from . import config
from .modelos import Mediana


def cargar(carpeta=config.MODELOS):
    """Valor de la mediana guardado por el notebook 03 (`models/mediana.json`)."""
    return json.loads((carpeta / "mediana.json").read_text())["valor"]


__all__ = ["Mediana", "cargar"]
