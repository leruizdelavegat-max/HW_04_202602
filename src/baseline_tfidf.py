"""Baseline 1 · TF-IDF + Ridge.

Palabras 1–2 + caracteres 3–5, `sublinear_tf`, `min_df=2` y CON dígitos (los años son la señal: sin ellos el
MAE sube un 107 %, ver P3). `alpha` elegido solo en validación (`output/tables/grid_tfidf.csv`).
Implementación en `modelos.py`; se entrena en `notebooks/03_baselines.ipynb`."""
import json

from . import config
from .modelos import _sin_digitos, tfidf_ridge


def hiperparametros():
    """Configuración elegida en validación (`output/tables/hiperparametros_baselines.json`)."""
    return json.loads((config.TABLAS / "hiperparametros_baselines.json").read_text(encoding="utf-8"))


def cargar(carpeta=config.MODELOS):
    import joblib
    return joblib.load(carpeta / "tfidf_ridge.joblib")


__all__ = ["_sin_digitos", "cargar", "hiperparametros", "tfidf_ridge"]
