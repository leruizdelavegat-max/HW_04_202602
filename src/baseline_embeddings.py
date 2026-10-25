"""Baseline 2 · Embeddings congelados + Ridge.

`all-MiniLM-L6-v2` sin fine-tuning. Los CVs que superan el límite del encoder se parten en trozos de 180
palabras (solape 30) y se promedian sus vectores. `alpha` elegido solo en validación
(`output/tables/grid_embeddings.csv`). Implementación en `modelos.py`; se entrena en `notebooks/03_baselines.ipynb`."""
from . import config
from .baseline_tfidf import hiperparametros
from .modelos import embeber, trozos

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def cargar(carpeta=config.MODELOS):
    """{"encoder": nombre, "ridge": Ridge} guardado por el notebook 03."""
    import joblib
    return joblib.load(carpeta / "embeddings_ridge.joblib")


__all__ = ["ENCODER", "cargar", "embeber", "hiperparametros", "trozos"]
