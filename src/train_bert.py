"""Etapa 5 · Fine-tuning de DistilBERT como regresor.

`AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`, pérdida MSE (Huber comparada en
P3), target ÷ tope durante el entrenamiento, salida × tope recortada a [0, tope], 512 tokens con estrategia
cabeza + cola (382 + 128) y early stopping por MAE de validación. Implementación en `modelos.py`; se entrena en
`notebooks/04_bert_regresor.ipynb` (GPU T4). La configuración usada está en `output/tables/bert_config.json`."""
import json

from . import config
from .modelos import DatasetRegresion, codificar, entrenar_bert, predecir_bert, texto_visto

MODELO_BASE = "distilbert-base-uncased"


def configuracion():
    return json.loads((config.TABLAS / "bert_config.json").read_text(encoding="utf-8"))


def cargar(carpeta=config.MODELOS / "bert_regresor"):
    """(modelo, tokenizer) ya entrenados; los pesos no están en el repo (ver README §10)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return AutoModelForSequenceClassification.from_pretrained(carpeta), AutoTokenizer.from_pretrained(carpeta)


__all__ = ["DatasetRegresion", "MODELO_BASE", "cargar", "codificar", "configuracion", "entrenar_bert",
           "predecir_bert", "texto_visto"]
