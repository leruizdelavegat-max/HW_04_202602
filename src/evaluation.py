"""Etapa 6 · Evaluación: métricas, comparación, bootstrap pareado y pruebas contrafactuales (P1).

Implementación en `metricas.py` y `reglas.py`; se ejecuta en `notebooks/05_evaluacion_preguntas.ipynb`.
`formato_enunciado` da a `model_comparison.csv` los nombres de columna del enunciado."""
import json

import pandas as pd

from . import config
from .metricas import (bandas, bootstrap_ganancia, mae_por_grupo, matriz_bandas, nombres_bandas, resumen,
                       tabla_comparacion)
from .reglas import adelantar_inicio, desplazar_fechas, quitar_empleo_mas_antiguo

# nombre interno → nombre del enunciado (tiempos en segundos y en milisegundos por CV)
COLUMNAS_ENUNCIADO = {
    "modelo": "Model", "MAE": "MAE (years)", "RMSE": "RMSE (years)",
    "acc_±1": "Acc ±1", "acc_±2": "Acc ±2", "acc_±3": "Acc ±3",
    "acc_banda": "Band accuracy", "macroF1_banda": "Band macro-F1", "R2": "R²",
    "tiempo_entrenamiento_s": "Training time (s)", "tiempo_inferencia_ms_por_CV": "Inference time per CV (ms)",
    "sesgo_medio": "Mean error (pred − true)",
}
ORDEN = list(COLUMNAS_ENUNCIADO.values())


def formato_enunciado(comp):
    """Tabla de `tabla_comparacion` con las columnas del enunciado (más el sesgo medio al final)."""
    t = comp.rename(columns=COLUMNAS_ENUNCIADO)
    return t[[c for c in ORDEN if c in t]]


def a_formato_interno(comp):
    """Inverso de `formato_enunciado` (acepta también una tabla ya en formato interno)."""
    return comp.rename(columns={v: k for k, v in COLUMNAS_ENUNCIADO.items()})


def exportar_metricas(destino=config.OUTPUT / "metrics"):
    """Copia en `output/metrics/` las métricas de test de cada modelo y la ganancia de BERT con su IC, a partir de
    las tablas ya calculadas (no reentrena ni vuelve a predecir)."""
    destino.mkdir(parents=True, exist_ok=True)
    comp = a_formato_interno(pd.read_csv(config.TABLAS / "model_comparison.csv"))
    gan = pd.read_csv(config.TABLAS / "ganancia_bert_ic.csv")
    for _, fila in comp.iterrows():
        nombre = fila["modelo"].split(" (")[0].lower().replace(" + ", "_").replace(" ", "_").replace("-", "")
        (destino / f"{nombre}.json").write_text(json.dumps(
            {"particion": "test", **{COLUMNAS_ENUNCIADO[k]: v for k, v in fila.items()}}, ensure_ascii=False,
            indent=2), encoding="utf-8")
    (destino / "bert_gain_ci.json").write_text(json.dumps(gan.to_dict(orient="records"), ensure_ascii=False,
                                                          indent=2), encoding="utf-8")
    return sorted(p.name for p in destino.glob("*.json"))


__all__ = ["COLUMNAS_ENUNCIADO", "a_formato_interno", "adelantar_inicio", "bandas", "bootstrap_ganancia",
           "desplazar_fechas", "exportar_metricas", "formato_enunciado", "mae_por_grupo", "matriz_bandas",
           "nombres_bandas", "quitar_empleo_mas_antiguo", "resumen", "tabla_comparacion"]
