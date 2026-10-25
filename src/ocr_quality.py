"""Etapa 2 · Calidad del OCR frente al `text` de referencia: CER, WER y años leídos.

Mismo cálculo que `notebooks/02_calidad_limpieza_split.ipynb`: ambos textos se normalizan igual
(`texto.normalizar_para_cer`) y se mide con jiwer. `text` se usa SOLO aquí, nunca como entrada de un modelo."""
import numpy as np

from .texto import anios, normalizar_para_cer, recall_anios


def _medir(fn, referencia, ocr):
    """NaN si la referencia está vacía (jiwer devolvería un número inflado)."""
    r, h = normalizar_para_cer(referencia), normalizar_para_cer(ocr)
    return fn(r, h) if r else np.nan


def cer(referencia, ocr):
    import jiwer
    return _medir(jiwer.cer, referencia, ocr)


def wer(referencia, ocr):
    import jiwer
    return _medir(jiwer.wer, referencia, ocr)


def calidad(df, col_ref="text", col_ocr="texto_ocr"):
    """Agrega CER, WER y recall_anios (proporción de años de la referencia presentes en el OCR) por CV."""
    out = df.copy()
    out["CER"] = [cer(r, h) for r, h in zip(df[col_ref], df[col_ocr])]
    out["WER"] = [wer(r, h) for r, h in zip(df[col_ref], df[col_ocr])]
    out["recall_anios"] = [recall_anios(r, h) for r, h in zip(df[col_ref], df[col_ocr])]
    return out


def por_ocupacion(df):
    """Tabla de `output/tables/ocr_calidad_por_ocupacion.csv` (con fila TOTAL)."""
    t = df.groupby("job").agg(n=("id", "size"), CER_medio=("CER", "mean"), CER_mediana=("CER", "median"),
                              WER_medio=("WER", "mean"), anios_leidos=("recall_anios", "mean"),
                              conf_media=("conf_media", "mean"))
    t.loc["TOTAL"] = [len(df), df.CER.mean(), df.CER.median(), df.WER.mean(), df.recall_anios.mean(),
                      df.conf_media.mean()]
    return t


__all__ = ["anios", "calidad", "cer", "normalizar_para_cer", "por_ocupacion", "recall_anios", "wer"]
