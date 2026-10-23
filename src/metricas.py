"""Métricas de regresión, de bandas y bootstrap pareado."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

from . import config


def bandas(y):
    return np.array([config.banda(v) for v in np.asarray(y, dtype=float)])


def resumen(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(p - y)
    by, bp = bandas(y), bandas(p)
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "acc_±1": float((err <= 1).mean()),
        "acc_±2": float((err <= 2).mean()),
        "acc_±3": float((err <= 3).mean()),
        "acc_banda": float((by == bp).mean()),
        "macroF1_banda": f1_score(by, bp, average="macro", labels=range(len(config.BANDAS)), zero_division=0),
        "R2": r2_score(y, p),
        "sesgo_medio": float((p - y).mean()),
    }


def matriz_bandas(y, p):
    return confusion_matrix(bandas(y), bandas(p), labels=range(len(config.BANDAS)))


def nombres_bandas(tope):
    n = list(config.NOMBRES_BANDAS)
    n[-1] = f"11–{int(tope)}"
    return n


def bootstrap_ganancia(y, p_base, p_bert, n=2000, seed=config.SEED):
    """Ganancia = MAE(baseline) − MAE(BERT), remuestreando los MISMOS CVs para ambos modelos
    (bootstrap pareado). IC 95 % por percentiles. Positivo = BERT mejor."""
    y, a, b = (np.asarray(v, float) for v in (y, p_base, p_bert))
    ea, eb = np.abs(a - y), np.abs(b - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    difs = ea[idx].mean(axis=1) - eb[idx].mean(axis=1)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {"ganancia_MAE": float(ea.mean() - eb.mean()), "IC95_inf": float(lo), "IC95_sup": float(hi),
            "significativa": bool(lo > 0 or hi < 0), "n_boot": n}


def mae_por_grupo(df, col_grupo, col_y="y", col_p="pred"):
    g = df.assign(err=(df[col_p] - df[col_y]))
    return g.groupby(col_grupo).agg(n=("err", "size"),
                                    MAE=("err", lambda e: e.abs().mean()),
                                    error_medio=("err", "mean")).reset_index()


def tabla_comparacion(y, predicciones, tiempos):
    """`predicciones`: {modelo: array}; `tiempos`: {modelo: (seg_entrenamiento, seg_inferencia_por_CV)}."""
    filas = []
    for nombre, p in predicciones.items():
        fila = {"modelo": nombre, **resumen(y, p)}
        te, ti = tiempos.get(nombre, (np.nan, np.nan))
        fila["tiempo_entrenamiento_s"] = te
        fila["tiempo_inferencia_ms_por_CV"] = ti * 1000 if ti == ti else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)
