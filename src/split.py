"""Etapa 3 · Partición 70/15/15 por ID, estratificada por ocupación × banda de experiencia, semilla 42.

Mismo código que `notebooks/02_calidad_limpieza_split.ipynb`. Los archivos `data/splits/{train,val,test}.csv`
tienen `id`, `job` y `exp_year` (target ya recortado al tope) y los comparten TODOS los modelos."""
import numpy as np
import pandas as pd

from . import config
from .data import recortar_target

NOMBRES = ("train", "val", "test")


def dividir(df, tope, seed=config.SEED):
    """Agrega `y` (target recortado), `banda` y `split` a un DataFrame con `id`, `job` y `exp_year`."""
    from sklearn.model_selection import train_test_split
    df = df.copy()
    df["y"] = recortar_target(df[config.TARGET], tope)
    df["banda"] = df["y"].map(config.banda)
    estrato = df["job"] + "|" + df["banda"].astype(str)
    tr, temp = train_test_split(df["id"], test_size=0.30, stratify=estrato, random_state=seed)
    va, te = train_test_split(temp, test_size=0.50, stratify=estrato[temp.index], random_state=seed)
    df["split"] = np.where(df["id"].isin(tr), "train", np.where(df["id"].isin(va), "val", "test"))
    return df


def guardar(df, carpeta=config.SPLITS):
    """Escribe train.csv, val.csv y test.csv (id, job, exp_year recortado)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre in NOMBRES:
        parte = df.loc[df.split == nombre, ["id", "job", "y"]].rename(columns={"y": config.TARGET})
        parte.sort_values("id").to_csv(carpeta / f"{nombre}.csv", index=False)


def cargar(nombre, carpeta=config.SPLITS):
    return pd.read_csv(carpeta / f"{nombre}.csv", dtype={"id": str})


__all__ = ["NOMBRES", "cargar", "dividir", "guardar"]
