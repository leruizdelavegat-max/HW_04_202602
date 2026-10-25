"""Muestreo estratificado de PopResume y descarga selectiva de imágenes (sin foto)."""
import io
import time

import numpy as np
import pandas as pd

from . import config


def cargar_metadatos():
    """Config `text` de PopResume: una fila por CV (60 884). Pesa poco: se baja entera.
    No se asume el nombre del split: la ficha del dataset no lo fija; se toma el único que haya."""
    from datasets import load_dataset
    dd = load_dataset(config.HF_DATASET, "text")
    ds = dd["train"] if "train" in dd else dd[next(iter(dd))]
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str).str.strip()
    faltan = [c for c in ("id", "job", config.TARGET, "text") if c not in df]
    assert not faltan, f"Columnas esperadas que no están en PopResume: {faltan}. Hay: {list(df.columns)}"
    return df


def _clave_id(x):
    """Ids comparables aunque un archivo los guarde con ceros a la izquierda ('000123') y otro como número (123)."""
    s = str(x).strip()
    return (s.lstrip("0") or "0") if s.isdigit() else s


def muestrear(meta, n=config.N_POR_OCUPACION, seed=config.SEED):
    """n CVs por ocupación. Dentro de cada ocupación se estratifica por banda de experiencia
    (asignación proporcional), así la muestra conserva la distribución del target."""
    rng = np.random.default_rng(seed)
    meta = meta.dropna(subset=[config.TARGET]).copy()
    meta["_banda"] = meta[config.TARGET].clip(0, config.TOPE_BASE).map(config.banda)
    partes = []
    for job, g in meta.groupby("job", sort=True):
        if len(g) < n:
            raise ValueError(f"{job}: solo {len(g)} CVs, se piden {n}")
        cuotas = (g["_banda"].value_counts(normalize=True) * n)
        enteras = np.floor(cuotas).astype(int)
        # reparto de los sobrantes por mayor residuo (suma exacta = n)
        resto = n - enteras.sum()
        for b in (cuotas - enteras).sort_values(ascending=False).index[:resto]:
            enteras[b] += 1
        for b, k in enteras.items():
            sub = g[g["_banda"] == b]
            partes.append(sub.iloc[rng.choice(len(sub), size=k, replace=False)])
    muestra = pd.concat(partes).drop(columns="_banda").sort_values(["job", "id"]).reset_index(drop=True)
    assert muestra["id"].is_unique
    return muestra


def shards_de_imagenes(token=None):
    """Lista los parquet de la config `images` agrupados por ocupación (slug del nombre de archivo)."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    rutas = sorted(fs.glob(f"datasets/{config.HF_DATASET}/data/images/*.parquet"))
    assert rutas, ("No se encontraron los parquet de imágenes. ¿Aceptaste las condiciones de PopResume en "
                   "Hugging Face con la misma cuenta del token HF_TOKEN?")
    por_slug = {}
    for r in rutas:
        slug = r.rsplit("/", 1)[-1].rsplit("-", 1)[0]
        por_slug.setdefault(slug, []).append(r)
    return fs, por_slug


def descargar_imagenes(ids, destino, token=None, log=print):
    """Recorre los shards por *row group*: primero lee solo `id` y `has_photo` (barato) y
    únicamente baja la columna `image` de los row groups que contienen CVs de la muestra.
    Guarda cada CV como PNG en `destino/<id>.png`. Es reanudable: salta lo que ya existe."""
    import pyarrow.parquet as pq
    destino.mkdir(parents=True, exist_ok=True)
    # clave normalizada → id tal como está en la muestra (así se llama el PNG)
    faltan = {_clave_id(i): str(i) for i in ids if not (destino / f"{i}.png").exists()}
    log(f"Imágenes por bajar: {len(faltan)} de {len(ids)}")
    if not faltan:
        return
    fs, por_slug = shards_de_imagenes(token)
    t0 = time.perf_counter()
    for slug, rutas in por_slug.items():
        for ruta in rutas:
            if not faltan:
                return
            with fs.open(ruta, "rb", block_size=8 * 2**20) as fh:
                pf = pq.ParquetFile(fh)
                for rg in range(pf.num_row_groups):
                    clave = pf.read_row_group(rg, columns=["id", "has_photo"]).to_pandas()
                    clave["k"] = clave["id"].map(_clave_id)
                    sin_foto = clave["has_photo"].map(lambda v: str(v).strip().lower() in ("false", "0"))
                    filas = clave.index[sin_foto & clave["k"].isin(faltan)]
                    if len(filas) == 0:
                        continue
                    imgs = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
                    for f in filas:
                        img = imgs[f]
                        datos = img["bytes"] if isinstance(img, dict) else img
                        k = clave.at[f, "k"]
                        (destino / f"{faltan[k]}.png").write_bytes(_a_png(datos))
                        del faltan[k]
            log(f"  {ruta.rsplit('/', 1)[-1]}: faltan {len(faltan)} · {time.perf_counter() - t0:.0f} s")
    if faltan:
        log(f"⚠ No se encontraron {len(faltan)} ids sin foto: {sorted(faltan.values())[:10]}…")


def _a_png(datos):
    """Normaliza a PNG (los shards pueden traer JPEG o PNG)."""
    from PIL import Image
    if datos[:8] == b"\x89PNG\r\n\x1a\n":
        return datos
    buf = io.BytesIO()
    Image.open(io.BytesIO(datos)).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def recortar_target(y, tope):
    return np.clip(np.asarray(y, dtype=float), 0, tope)


def elegir_tope(y):
    """Tope 20 salvo que >15 % quede en él; entonces el menor alternativo que lo resuelva."""
    y = np.asarray(y, dtype=float)
    for tope in (config.TOPE_BASE, *config.TOPES_ALTERNATIVOS):
        if (y >= tope).mean() <= config.UMBRAL_TOPE:
            return tope
    return config.TOPES_ALTERNATIVOS[-1]
