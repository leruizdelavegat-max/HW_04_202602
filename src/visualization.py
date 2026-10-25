"""Figuras de `output/figures/`, con el mismo código que generó cada una en los notebooks.

Cada función recibe los datos ya calculados y devuelve la figura de matplotlib; `guardar` la escribe con los
mismos ajustes (130 dpi, bordes ajustados)."""
import numpy as np

from . import config

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#0ca30c", "#898781"


def guardar(fig, nombre):
    fig.savefig(config.FIGURAS / nombre, dpi=130, bbox_inches="tight")


def cajas_ocr(imagen, lineas, titulo=""):
    """ocr_ejemplo_cajas.png: cajas de PaddleOCR (número = orden de lectura; color = confianza)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, eje = plt.subplots(figsize=(7, 9))
    eje.imshow(imagen)
    for k, l in enumerate(lineas):
        x0, y0, x1, y1 = l["box"]
        color = VERDE if l["conf"] >= 0.9 else "#fab219" if l["conf"] >= 0.7 else "#d03b3b"
        eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1, edgecolor=color))
        eje.text(x0, y0, str(k), fontsize=5, color=AZUL)
    eje.set_title(titulo); eje.axis("off")
    return fig


def calidad_ocr(df):
    """ocr_cer_wer.png: CER y WER por ocupación y años de referencia encontrados en el OCR."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=df, x="CER", y="job", ax=ejes[0], color=AZUL, fliersize=2); ejes[0].set_title("CER por ocupación")
    sns.boxplot(data=df, x="WER", y="job", ax=ejes[1], color=NARANJA, fliersize=2)
    ejes[1].set_title("WER por ocupación"); ejes[1].set_ylabel("")
    sns.histplot(df["recall_anios"], bins=20, ax=ejes[2], color=VERDE)
    ejes[2].set_title("Años de referencia encontrados en el OCR")
    fig.tight_layout()
    return fig


def distribucion_target(df, tope):
    """distribucion_target.png: histograma de años (con % en el tope) y caja por ocupación."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ejes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(df["y"], bins=range(0, int(tope) + 2), ax=ejes[0], color=AZUL)
    ejes[0].set_title(f"Años de experiencia (recortado a {tope}) · en el tope: {(df.y >= tope).mean():.1%}")
    sns.boxplot(data=df, x="y", y="job", ax=ejes[1], color=NARANJA); ejes[1].set_title("Por ocupación")
    ejes[1].set_ylabel("")
    fig.tight_layout()
    return fig


def pred_vs_real(te, modelos, etiquetas, tope):
    """pred_vs_real.png: un panel por modelo, línea de 45° (gris) y media por año real (naranja)."""
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(1, len(modelos), figsize=(4 * len(modelos), 4), sharey=True)
    for eje, m in zip(np.atleast_1d(ejes), modelos):
        eje.scatter(te.y + np.random.default_rng(0).uniform(-.25, .25, len(te)), te[f"p_{m}"], s=6, alpha=.35,
                    color=AZUL)
        eje.plot([0, tope], [0, tope], color=GRIS, lw=1)
        media = te.groupby("y")[f"p_{m}"].mean()
        eje.plot(media.index, media.values, color=NARANJA, lw=2)
        eje.set_title(f"{etiquetas[m]}\nMAE {np.abs(te[f'p_{m}'] - te.y).mean():.2f}", fontsize=10)
        eje.set_xlabel("años reales"); eje.set_xlim(-1, tope + 1); eje.set_ylim(-1, tope + 1)
    np.atleast_1d(ejes)[0].set_ylabel("años predichos")
    fig.tight_layout()
    return fig


def mae_banda_ocupacion(por_banda, por_ocup):
    """mae_banda_ocupacion.png: mapas de calor del MAE por banda real y por ocupación."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ejes = plt.subplots(1, 2, figsize=(16, 3.8))
    sns.heatmap(por_banda, annot=True, fmt=".2f", cmap="Blues", ax=ejes[0]); ejes[0].set_title("MAE por banda real")
    sns.heatmap(por_ocup, annot=True, fmt=".2f", cmap="Blues", ax=ejes[1]); ejes[1].set_title("MAE por ocupación")
    ejes[1].set_ylabel("")
    fig.tight_layout()
    return fig


def matrices_confusion(matrices, etiquetas, nombres_bandas):
    """matrices_confusion.png: matriz de bandas (real × predicha) por modelo. `matrices`: {modelo: matriz}."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ejes = plt.subplots(1, len(matrices), figsize=(3.6 * len(matrices), 3.4))
    for eje, (m, cm) in zip(np.atleast_1d(ejes), matrices.items()):
        sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=eje,
                    xticklabels=nombres_bandas, yticklabels=nombres_bandas)
        eje.set_title(etiquetas[m], fontsize=10); eje.set_xlabel("predicha"); eje.set_ylabel("real")
    fig.tight_layout()
    return fig


__all__ = ["cajas_ocr", "calidad_ocr", "distribucion_target", "guardar", "mae_banda_ocupacion",
           "matrices_confusion", "pred_vs_real"]
