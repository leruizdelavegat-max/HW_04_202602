"""App de 4 pestañas: problema · OCR y EDA · resultados · estimador interactivo.

Ejecutar desde la raíz del repo:  .venv\Scripts\python -m streamlit run app/streamlit_app.py
"""
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src import config, metricas, reglas, texto  # noqa: E402

st.set_page_config(page_title="Años de experiencia desde la imagen del CV", page_icon="📄", layout="wide")

ETIQUETA = {"mediana": "Mediana", "tfidf_ridge": "TF-IDF + Ridge", "embeddings_ridge": "Embeddings + Ridge",
            "regla_fechas": "Regla de fechas (bonus)", "bert": "DistilBERT (fine-tuned)"}
MODELOS = list(ETIQUETA)


# ── Carga de artefactos (todo opcional: la app avisa qué falta) ────────────────

@st.cache_data
def leer_csv(rel, **kw):
    p = RAIZ / rel
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def leer_json(rel):
    p = RAIZ / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def leer_parquet(rel):
    p = RAIZ / rel
    return pd.read_parquet(p) if p.exists() else None


def falta(nombre):
    st.info(f"Aún no existe `{nombre}`. Ejecuta los notebooks de Colab y copia sus salidas al repo.")


run = leer_json("data/processed/run_config.json") or {}
TOPE = run.get("tope", config.TOPE_BASE)
BANDAS = metricas.nombres_bandas(TOPE)
dataset = leer_parquet("data/processed/dataset.parquet")
pred_test = leer_csv("output/tables/predicciones_test.csv", dtype={"id": str})


def figura(fig, pie):
    st.pyplot(fig, clear_figure=True)
    st.caption(pie)


st.title("¿Cuántos años de experiencia tiene este CV?")
st.write("De la **imagen** de un currículum al número: PaddleOCR lee el texto y cuatro modelos estiman los años "
         "de experiencia laboral (0–%d)." % TOPE)

tab1, tab2, tab3, tab4 = st.tabs(["1 · Problema y metodología", "2 · OCR y análisis exploratorio",
                                  "3 · Resultados de modelos", "4 · Estimador interactivo"])

# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    ocr_res = leer_json("output/tables/ocr_resumen.json") or {}
    masc = leer_json("output/tables/enmascaramiento.json") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CVs en la muestra", f"{run.get('n_total', 4000):,}")
    c2.metric("Por ocupación", run.get("n_por_ocupacion", 800))
    c3.metric("Tope del target", f"{TOPE} años")
    c4.metric("CVs con máscara", masc.get("cvs_con_alguna_mascara", "—"))
    st.markdown(f"""
### Objetivo
Estimar los **años de experiencia laboral** a partir de la imagen de un CV. No es buscar una frase: hay que
combinar fechas de empleo, número de puestos, progresión de cargos y fin de estudios.

### Datos
**PopResume** (Yu, Park & Moon, EMNLP 2026; CC BY 4.0): 60 884 CVs sintéticos de 5 ocupaciones de EE. UU.
Muestra estratificada de **{run.get('n_por_ocupacion', 800)} CVs por ocupación** (asignación proporcional por banda de
experiencia), semilla **{run.get('seed', config.SEED)}**, solo imágenes **sin foto** (`has_photo == False`).

### Target
`exp_year` recortado a 0–{TOPE}. Con tope 20 queda en el tope el {100 * run.get('pct_en_tope_20', float('nan')):.1f} % de la
muestra (regla del issue: subir el tope solo si supera el 15 %).

### Regla de enmascaramiento
Las features salen **solo** del OCR. Se reemplazan por `[MASK]` la experiencia total declarada («12 years of
experience», «Experience: 7 yrs»), la edad y la fecha de nacimiento. Las **fechas de cada empleo y los años de
graduación se conservan**: son la señal legítima. Además se borran nombre, correo, teléfono y dirección.
Nunca se usan `age`, `sex`, `race`, nombres, `state` ni `region` como features.

### OCR y orden de lectura
PaddleOCR {ocr_res.get('paddleocr', '3.x')} con `{config.OCR_DET}` + `{config.OCR_REC}` (renders limpios en inglés; el mismo
modelo corre en CPU en esta app). Orden de lectura: se detecta un canal vertical libre de cajas; si existe, la
página se lee **columna izquierda completa y luego la derecha**, por franjas separadas por títulos a todo el ancho.
Umbral de confianza elegido minimizando el CER: **{run.get('umbral_confianza', '—')}**.

### Modelos (mismo split 70/15/15 estratificado por ocupación × banda)
0. **Mediana** del train (piso). 1. **TF-IDF + Ridge** (palabras 1–2, con dígitos). 2. **Embeddings congelados**
(all-MiniLM-L6-v2, trozos promediados) + Ridge. 3. **DistilBERT** como regresor (MSE, target ÷ {TOPE}, cabeza + cola
de 512 tokens, early stopping por MAE). Bonus: **regla** que suma rangos de fechas fusionando solapamientos.

### Limitaciones
CVs **sintéticos** renderizados con plantillas, 5 ocupaciones, estadísticas de EE. UU., solo inglés. El estimador
sirve para **pre-llenar o verificar** un campo, nunca para decidir una contratación.
""")

# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Distribución del target")
    if dataset is None:
        falta("data/processed/dataset.parquet")
    else:
        en_tope = (dataset.y >= TOPE).sum()
        fig, ejes = plt.subplots(1, 2, figsize=(13, 3.6))
        sns.histplot(dataset.y, bins=range(0, int(TOPE) + 2), ax=ejes[0], color="#2a78d6")
        ejes[0].set_title(f"Todos los CVs (en el tope de {TOPE}: {en_tope})"); ejes[0].set_xlabel("años")
        sns.boxplot(data=dataset, x="y", y="job", ax=ejes[1], color="#eb6834"); ejes[1].set_ylabel(""); ejes[1].set_xlabel("años")
        figura(fig, f"La experiencia se concentra en los primeros años y hay {en_tope} CVs ({en_tope / len(dataset):.1%}) "
                    "apilados en el tope. Las ocupaciones difieren: eso ya da algo de señal a cualquier modelo.")
        st.dataframe(pd.crosstab(dataset.job, dataset.banda.map(dict(enumerate(BANDAS))), margins=True), width="stretch")

    st.subheader("Calidad del OCR")
    cal = leer_csv("output/tables/ocr_calidad_por_ocupacion.csv", index_col=0)
    if cal is None:
        falta("output/tables/ocr_calidad_por_ocupacion.csv")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("CER medio", f"{cal.loc['TOTAL', 'CER_medio']:.2%}")
        c2.metric("WER medio", f"{cal.loc['TOTAL', 'WER_medio']:.2%}")
        c3.metric("Años de referencia leídos", f"{cal.loc['TOTAL', 'anios_leidos']:.1%}")
        st.dataframe(cal.round(4), width="stretch")
        st.caption("CER/WER: errores por carácter/palabra frente al texto original `text` (solo se usa para medir). "
                   "«Años leídos» es la proporción de años de la referencia que aparecen en el OCR: la señal clave del target.")
    if (RAIZ / "output/figures/ocr_cer_wer.png").exists():
        st.image(str(RAIZ / "output/figures/ocr_cer_wer.png"),
                 caption="Distribución del CER y WER por ocupación y de los años recuperados por CV.")

    st.subheader("Ejemplo: imagen con cajas de PaddleOCR y texto extraído")
    ej = leer_json("output/tables/ocr_ejemplo.json")
    c1, c2 = st.columns(2)
    if (RAIZ / "output/figures/ocr_ejemplo_cajas.png").exists():
        c1.image(str(RAIZ / "output/figures/ocr_ejemplo_cajas.png"),
                 caption="Cada caja es una línea detectada; el número es su orden de lectura y el color su confianza.")
    if ej:
        c2.text_area("Texto en orden de lectura", ej["texto"], height=560)

# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    comp = leer_csv("output/tables/model_comparison.csv")
    if comp is None or pred_test is None:
        falta("output/tables/model_comparison.csv")
    else:
        st.subheader("Comparación en test")
        st.dataframe(comp.round(3), width="stretch", hide_index=True)
        st.caption("MAE es la métrica principal (años de error promedio). acc_±k: % de CVs a k años o menos. "
                   "Los tiempos de inferencia son por CV en la GPU T4 de Colab.")

        st.subheader("Predicho vs real")
        m = st.selectbox("Modelo", MODELOS, format_func=ETIQUETA.get, index=MODELOS.index("bert"))
        fig, eje = plt.subplots(figsize=(6, 5))
        eje.scatter(pred_test.y + np.random.default_rng(0).uniform(-.25, .25, len(pred_test)), pred_test[f"p_{m}"],
                    s=8, alpha=.4, color="#2a78d6")
        eje.plot([0, TOPE], [0, TOPE], color="#898781", lw=1, label="45°")
        media = pred_test.groupby("y")[f"p_{m}"].mean()
        eje.plot(media.index, media.values, color="#eb6834", lw=2, label="predicción media por año real")
        eje.set_xlabel("años reales"); eje.set_ylabel("años predichos"); eje.legend()
        c1, c2 = st.columns([3, 2])
        with c1:
            figura(fig, "Si la línea naranja es más plana que la diagonal, el modelo comprime hacia la media "
                        "(regresión a la media): sobreestima juniors y subestima seniors.")
        with c2:
            mat = leer_json("output/tables/matrices_confusion.json")
            if mat:
                fig, eje = plt.subplots(figsize=(4.2, 3.6))
                sns.heatmap(np.array(mat["matrices"][m]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=eje,
                            xticklabels=mat["bandas"], yticklabels=mat["bandas"])
                eje.set_xlabel("banda predicha"); eje.set_ylabel("banda real")
                figura(fig, "Matriz de confusión por bandas: la diagonal son aciertos de banda.")

        st.subheader("MAE por banda real y por ocupación")
        c1, c2 = st.columns(2)
        pb, po = leer_csv("output/tables/mae_por_banda.csv", index_col=0), leer_csv("output/tables/mae_por_ocupacion.csv", index_col=0)
        if pb is not None:
            c1.dataframe(pb.round(2), width="stretch")
            c1.caption("Dónde se equivoca cada modelo: los extremos (juniors y seniors) suelen ser los más difíciles.")
        if po is not None:
            c2.dataframe(po.round(2), width="stretch")
            c2.caption("Error por ocupación con el mismo test.")

        st.subheader("Ganancia de BERT (IC 95 % bootstrap pareado)")
        gan = leer_csv("output/tables/ganancia_bert_ic.csv")
        if gan is not None:
            fig, eje = plt.subplots(figsize=(7, 2.6))
            y_pos = np.arange(len(gan))
            eje.errorbar(gan.ganancia_MAE, y_pos, xerr=[gan.ganancia_MAE - gan.IC95_inf, gan.IC95_sup - gan.ganancia_MAE],
                         fmt="o", color="#2a78d6", capsize=4)
            eje.axvline(0, color="#d03b3b", lw=1); eje.set_yticks(y_pos, gan.baseline)
            eje.set_xlabel("MAE(baseline) − MAE(BERT)  [años]")
            figura(fig, "A la derecha de la línea roja BERT es mejor; si el intervalo no toca la línea, la diferencia es "
                        "estadísticamente significativa.")
            st.dataframe(gan.round(3), width="stretch", hide_index=True)

        st.subheader("Sensibilidad metodológica (Pregunta 3)")
        q3 = leer_csv("output/tables/q3_sensibilidad.csv")
        if q3 is not None:
            st.dataframe(q3.round(3), width="stretch", hide_index=True)
            st.caption("Cada fila cambia una sola decisión respecto a la configuración base y reporta el MAE de test.")

        st.subheader("Comprobación de equidad (Pregunta 4)")
        q4 = leer_csv("output/tables/q4_equidad.csv")
        if q4 is not None:
            st.dataframe(q4.round(3), width="stretch", hide_index=True)
            st.caption("MAE y error medio (predicho − real) de BERT por sexo y raza. `sex` y `race` no son features: "
                       "solo se usan aquí para auditar.")

# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando PaddleOCR (CPU)…")
def ocr_cpu():
    from src import ocr
    return ocr.crear_ocr(device="cpu")


@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_modelos():
    import joblib
    out = {}
    M = config.MODELOS
    if (M / "mediana.json").exists():
        v = json.loads((M / "mediana.json").read_text())["valor"]
        out["mediana"] = lambda ts: np.full(len(ts), v)
    if (M / "tfidf_ridge.joblib").exists():
        tf = joblib.load(M / "tfidf_ridge.joblib")
        out["tfidf_ridge"] = lambda ts: np.clip(tf.predict(ts), 0, TOPE)
    if (M / "embeddings_ridge.joblib").exists():
        from sentence_transformers import SentenceTransformer
        from src import modelos
        e = joblib.load(M / "embeddings_ridge.joblib")
        enc = SentenceTransformer(e["encoder"], device="cpu")
        out["embeddings_ridge"] = lambda ts: np.clip(e["ridge"].predict(modelos.embeber(ts, enc)), 0, TOPE)
    if (M / "regla.json").exists():
        anio = json.loads((M / "regla.json").read_text())["anio_actual"]
        out["regla_fechas"] = lambda ts: np.array([reglas.estimar(t, anio, TOPE) for t in ts])
    if (M / "bert_regresor" / "config.json").exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src import modelos
        tok = AutoTokenizer.from_pretrained(M / "bert_regresor")
        bert = AutoModelForSequenceClassification.from_pretrained(M / "bert_regresor")
        out["bert"] = lambda ts: modelos.predecir_bert(bert, tok, ts, TOPE, device="cpu")
    return out


def a_imagen(archivo):
    """Guarda la subida como PNG (PDF → primera página a ~200 dpi)."""
    destino = Path(tempfile.mkdtemp()) / "cv.png"
    if archivo.name.lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(archivo.getvalue())
        pdf[0].render(scale=200 / 72).to_pil().save(destino)
    else:
        from PIL import Image
        import io
        Image.open(io.BytesIO(archivo.getvalue())).convert("RGB").save(destino)
    return destino


with tab4:
    st.subheader("Sube un CV (imagen o PDF)")
    st.caption("El OCR corre en vivo en CPU con el mismo modelo del entrenamiento (puede tardar 10–60 s en una laptop). "
               "Los modelos se descargan de Drive a `models/` (ver README).")
    archivo = st.file_uploader("CV", type=["png", "jpg", "jpeg", "pdf"])
    if archivo is not None and st.button("Leer y estimar", type="primary"):
        from src import ocr as ocrmod
        ruta = a_imagen(archivo)
        with st.spinner("PaddleOCR leyendo…"):
            lineas, alto, ancho = ocrmod.leer_imagen(ocr_cpu(), ruta)
            ordenadas, _ = ocrmod.ordenar_lectura(lineas, ancho)
        limpio, conteo = texto.limpiar(ordenadas, run.get("umbral_confianza", 0.0))
        modelos_cargados = cargar_modelos()
        c1, c2 = st.columns([2, 3])
        c1.image(str(ruta), caption=f"{len(lineas)} líneas detectadas")
        with c2:
            if not modelos_cargados:
                st.warning("No hay modelos en `models/`.")
            preds = {ETIQUETA[k]: float(f([limpio])[0]) for k, f in modelos_cargados.items()}
            st.dataframe(pd.Series(preds, name="años estimados").round(1).to_frame(), width="stretch")
            if sum(conteo.values()):
                st.caption(f"Se enmascararon {sum(conteo.values())} fragmentos que revelaban la respuesta: {conteo}")
            st.text_area("Texto que ven los modelos (OCR limpio)", limpio, height=360)

    st.divider()
    st.subheader("Explorar CVs de test")
    if dataset is None or pred_test is None:
        falta("output/tables/predicciones_test.csv")
    else:
        test = pred_test.merge(dataset[["id", "texto_final"]], on="id")
        ocup = st.selectbox("Ocupación", sorted(test.job.unique()))
        sub = test[test.job == ocup].sort_values("id")
        cv = st.selectbox("CV", sub.id, format_func=lambda i: f"{i} · real {sub.set_index('id').y[i]:.0f} años")
        fila = sub.set_index("id").loc[cv]
        cols = st.columns(len(MODELOS) + 1)
        cols[0].metric("Real", f"{fila.y:.0f}")
        for c, m in zip(cols[1:], MODELOS):
            c.metric(ETIQUETA[m], f"{fila[f'p_{m}']:.1f}", f"{fila[f'p_{m}'] - fila.y:+.1f}", delta_color="off")
        st.text_area("Texto OCR limpio", fila.texto_final, height=300)

        st.subheader("Buenas y malas predicciones de BERT")
        test["err"] = test.p_bert - test.y
        top = leer_csv("output/tables/top10_errores_bert.csv", dtype={"id": str})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Aciertos** (error más pequeño)")
            for r in test.reindex(test.err.abs().sort_values().index).head(3).itertuples():
                with st.expander(f"{r.id} · {r.job} · real {r.y:.0f} · BERT {r.p_bert:.1f}"):
                    st.write("Las fechas de empleo se leyeron bien y el historial cabe en los 512 tokens; "
                             f"la regla de fechas estima {r.p_regla_fechas:.1f}.")
                    st.text(r.texto_final[:1200])
        with c2:
            st.markdown("**Errores grandes** (con causa del análisis de errores)")
            causas = top.set_index("id").causa_sugerida if top is not None else {}
            for r in test.reindex(test.err.abs().sort_values(ascending=False).index).head(3).itertuples():
                with st.expander(f"{r.id} · {r.job} · real {r.y:.0f} · BERT {r.p_bert:.1f}"):
                    st.write(f"Causa: **{causas.get(r.id, 'sin clasificar')}**. "
                             f"CER del OCR {r.CER:.1%}, años de la referencia recuperados {r.recall_anios:.0%}.")
                    st.text(r.texto_final[:1200])
