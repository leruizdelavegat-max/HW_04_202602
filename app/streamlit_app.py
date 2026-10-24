r"""App de 4 pestañas: problema · OCR y EDA · resultados · estimador interactivo.

Ejecutar desde la raíz del repo:  .venv\Scripts\python -m streamlit run app.py
(equivale a  streamlit run app/streamlit_app.py).

Todo lo que se muestra sale de los archivos del pipeline (data/processed, output/tables, output/figures) o se
calcula aquí a partir de ellos. Si un archivo falta, la sección lo dice: no hay valores de respaldo inventados.
Privacidad: ningún texto ni imagen se muestra sin pasar por src/anonimizar.py o por la limpieza del pipeline.
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
for p in (RAIZ, RAIZ / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src import config, reglas, texto  # noqa: E402
import ui  # noqa: E402
from ui import ETIQUETA, MODELOS, REQUERIDOS, num, pct  # noqa: E402

st.set_page_config(page_title="Años de experiencia desde la imagen del CV", page_icon="📄", layout="wide")
ui.estilo()

OCUPACION = {"accountants_auditors": "Contadores y auditores", "construction_laborers": "Obreros de construcción",
             "elementary_middle_school_teachers": "Docentes de primaria y secundaria",
             "registered_nurses": "Enfermería registrada", "software_developers": "Desarrolladores de software"}


# ── Carga de artefactos (cada uno opcional: la app avisa qué falta) ───────────────

@st.cache_data
def leer_csv(rel, **kw):
    p = RAIZ / rel
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def leer_json(rel):
    p = RAIZ / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def leer_dataset():
    p = RAIZ / "data/processed/dataset.parquet"
    if not p.exists():
        return None
    ds = pd.read_parquet(p)
    ds["ocupacion"] = ds.job.map(OCUPACION).fillna(ds.job)
    return ds


@st.cache_data
def indicadores(ds):
    """Indicadores calculados sobre el estado real del dataset procesado (no sobre valores guardados)."""
    contacto = lambda t: bool(texto.RE_EMAIL.search(t) or texto.RE_TEL.search(t))  # noqa: E731
    return {
        "n": len(ds),
        "por_ocupacion": ds.job.value_counts(),
        "split": ds.split.value_counts(),
        "en_tope": int((ds.y >= TOPE).sum()),
        "con_mascara": int(ds.texto_final.str.contains(r"\[MASK\]", regex=True).sum()),
        "contacto_crudo": int(ds.texto_crudo.map(contacto).sum()),
        "contacto_final": int(ds.texto_final.map(contacto).sum()),
        "fechas_pegadas": int(ds.texto_crudo.map(lambda t: bool(texto.RE_FECHAS_PEGADAS.search(t or ""))).sum()),
    }


run = leer_json("data/processed/run_config.json") or {}
TOPE = run.get("tope", config.TOPE_BASE)
BANDAS = list(config.NOMBRES_BANDAS[:-1]) + [f"11–{int(TOPE)}"]
dataset = leer_dataset()
pred_test = leer_csv("output/tables/predicciones_test.csv", dtype={"id": str})
if pred_test is not None:
    pred_test["ocupacion"] = pred_test.job.map(OCUPACION).fillna(pred_test.job)
regla_cfg = leer_json("output/tables/hiperparametros_baselines.json") or {}
ANIO_REGLA = (regla_cfg.get("regla") or {}).get("anio_actual")

COMO_COLAB = "Se genera con los notebooks de Colab (ver README, «Reproducción»)."

st.title("¿Cuántos años de experiencia tiene este CV?")
st.markdown('<p class="subtitulo">De la <b>imagen</b> de un currículum a un número: PaddleOCR lee el texto y '
            f'cuatro modelos estiman los años de experiencia laboral (0–{TOPE}). Datos: PopResume, '
            'CVs sintéticos.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["1 · Problema y metodología", "2 · OCR y análisis exploratorio",
                                  "3 · Resultados de modelos", "4 · Estimador interactivo"])

# ════════════════════════════════════════════════════════════════════════════════
# 1 · PROBLEMA Y METODOLOGÍA
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if dataset is None:
        ui.falta("data/processed/dataset.parquet", COMO_COLAB)
        ind = None
    else:
        ind = indicadores(dataset)
        por_ocup = ind["por_ocupacion"]
        reparto = (f"{por_ocup.iloc[0]:,} × {len(por_ocup)} ocupaciones".replace(",", " ")
                   if por_ocup.nunique() == 1 else f"entre {por_ocup.min()} y {por_ocup.max()} por ocupación")
        sp = ind["split"]
        ui.kpis([
            ("CVs analizados", num(ind["n"], 0), reparto),
            ("Partición train / val / test", " / ".join(num(sp.get(k, 0), 0) for k in ("train", "val", "test")),
             "estratificada por ocupación × banda"),
            ("Tope del target", f"{TOPE} años", f"{ind['en_tope']} CVs en el tope ({pct(ind['en_tope'] / ind['n'])})"),
            ("CVs con máscara", num(ind["con_mascara"], 0),
             "frases con experiencia total, edad o nacimiento"),
            ("CVs sin datos de contacto", num(ind["n"] - ind["contacto_final"], 0),
             f"de {num(ind['n'], 0)}; {num(ind['contacto_crudo'], 0)} los tenían en el OCR"),
        ])
        masc = leer_json("output/tables/enmascaramiento.json")
        if masc and masc.get("cvs_con_alguna_mascara") != ind["con_mascara"]:
            st.warning(f"El conteo guardado de CVs con máscara ({masc.get('cvs_con_alguna_mascara')}) no coincide con "
                       f"el dataset procesado ({ind['con_mascara']}). Vuelve a ejecutar el notebook 02.")

    st.subheader("Estado del pipeline")
    disponibles = {}
    try:
        from src import estimador
        disponibles = estimador.archivos_de_modelos()
    except Exception:
        pass
    ocr_res = leer_json("output/tables/ocr_resumen.json") or {}
    link = (RAIZ / "video/link.txt").read_text(encoding="utf-8").strip() if (RAIZ / "video/link.txt").exists() else ""
    ui.pasos([
        ("1 · Muestreo", (RAIZ / "data/sample_ids.csv").exists(), f"semilla {run.get('seed', '—')}, sin foto"),
        ("2 · OCR", bool(ocr_res), f"{ocr_res.get('cvs_procesados', '—')} CVs · {ocr_res.get('imagenes_fallidas', '—')} fallos"),
        ("3 · Limpieza y máscara", dataset is not None, "texto OCR → features"),
        ("4 · Modelos y evaluación", (RAIZ / "output/tables/model_comparison.csv").exists(), "test n = "
         + (str(len(pred_test)) if pred_test is not None else "—")),
        ("5 · Pesos para la app", bool(disponibles) and all(disponibles.values()),
         f"{sum(disponibles.values())}/5 en models/" if disponibles else "models/ no revisado"),
        ("6 · Video", "http" in link and "PENDIENTE" not in link.upper(), "enlace en video/link.txt"),
    ])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Objetivo")
        st.markdown(
            "Estimar los **años de experiencia laboral** a partir de la **imagen** de un CV. No basta con buscar una "
            "frase: hay que leer bien las fechas de cada empleo, distinguirlas de las de estudios, sumar los rangos "
            "sin contar dos veces los solapamientos y tolerar errores del OCR.")
        st.subheader("Target")
        st.markdown(
            f"`exp_year` de PopResume, recortado a 0–{TOPE}. Regla del enunciado: subir el tope solo si más del 15 % "
            "de la muestra queda en él"
            + (f"; aquí queda el {pct(run['pct_en_tope_20'])}, así que se mantiene en {TOPE}." if "pct_en_tope_20" in run
               else ".")
            + f" Bandas para la evaluación por clases: {', '.join(BANDAS)} años.")
    with c2:
        st.subheader("Datos y muestreo")
        st.markdown(
            "**PopResume** (`sumin-yu/PopResume`, CC BY 4.0): 60 884 CVs sintéticos de 5 ocupaciones de EE. UU. "
            f"Muestra estratificada de **{num(config.N_POR_OCUPACION, 0)} CVs por ocupación** (asignación proporcional "
            f"por banda de experiencia), semilla **{run.get('seed', config.SEED)}**, solo imágenes **sin foto** "
            "(`has_photo == False`). Los IDs muestreados están en `data/sample_ids.csv` y la partición 70/15/15 "
            "en `data/splits/`.")

    st.subheader("Enmascaramiento y privacidad")
    st.markdown(
        "Las features salen **solo del OCR de la imagen**; la columna `text` del dataset se usa únicamente para medir "
        "el CER/WER. Antes de modelar se leyeron 50 textos OCR (`output/ocr_muestra_50.txt`) y se aplicaron dos filtros:\n"
        "- **Fugas del target → `[MASK]`**: experiencia total declarada («12 years of experience», «Experience: 7 yrs»), "
        "edad y fecha de nacimiento. Las fechas de cada empleo y los años de graduación **se conservan**: son la señal "
        "legítima. En PopResume ninguna plantilla declara la experiencia total, por eso el conteo es "
        + (f"**{ind['con_mascara']}**." if ind else "el que indique el dataset procesado.") + "\n"
        "- **Datos personales**: se eliminan nombre, correo, teléfono, URL, dirección y estado (`TX, USA`).\n\n"
        "Nunca se usan `age`, `sex`, `race`, nombres, `state` ni `region` como features; `sex` y `race` se usan "
        "solo para la comprobación de equidad (P4).")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("OCR y orden de lectura")
        st.markdown(
            f"PaddleOCR {ocr_res.get('paddleocr', '3.x')} con `{config.OCR_DET}` + `{config.OCR_REC}`, ejecutado en "
            f"{ocr_res.get('hardware', 'GPU')} en Colab; la app usa el mismo modelo en CPU. Orden de lectura: se busca "
            "un canal vertical sin cajas; si existe, se lee la columna izquierda completa y luego la derecha, por franjas "
            "separadas por los títulos que cruzan la página. El OCR pega las fechas (`2019.112023.01`) y la limpieza "
            "las separa de nuevo. Umbral de confianza elegido minimizando el CER: "
            f"**{run.get('umbral_confianza', '—')}**.")
    with c2:
        st.subheader("Decisiones metodológicas")
        st.markdown(
            "- Hiperparámetros y *early stopping* elegidos en **validación**; **test** solo se usa para el informe final.\n"
            "- Misma partición para todos los modelos (estratificada por ocupación × banda).\n"
            "- MAE como métrica principal; IC 95 % por *bootstrap* pareado (2 000 remuestreos) para la ganancia de BERT.\n"
            f"- Las predicciones se recortan a 0–{TOPE} antes de evaluar.")

    st.subheader("Modelos")
    bert_cfg = leer_json("output/tables/bert_config.json") or {}
    hp = regla_cfg
    tf, em = hp.get("tfidf", {}), hp.get("embeddings", {})
    filas_modelos = [
        ("0 · Mediana", "—", "Mediana de `y` en train: el piso que cualquier modelo debe superar."),
        ("1 · TF-IDF + Ridge", "Texto OCR limpio",
         f"n-gramas de palabras {tf.get('ngram_palabras', '—')} y caracteres {tf.get('ngram_caracteres', '—')}, "
         f"con dígitos, α = {tf.get('alpha', '—')} (validación)."),
        ("2 · Embeddings + Ridge", "Texto OCR limpio",
         f"{str(em.get('encoder', '—')).split('/')[-1]} congelado, trozos de {em.get('trozo_palabras', '—')} palabras "
         f"promediados, α = {em.get('alpha', '—')}."),
        ("3 · DistilBERT (fine-tuned)", "Texto OCR limpio",
         f"Regresor sobre `{bert_cfg.get('modelo_base', 'distilbert-base-uncased')}`, pérdida "
         f"{str(bert_cfg.get('perdida', 'mse')).upper()}, target ÷ {TOPE}, {bert_cfg.get('max_len', 512)} tokens "
         f"(cabeza + cola), lr {bert_cfg.get('lr', '—')}, early stopping por MAE de validación."),
        ("Bonus · Regla de fechas", "Texto OCR limpio",
         "Suma de los rangos bajo un título de experiencia, fusionando solapamientos; excluye fechas de estudios, "
         "actividades y cursos. No es uno de los 4 modelos."),
    ]
    # Tabla Markdown (no st.dataframe): así la columna de configuración se lee completa en cualquier ancho
    st.markdown("| Modelo | Entrada | Configuración |\n|---|---|---|\n"
                + "\n".join(f"| **{a}** | {b} | {c} |" for a, b, c in filas_modelos))

    st.subheader("Limitaciones")
    st.markdown(
        "- CVs **sintéticos** generados con plantillas: el formato de fechas es muy regular, por eso la regla de fechas "
        "es casi perfecta y los resultados no se trasladan sin más a CVs reales.\n"
        "- 5 ocupaciones, estadísticas de EE. UU. y solo inglés: en un CV en español o con otro formato las "
        "estimaciones son orientativas.\n"
        "- Todos los CVs de PopResume tienen historial laboral: los modelos nunca vieron un perfil **sin experiencia** "
        "(estudiante o egresado) y pueden sobreestimarlo si lista actividades o cursos con fechas. La regla de fechas "
        "solo cuenta rangos bajo un título de experiencia y la app avisa cuando el CV no lo tiene.\n"
        "- El estimador sirve para **pre-llenar o verificar** un campo, nunca para decidir una contratación.")

# ════════════════════════════════════════════════════════════════════════════════
# 2 · OCR Y ANÁLISIS EXPLORATORIO
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if dataset is None:
        ui.falta("data/processed/dataset.parquet", COMO_COLAB)
    else:
        ind = indicadores(dataset)
        st.subheader("Distribución de los años de experiencia")
        base = dataset[["y", "ocupacion"]]
        c1, c2 = st.columns([5, 6], gap="large")
        with c1:
            hist = alt.Chart(base).mark_bar(color=ui.AZUL, opacity=.85).encode(
                x=alt.X("y:Q", bin=alt.Bin(step=1, extent=[0, TOPE + 1]), title="años de experiencia"),
                y=alt.Y("count():Q", title="número de CVs"),
                tooltip=[alt.Tooltip("y:Q", bin=alt.Bin(step=1, extent=[0, TOPE + 1]), title="años"),
                         alt.Tooltip("count():Q", title="CVs")]).properties(height=300)
            ui.grafico(hist)
        with c2:
            orden = base.groupby("ocupacion").y.median().sort_values().index.tolist()
            caja = alt.Chart(base).mark_boxplot(extent=1.5, color=ui.AZUL, size=18, opacity=.85).encode(
                x=alt.X("y:Q", title="años de experiencia", scale=alt.Scale(domain=[0, TOPE])),
                y=alt.Y("ocupacion:N", title=None, sort=orden, axis=alt.Axis(labelLimit=220))).properties(height=300)
            ui.grafico(caja)
        ui.nota(f"<b>Histograma</b> (barras de 1 año): muestra la forma completa de la distribución; la experiencia se "
                f"concentra entre 2 y 8 años y hay <b>{ind['en_tope']} CVs ({pct(ind['en_tope'] / ind['n'])}) en el tope "
                f"de {TOPE}</b>. <b>Diagrama de caja</b> por ocupación (ordenado por mediana): compara medianas y "
                "dispersión de un vistazo; las diferencias entre ocupaciones ya dan algo de señal a cualquier modelo.")
        tabla = pd.crosstab(dataset.ocupacion, pd.Categorical(dataset.banda.map(dict(enumerate(BANDAS))),
                                                              categories=BANDAS), margins=True, margins_name="Total")
        tabla.index.name, tabla.columns.name = "Ocupación", None
        tabla.columns = [f"{c} años" if c != "Total" else c for c in tabla.columns]
        with st.expander("CVs por ocupación y banda de experiencia (muestreo estratificado)"):
            st.dataframe(tabla, width="stretch")

        st.subheader("Calidad del OCR")
        ui.kpis([
            ("CER medio", pct(dataset.CER.mean(), 2), f"mediana {pct(dataset.CER.median(), 2)}"),
            ("WER medio", pct(dataset.WER.mean(), 2), f"mediana {pct(dataset.WER.median(), 2)}"),
            ("Años de la referencia leídos", pct(dataset.recall_anios.mean()),
             f"{pct((dataset.recall_anios == 1).mean())} de los CVs con todos sus años"),
            ("Confianza media", num(dataset.conf_media.mean(), 3), "score de reconocimiento"),
            ("Tiempo por CV", f"{num(ocr_res['segundos_por_cv_media'], 2)} s" if "segundos_por_cv_media" in ocr_res
             else None, ocr_res.get("hardware", "")),
        ])
        err = dataset[["ocupacion", "CER", "WER"]].melt("ocupacion", var_name="métrica", value_name="v")
        err["v"] *= 100
        caja_err = alt.Chart(err).mark_boxplot(extent=1.5, size=14).encode(
            x=alt.X("v:Q", title="error (%)"),
            y=alt.Y("ocupacion:N", title=None, sort=orden, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("métrica:N", scale=alt.Scale(domain=["CER", "WER"], range=[ui.AZUL, "#D9762B"]),
                            legend=None)).properties(height=230)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**CER** · error por carácter")
            ui.grafico(caja_err.transform_filter(alt.datum["métrica"] == "CER"))
        with c2:
            st.markdown("**WER** · error por palabra")
            ui.grafico(caja_err.transform_filter(alt.datum["métrica"] == "WER"))
        ui.nota("CER y WER comparan el OCR con el texto original `text` (que solo se usa para medir). Se eligen diagramas "
                "de caja por ocupación para ver si alguna plantilla se lee peor. El WER es mayor que el CER porque un "
                "solo carácter mal leído invalida la palabra completa.")

        st.subheader("Lectura de fechas")
        umb = leer_csv("output/tables/umbral_confianza.csv")
        ui.kpis([
            ("CVs con fechas pegadas", f"{num(ind['fechas_pegadas'], 0)} ({pct(ind['fechas_pegadas'] / ind['n'], 0)})",
             "«2019.112023.01»: el OCR pierde la raya"),
            ("Texto final sin fechas pegadas", f"{num(int((~dataset.texto_final.map(lambda t: bool(texto.RE_FECHAS_PEGADAS.search(t)))).sum()), 0)} / {num(ind['n'], 0)}",
             "CVs con todas sus fechas separadas tras la limpieza"),
            ("Años de la referencia leídos", pct(dataset.recall_anios.mean()), "señal clave del target"),
        ])
        if umb is not None and umb.CER.nunique() == 1:
            ui.nota(f"El umbral de confianza no cambia nada: el CER es {pct(umb.CER.iloc[0], 2)} con cualquier umbral "
                    f"entre {umb.umbral.min()} y {umb.umbral.max()} porque ninguna línea baja de esa confianza; se usa "
                    f"{run.get('umbral_confianza', 0)} (no descartar texto).")
        elif umb is not None:
            ui.grafico(alt.Chart(umb).mark_line(point=True, color=ui.AZUL).encode(
                x=alt.X("umbral:Q", title="umbral de confianza"), y=alt.Y("CER:Q", title="CER", axis=alt.Axis(format="%")))
                .properties(height=220))

        st.subheader("Ejemplo: imagen con cajas de PaddleOCR y texto extraído")
        ej = leer_json("output/tables/ocr_ejemplo.json")
        fig_ej = RAIZ / "output/figures/ocr_ejemplo_cajas.png"
        c1, c2 = st.columns(2, gap="large")
        with c1:
            if fig_ej.exists() and ej and ej.get("anonimizado"):
                st.image(str(fig_ej), width="stretch")
                ui.nota("Cada caja es una línea detectada; el número es su orden de lectura y el color su confianza "
                        "(verde ≥ 0,9). Las barras oscuras ocultan el nombre y los datos de contacto.")
            elif fig_ej.exists():
                st.warning("La figura de ejemplo no está marcada como anonimizada; no se muestra.")
            else:
                ui.falta("output/figures/ocr_ejemplo_cajas.png", COMO_COLAB)
        with c2:
            if ej:
                from src import anonimizar
                vista = st.segmented_control("Texto", ["Salida de PaddleOCR", "Texto final para los modelos"],
                                             default="Salida de PaddleOCR", key="vista_ejemplo")
                fila = dataset.loc[dataset.id == ej["id"]]
                if vista == "Texto final para los modelos" and len(fila):
                    st.code(fila.texto_final.iloc[0], language=None, wrap_lines=True)
                else:
                    st.code(anonimizar.anonimizar_texto(ej["texto"]), language=None, wrap_lines=True)
                ui.nota("La salida del OCR, ya en orden de lectura y con las fechas reparadas, frente al texto que "
                        "reciben los modelos: sin datos personales ni estado y con las viñetas normalizadas.")

# ════════════════════════════════════════════════════════════════════════════════
# 3 · RESULTADOS DE MODELOS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    comp = leer_csv("output/tables/model_comparison.csv")
    if comp is None or pred_test is None:
        ui.falta("output/tables/model_comparison.csv y predicciones_test.csv", COMO_COLAB)
    else:
        n_test = len(pred_test)
        clave = {v: k for k, v in ETIQUETA.items()}
        comp["clave"] = comp.modelo.map(clave)
        comp = comp.set_index("clave").loc[[m for m in MODELOS if m in set(comp.modelo.map(clave))]].reset_index()
        st.info(f"Todas las métricas de esta pestaña son de la partición de **test** (n = {n_test} CVs), que no se usó "
                "para entrenar ni para elegir hiperparámetros.", icon=":material/info:")
        seccion = st.segmented_control("Sección", ["Comparación", "Errores", "Preguntas P1–P4"],
                                       default="Comparación", key="seccion_resultados") or "Comparación"

        if seccion == "Comparación":
            st.subheader("Comparación de modelos")
            req = comp[comp.clave.isin(REQUERIDOS)]
            mejor_req = req.loc[req.MAE.idxmin()]
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                barras = alt.Chart(comp.assign(tipo=np.where(comp.clave.isin(REQUERIDOS), "Requerido", "Bonus"))).mark_bar(
                ).encode(
                    x=alt.X("MAE:Q", title="MAE en test (años)"),
                    y=alt.Y("modelo:N", title=None, sort=list(comp.modelo), axis=alt.Axis(labelLimit=220)),
                    color=alt.Color("modelo:N", scale=ui.ESCALA_MODELOS, legend=None),
                    opacity=alt.condition(alt.datum.tipo == "Bonus", alt.value(.55), alt.value(1)),
                    tooltip=["modelo", alt.Tooltip("MAE:Q", format=".3f"), alt.Tooltip("RMSE:Q", format=".3f")])
                etiquetas = barras.mark_text(align="left", dx=4, color="#1B2430").encode(
                    text=alt.Text("MAE:Q", format=".2f"), opacity=alt.value(1), color=alt.value("#1B2430"))
                ui.grafico((barras + etiquetas).properties(height=210))
            with c2:
                ui.nota(f"Entre los <b>4 modelos del enunciado</b>, el menor MAE es el de <b>{mejor_req.modelo}</b> "
                        f"({num(mejor_req.MAE, 3)} años). La regla de fechas (bonus, barra atenuada) es aún más exacta "
                        f"({num(comp.set_index('clave').MAE.get('regla_fechas', np.nan), 3)}): en PopResume la experiencia "
                        "es casi exactamente la suma de los rangos de fechas. El IC de las diferencias está más abajo.")
            tabla = comp.drop(columns=["clave"]).rename(columns={
                "modelo": "Modelo", "acc_±1": "±1 año", "acc_±2": "±2 años", "acc_±3": "±3 años",
                "acc_banda": "Exactitud banda", "macroF1_banda": "Macro-F1 banda", "R2": "R²", "sesgo_medio": "Sesgo medio",
                "tiempo_entrenamiento_s": "Entren. (s)", "tiempo_inferencia_ms_por_CV": "Infer. (ms/CV)"})
            filas_req = tabla.Modelo.isin([ETIQUETA[m] for m in REQUERIDOS])
            menor = ["MAE", "RMSE"]
            mayor = ["±1 año", "±2 años", "±3 años", "Exactitud banda", "Macro-F1 banda", "R²"]

            def resaltar(col):
                if col.name not in menor + mayor:
                    return [""] * len(col)
                ref = col[filas_req].min() if col.name in menor else col[filas_req].max()
                return ["font-weight:700; background-color:#FCEBDD" if r and v == ref else ""
                        for r, v in zip(filas_req, col)]

            fmt = {c: "{:.3f}" for c in ["MAE", "RMSE", "R²", "Sesgo medio"]}
            fmt.update({c: "{:.1%}" for c in mayor if c != "R²"})
            fmt.update({"Entren. (s)": "{:.1f}", "Infer. (ms/CV)": "{:.2f}"})
            st.dataframe(tabla.style.apply(resaltar).format(fmt), hide_index=True, width="stretch")
            ui.nota("Resaltado: mejor valor entre los 4 modelos requeridos. MAE = años de error promedio (métrica "
                    "principal); ±k = % de CVs con error ≤ k años; bandas = " + ", ".join(BANDAS) + " años. Sesgo medio = "
                    "predicho − real. Tiempos medidos en Colab (BERT en GPU T4; inferencia por CV).")

            st.subheader("Predicho frente a real")
            m = st.selectbox("Modelo", MODELOS, format_func=ETIQUETA.get, index=MODELOS.index("bert"), key="m_scatter")
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                dfp = pred_test[["y", f"p_{m}", "ocupacion"]].rename(columns={f"p_{m}": "p"})
                media = dfp.assign(yr=dfp.y.round()).groupby("yr", as_index=False).p.mean()
                puntos = alt.Chart(dfp).mark_circle(size=22, opacity=.45, color=ui.COLOR[m]).encode(
                    x=alt.X("y:Q", title="años reales", scale=alt.Scale(domain=[0, TOPE])),
                    y=alt.Y("p:Q", title="años predichos", scale=alt.Scale(domain=[0, TOPE])),
                    tooltip=[alt.Tooltip("y:Q", format=".1f", title="real"), alt.Tooltip("p:Q", format=".1f",
                                                                                          title="predicho"), "ocupacion"])
                diag = alt.Chart(pd.DataFrame({"y": [0, TOPE], "p": [0, TOPE]})).mark_line(
                    color=ui.GRIS, strokeDash=[5, 4]).encode(x="y:Q", y="p:Q")
                linea = alt.Chart(media).mark_line(color="#14263D", strokeWidth=2).encode(x="yr:Q", y="p:Q")
                ui.grafico((puntos + diag + linea).properties(height=360))
                ui.nota("Línea discontinua: 45° (predicción perfecta). Línea oscura: predicción media por año real. "
                        "Si es más plana que la diagonal, el modelo comprime hacia la media.")
            with c2:
                mat = leer_json("output/tables/matrices_confusion.json")
                if mat and m in mat.get("matrices", {}):
                    M = np.array(mat["matrices"][m])
                    dfm = pd.DataFrame([(BANDAS[i], BANDAS[j], int(M[i, j]), M[i, j] / max(M[i].sum(), 1))
                                        for i in range(len(M)) for j in range(len(M))],
                                       columns=["real", "predicha", "n", "fila"])
                    base_m = alt.Chart(dfm).encode(x=alt.X("predicha:N", sort=BANDAS, title="banda predicha"),
                                                   y=alt.Y("real:N", sort=BANDAS, title="banda real"))
                    calor = base_m.mark_rect().encode(color=alt.Color("fila:Q", scale=alt.Scale(scheme="blues",
                                                                                                domain=[0, 1]), legend=None),
                                                      tooltip=["real", "predicha", "n"])
                    txt = base_m.mark_text(fontSize=13).encode(
                        text="n:Q", color=alt.condition(alt.datum.fila > .5, alt.value("white"), alt.value("#1B2430")))
                    ui.grafico((calor + txt).properties(height=300))
                    ui.nota("Matriz de confusión por bandas: la diagonal son aciertos de banda; el color es la proporción "
                            "de cada fila.")
                else:
                    ui.falta("output/tables/matrices_confusion.json", COMO_COLAB)

            st.subheader("MAE por banda real y por ocupación")
            con_mediana = st.toggle("Incluir la mediana (su error es mucho mayor y aplana la escala)", value=False)
            mods = [ETIQUETA[k] for k in MODELOS if con_mediana or k != "mediana"]
            c1, c2 = st.columns(2, gap="large")
            for col, rel, eje, orden_eje, titulo in (
                    (c1, "output/tables/mae_por_banda.csv", "banda real (años)", BANDAS, "Por banda de experiencia"),
                    (c2, "output/tables/mae_por_ocupacion.csv", "ocupación", None, "Por ocupación")):
                t = leer_csv(rel, index_col=0)
                with col:
                    st.markdown(f"**{titulo}**")
                    if t is None:
                        ui.falta(rel, COMO_COLAB)
                        continue
                    larga = t.loc[[x for x in mods if x in t.index]].reset_index(names="modelo").melt(
                        "modelo", var_name="grupo", value_name="MAE")
                    if orden_eje is None:
                        larga["grupo"] = larga.grupo.map(OCUPACION).fillna(larga.grupo)
                    ui.grafico(alt.Chart(larga).mark_bar().encode(
                        y=alt.Y("grupo:N", title=None, sort=orden_eje, axis=alt.Axis(labelLimit=200)),
                        yOffset=alt.YOffset("modelo:N", sort=mods),
                        x=alt.X("MAE:Q", title="MAE (años)"),
                        color=alt.Color("modelo:N", scale=ui.escala(mods), title=None),
                        tooltip=["modelo", "grupo", alt.Tooltip("MAE:Q", format=".3f")]).properties(height=330))
            ui.nota("Barras agrupadas para comparar todos los modelos en cada grupo con la misma escala. Las tablas "
                    "completas están debajo.")
            with st.expander("Tablas de MAE por banda y por ocupación"):
                for rel in ("output/tables/mae_por_banda.csv", "output/tables/mae_por_ocupacion.csv"):
                    t = leer_csv(rel, index_col=0)
                    if t is not None:
                        st.dataframe(t.rename(columns=OCUPACION).style.format("{:.3f}"), width="stretch")

            st.subheader("Ganancia de BERT sobre cada baseline (IC 95 %)")
            gan = leer_csv("output/tables/ganancia_bert_ic.csv")
            if gan is None:
                ui.falta("output/tables/ganancia_bert_ic.csv", COMO_COLAB)
            else:
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    b = alt.Chart(gan).encode(y=alt.Y("baseline:N", title=None, sort=list(gan.baseline),
                                                      axis=alt.Axis(labelLimit=200)))
                    ui.grafico((b.mark_rule(strokeWidth=2, color=ui.AZUL).encode(
                        x=alt.X("IC95_inf:Q", title="MAE(baseline) − MAE(BERT)  [años]"), x2="IC95_sup:Q")
                        + b.mark_point(filled=True, size=70, color=ui.AZUL).encode(
                            x="ganancia_MAE:Q", tooltip=["baseline", alt.Tooltip("ganancia_MAE:Q", format=".3f"),
                                                         alt.Tooltip("IC95_inf:Q", format=".3f"),
                                                         alt.Tooltip("IC95_sup:Q", format=".3f")])
                        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=ui.ROJO).encode(x="x:Q"))
                        .properties(height=200))
                with c2:
                    ui.nota(f"<i>Bootstrap</i> pareado sobre los {n_test} CVs de test ({int(gan.n_boot.iloc[0])} remuestreos). "
                            "A la derecha de la línea roja BERT es mejor; si el intervalo no la toca, la diferencia es "
                            "significativa al 95 %. Frente a la regla de fechas (bonus) la diferencia es negativa: la "
                            "regla es mejor.")
                t = gan.rename(columns={"baseline": "Baseline", "ganancia_MAE": "Ganancia de MAE", "IC95_inf": "IC 95 % inf.",
                                        "IC95_sup": "IC 95 % sup.", "significativa": "Significativa", "n_boot": "Remuestreos"})
                t["Significativa"] = t["Significativa"].map({True: "Sí", False: "No"})
                st.dataframe(t.style.format({c: "{:+.3f}" for c in ["Ganancia de MAE", "IC 95 % inf.", "IC 95 % sup."]}),
                             hide_index=True, width="stretch")

        elif seccion == "Errores":
            st.subheader("Los 10 mayores errores de BERT, por causa")
            top = leer_csv("output/tables/top10_errores_bert.csv", dtype={"id": str})
            if top is None:
                ui.falta("output/tables/top10_errores_bert.csv", COMO_COLAB)
            else:
                top["id"] = top.id.str.zfill(5)
                c1, c2 = st.columns([2, 3], gap="large")
                with c1:
                    causas = top.causa_sugerida.value_counts().rename_axis("causa").reset_index(name="CVs")
                    ui.grafico(alt.Chart(causas).mark_bar(color=ui.AZUL).encode(
                        x=alt.X("CVs:Q", title="CVs", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("causa:N", title=None, sort="-x")).properties(height=140))
                    ui.nota("Causas del enunciado: fecha mal leída, orden de lectura, hueco laboral, empleos solapados, "
                            "fechas ausentes, truncamiento u «otra». Con CER ≈ 1 % y todos los años leídos, ningún error "
                            "grande se debe al OCR; «otra» significa que el texto estaba bien y el error es del modelo.")
                with c2:
                    vista = top.assign(ocupacion=top.job.map(OCUPACION))[
                        ["id", "ocupacion", "y", "p_bert", "err_bert", "regla_estima", "CER", "causa_sugerida"]]
                    st.dataframe(vista, hide_index=True, width="stretch", column_config={
                        "id": "ID", "ocupacion": "Ocupación", "y": st.column_config.NumberColumn("Real", format="%.2f"),
                        "p_bert": st.column_config.NumberColumn("BERT", format="%.2f"),
                        "err_bert": st.column_config.NumberColumn("|Error|", format="%.2f"),
                        "regla_estima": st.column_config.NumberColumn("Regla", format="%.2f"),
                        "CER": st.column_config.NumberColumn("CER", format="%.3f"), "causa_sugerida": "Causa"})

            st.subheader("¿Afecta la calidad del OCR? MAE por cuartil de CER")
            q2c = leer_csv("output/tables/q2_mae_por_cuartil_cer.csv", index_col=0)
            if q2c is not None:
                larga = q2c.loc[[ETIQUETA[k] for k in MODELOS if k != "mediana" and ETIQUETA[k] in q2c.index]] \
                    .reset_index(names="modelo").melt("modelo", var_name="cuartil", value_name="MAE")
                ui.grafico(alt.Chart(larga).mark_line(point=True).encode(
                    x=alt.X("cuartil:N", title="cuartil de CER del CV (Q1 = mejor OCR)", sort=list(q2c.columns),
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("MAE:Q", title="MAE (años)"),
                    color=alt.Color("modelo:N", scale=ui.escala(set(larga.modelo)), title=None),
                    tooltip=["modelo", "cuartil", alt.Tooltip("MAE:Q", format=".3f")]).properties(height=260))
                ui.nota("Si el ruido del OCR perjudicara, el MAE subiría de Q1 a Q4. Las líneas son casi planas: en estos "
                        "renders limpios el CER (≈ 1 %) no explica los errores.")

        else:
            st.subheader("P1 · ¿Las predicciones se mueven como las de un reclutador?")
            q1 = leer_csv("output/tables/q1_perturbaciones.csv")
            if q1 is None:
                ui.falta("output/tables/q1_perturbaciones.csv", COMO_COLAB)
            else:
                b = alt.Chart(q1).encode(y=alt.Y("perturbacion:N", title=None, axis=alt.Axis(labelLimit=200)))
                ui.grafico((b.mark_bar().encode(
                    x=alt.X("cambio_medio:Q", title="cambio medio de la predicción (años)"),
                    yOffset=alt.YOffset("modelo:N"),
                    color=alt.Color("modelo:N", scale=ui.escala(set(q1.modelo)), title=None),
                    tooltip=["perturbacion", "modelo", alt.Tooltip("cambio_medio:Q", format="+.2f"),
                             alt.Tooltip("esperado_medio:Q", format="+.2f"),
                             alt.Tooltip("misma_direccion:Q", format=".0%")])
                    + b.mark_tick(color="#14263D", thickness=3, size=46).encode(x="esperado_medio:Q"))
                    .properties(height=300))
                ui.nota("20 CVs de test con tres cambios controlados. A: todas las fechas +5 años (un reclutador no "
                        "cambiaría su estimación). B: el empleo más antiguo empieza 5 años antes (≈ +5). C: se borra el "
                        "empleo más antiguo (resta su duración). La marca negra es el cambio esperado.")
                with st.expander("Tabla de P1"):
                    st.dataframe(q1, hide_index=True, width="stretch")

            st.subheader("P2 · Ruido del OCR: texto OCR frente al texto de referencia")
            q2 = leer_csv("output/tables/q2_ruido_ocr.csv")
            if q2 is not None:
                st.dataframe(q2.rename(columns={"modelo": "Modelo", "MAE_OCR": "MAE con OCR",
                                                "MAE_referencia": "MAE con `text`", "brecha": "Brecha",
                                                "brecha_%": "Brecha (%)"}).style.format(
                    {"MAE con OCR": "{:.3f}", "MAE con `text`": "{:.3f}", "Brecha": "{:+.3f}", "Brecha (%)": "{:+.1f}"}),
                    hide_index=True, width="stretch")
                ui.nota("Mismo modelo evaluado con el texto OCR y con el texto original (`text`, solo para medir): una "
                        "brecha cercana a cero indica que el OCR casi no cuesta exactitud.")

            st.subheader("P3 · Sensibilidad a decisiones metodológicas")
            q3 = leer_csv("output/tables/q3_sensibilidad.csv")
            if q3 is None:
                ui.falta("output/tables/q3_sensibilidad.csv", COMO_COLAB)
            else:
                q3 = q3.assign(modelo=q3.modelo.map(ETIQUETA).fillna(q3.modelo)).rename(columns={
                    "modelo": "Modelo", "decision": "Decisión", "variante": "Variante", "MAE_test": "MAE test",
                    "min_entrenamiento": "Entrenamiento (min)"})
                st.dataframe(q3.style.format({"MAE test": "{:.3f}", "Entrenamiento (min)": "{:.1f}"}, na_rep="—"),
                             hide_index=True, width="stretch", height=36 * (len(q3) + 1) + 4)
                ui.nota("Cada fila cambia una sola decisión respecto de la configuración base. Lo que más pesa es quitar "
                        "los dígitos en TF-IDF: las fechas son la señal principal.")

            st.subheader("P4 · Comprobación de equidad (BERT)")
            q4 = leer_csv("output/tables/q4_equidad.csv")
            if q4 is None:
                ui.falta("output/tables/q4_equidad.csv", COMO_COLAB)
            else:
                q4["grupo_n"] = q4.grupo + " (n = " + q4.n.astype(str) + ")"
                q4["atributo"] = q4.atributo.map({"sex": "Sexo", "race": "Raza"}).fillna(q4.atributo)
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    b = alt.Chart(q4).encode(y=alt.Y("grupo_n:N", title=None, sort=list(q4.grupo_n),
                                                     axis=alt.Axis(labelLimit=220)))
                    ui.grafico((b.mark_rule(strokeWidth=2).encode(
                        x=alt.X("IC95_inf:Q", title="MAE de BERT (años) con IC 95 %"), x2="IC95_sup:Q",
                        color=alt.Color("atributo:N", title=None, scale=alt.Scale(range=[ui.AZUL, "#D9762B"])))
                        + b.mark_point(filled=True, size=70).encode(
                            x="MAE:Q", color="atributo:N",
                            tooltip=["atributo", "grupo", "n", alt.Tooltip("MAE:Q", format=".3f"),
                                     alt.Tooltip("error_medio:Q", format="+.3f"),
                                     alt.Tooltip("y_real_medio:Q", format=".2f")])).properties(height=230))
                with c2:
                    ui.nota("`sex` y `race` no son features: solo se usan aquí para auditar. Los intervalos de sexo se "
                            "superponen casi por completo. El grupo con mayor MAE tiene pocos CVs (IC ancho) y más "
                            "experiencia media; con n tan pequeño no es una conclusión firme.")
                st.dataframe(q4.drop(columns=["grupo_n"]), hide_index=True, width="stretch", column_config={
                    "atributo": "Atributo", "grupo": "Grupo", "n": "n",
                    **{c: st.column_config.NumberColumn(c.replace("_", " "), format="%.3f")
                       for c in ["MAE", "IC95_inf", "IC95_sup", "error_medio", "y_real_medio"]}})

# ════════════════════════════════════════════════════════════════════════════════
# 4 · ESTIMADOR INTERACTIVO
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def ocr_cpu():
    from src import ocr
    return ocr.crear_ocr(device="cpu")


@st.cache_resource(show_spinner=False)
def cargar_modelos():
    from src import estimador
    return estimador.cargar_predictores(TOPE)


def explicar_ejemplo(r, bueno):
    """Explicación breve de un acierto o un error de BERT a partir de cantidades reales del CV."""
    n_rangos = len(reglas.detalle(r.texto_final, ANIO_REGLA or 2015))
    base = (f"{n_rangos} rango(s) laboral(es) leídos; la suma de fechas (regla) da {num(r.p_regla_fechas, 1)} años "
            f"frente a {num(r.y, 1)} reales; CER del OCR {pct(r.CER)}.")
    if bueno:
        return "Las fechas se leyeron completas y BERT reproduce la suma de rangos. " + base
    causa = getattr(r, "causa", None)
    origen = ("el texto estaba bien leído, así que el error es del modelo, no del OCR"
              if abs(r.p_regla_fechas - r.y) < .25 else "el texto tenía problemas de lectura")
    return (f"BERT se desvía {num(r.p_bert - r.y, 1)} años; {origen}. "
            + (f"Causa en el análisis de errores: **{causa}**. " if isinstance(causa, str) else "") + base)


with tab4:
    modo = st.segmented_control("Modo", ["Analizar un CV", "Explorar CVs de test"], default="Analizar un CV",
                                key="modo_estimador") or "Analizar un CV"

    if modo == "Analizar un CV":
        from src import estimador
        st.subheader("Sube un CV en imagen o PDF")
        disp = estimador.archivos_de_modelos()
        faltan = [ETIQUETA[k] for k, v in disp.items() if not v]
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            archivo = st.file_uploader("Archivo del CV (PNG, JPG o PDF; se usa la primera página)",
                                       type=["png", "jpg", "jpeg", "pdf"], key="cv_subido")
            boton = st.button("Leer y estimar", type="primary", disabled=archivo is None,
                              icon=":material/document_scanner:")
        with c2:
            ui.nota("<b>Privacidad.</b> El archivo se procesa en memoria; la copia temporal para el OCR se borra al "
                    "terminar y no se guarda nada. El nombre, el contacto y los identificadores se ocultan en el texto, "
                    "y la imagen se muestra <b>reconstruida solo con las líneas de texto no personales</b>: fotos, firmas "
                    "y texto no detectado nunca aparecen.")
            if faltan:
                st.warning(f"Faltan pesos en `models/` para: {', '.join(faltan)}. Esos modelos no se mostrarán. "
                           "Cópialos desde Drive (README, sección del estimador).")
            ui.nota("La primera vez se cargan PaddleOCR y los modelos en CPU (1–2 min); después, 10–60 s por CV.")

        if boton and archivo is not None:
            st.session_state.pop("resultado", None)
            try:
                with st.status("Procesando el CV…", expanded=False) as estado:
                    estado.update(label="Cargando PaddleOCR (CPU)…")
                    motor = ocr_cpu()
                    estado.update(label="Leyendo el CV con PaddleOCR…")
                    lect = estimador.procesar(motor, archivo.getvalue(), archivo.name, run.get("umbral_confianza", 0.0))
                    preds = {}
                    if lect.texto_modelo.strip():
                        estado.update(label="Estimando con los modelos…")
                        try:
                            preds = estimador.predecir(cargar_modelos(), lect.texto_modelo)
                        except Exception as e:                  # nunca se muestra texto del CV
                            preds = {"_errores": [f"carga de modelos: {type(e).__name__}"]}
                    estado.update(label="Listo", state="complete")
                st.session_state["resultado"] = (lect, preds)
            except estimador.ArchivoInvalido as e:
                st.error(str(e), icon=":material/error:")
            except Exception as e:
                st.error(f"No se pudo leer el CV ({type(e).__name__}). Prueba con una imagen más nítida o con un PDF.",
                         icon=":material/error:")

        if "resultado" in st.session_state:
            lect, preds = st.session_state["resultado"]
            errores = preds.get("_errores")
            preds = {k: v for k, v in preds.items() if k != "_errores"}
            if errores:
                st.warning("Algunos modelos no se pudieron ejecutar: " + "; ".join(errores))
            if not lect.lineas:
                st.warning("PaddleOCR no detectó texto en el archivo. Prueba con una imagen más nítida, recta y con "
                           "buena iluminación.")
            else:
                c1, c2 = st.columns([5, 7], gap="large")
                with c1:
                    st.image(lect.imagen, width="stretch")
                    ui.nota(f"Vista protegida: {len(lect.lineas)} líneas detectadas, <b>{lect.n_personales} ocultas</b> "
                            "por contener datos personales (barras oscuras). Contorno = confianza del OCR (verde ≥ 0,9, "
                            "ámbar ≥ 0,7, rojo < 0,7); número = orden de lectura"
                            + (" (se detectaron dos columnas)." if lect.dos_columnas else "."))
                with c2:
                    seccion_laboral = reglas.tiene_seccion_laboral(lect.texto_modelo)
                    if seccion_laboral is False:
                        st.warning("**El CV no tiene una sección de experiencia laboral** (p. ej. «Work experience», "
                                   "«Experiencia laboral», «Prácticas»). Puede tratarse de un perfil sin experiencia, como "
                                   "un estudiante o egresado. La regla de fechas da 0 porque no cuenta actividades, cursos "
                                   "ni estudios; los otros modelos se entrenaron solo con CVs con historial laboral y pueden "
                                   "sobreestimar al ver fechas de actividades extracurriculares o cursos.",
                                   icon=":material/work_off:")
                    req = {k: v for k, v in preds.items() if k in REQUERIDOS}
                    if not req:
                        st.info("No hay predicciones: faltan los modelos o el CV no tiene texto utilizable.")
                    else:
                        principal = "bert" if "bert" in req else min(req, key=REQUERIDOS.index)
                        rango = (min(req.values()), max(req.values()))
                        ui.estimacion(
                            f"Estimación principal · {ETIQUETA[principal]}", f"{num(req[principal], 1)} años",
                            f"Rango entre los {len(req)} modelos: {num(rango[0], 1)}–{num(rango[1], 1)} años. "
                            + ("DistilBERT es el de menor MAE en test entre los 4 modelos requeridos."
                               if principal == "bert" else "DistilBERT no está disponible."))
                        dfp = pd.DataFrame([(ETIQUETA[k], v) for k, v in preds.items() if k in ETIQUETA],
                                           columns=["modelo", "años"])
                        b = alt.Chart(dfp).encode(y=alt.Y("modelo:N", title=None, sort=[ETIQUETA[m] for m in MODELOS],
                                                          axis=alt.Axis(labelLimit=220)))
                        ui.grafico((b.mark_bar().encode(
                            x=alt.X("años:Q", title="años estimados", scale=alt.Scale(domain=[0, TOPE])),
                            color=alt.Color("modelo:N", scale=ui.ESCALA_MODELOS, legend=None))
                            + b.mark_text(align="left", dx=4).encode(x="años:Q", text=alt.Text("años:Q", format=".1f")))
                            .properties(height=190))

                    st.markdown("**Fechas laborales detectadas**")
                    rangos = reglas.detalle(lect.texto_modelo, ANIO_REGLA or 2015)
                    if rangos:
                        st.dataframe(pd.DataFrame(rangos)[["rango", "anios"]], hide_index=True, width="stretch",
                                     column_config={"rango": "Rango leído", "anios": st.column_config.NumberColumn(
                                         "Años", format="%.2f")})
                        ui.nota(f"Suma sin contar solapamientos (regla de fechas): <b>{num(preds.get('regla_fechas', reglas.estimar(lect.texto_modelo, ANIO_REGLA or 2015, TOPE)), 2)} años</b>. "
                                + ("Solo se cuentan los rangos bajo un título de experiencia y se excluyen las fechas de "
                                   "estudios. " if seccion_laboral else
                                   "No se reconoció ningún título de sección: se cuentan todos los rangos salvo los de "
                                   "estudios. ") + "«Present» se cuenta hasta "
                                f"{(ANIO_REGLA or 2015) + .5:.1f}".replace(".", ",")
                                + ", el año de referencia calibrado con PopResume.")
                    elif seccion_laboral is False:
                        ui.nota("Ninguna fecha cuenta como laboral: el CV no tiene sección de experiencia. Las fechas de "
                                "actividades, cursos, voluntariado o estudios no se suman.")
                    else:
                        ui.nota("No se detectaron rangos de fechas con un formato reconocible («2019.11–2023.01», "
                                "«Mar 2014 – Jun 2018», «2015 - Present»). La regla da 0 y los modelos se apoyan solo en "
                                "el vocabulario del CV.")

                    avisos = []
                    if req and rango[1] - rango[0] > 3:
                        avisos.append(f"Los modelos discrepan en {num(rango[1] - rango[0], 1)} años: la mediana ignora el "
                                      "texto y TF-IDF/embeddings dependen del vocabulario de PopResume.")
                    if "bert" in preds and "regla_fechas" in preds and rangos and abs(preds["bert"] - preds["regla_fechas"]) > 2:
                        avisos.append("BERT y la suma de fechas difieren en más de 2 años. En test la regla fue más exacta "
                                      "porque los CVs sintéticos siguen un formato fijo; en un CV real puede fallar si las "
                                      "fechas están en otro idioma o formato.")
                    avisos.append("Los modelos se entrenaron con CVs sintéticos en inglés de 5 ocupaciones: en otros CVs la "
                                  "estimación es orientativa.")
                    st.markdown("**Cómo interpretar las diferencias**")
                    st.markdown("\n".join(f"- {a}" for a in avisos))

                c1, c2 = st.columns(2, gap="large")
                with c1:
                    with st.expander("Texto extraído por PaddleOCR (anonimizado)"):
                        st.code(lect.texto_visible, language=None, wrap_lines=True)
                with c2:
                    with st.expander("Texto que reciben los modelos"):
                        st.code(lect.texto_modelo or "(vacío)", language=None, wrap_lines=True)
                        if sum(lect.mascaras.values()):
                            st.caption(f"Fragmentos enmascarados por revelar la respuesta: {sum(lect.mascaras.values())}.")
            if st.button("Borrar resultado", icon=":material/delete:"):
                st.session_state.pop("resultado", None)
                st.rerun()

    else:
        st.subheader("Explorar CVs de test")
        if dataset is None or pred_test is None:
            ui.falta("output/tables/predicciones_test.csv", COMO_COLAB)
        else:
            test = pred_test.merge(dataset[["id", "texto_final"]], on="id")
            top = leer_csv("output/tables/top10_errores_bert.csv", dtype={"id": str})
            causas = dict(zip(top.id.str.zfill(5), top.causa_sugerida)) if top is not None else {}
            test["causa"] = test.id.map(causas)
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            ocups = c1.multiselect("Ocupación", sorted(test.ocupacion.unique()), placeholder="Todas")
            anios = c2.slider("Años reales", 0.0, float(TOPE), (0.0, float(TOPE)), step=0.5)
            m = c3.selectbox("Error de", MODELOS, format_func=ETIQUETA.get, index=MODELOS.index("bert"))
            test["error"] = test[f"p_{m}"] - test.y
            emax = float(np.ceil(test.error.abs().max() * 10) / 10)
            rango_err = c4.slider("|Error| (años)", 0.0, emax, (0.0, emax), step=0.1)
            sel = test[(test.y.between(*anios)) & (test.error.abs().between(*rango_err))
                       & (test.ocupacion.isin(ocups) if ocups else True)].copy()
            sel = sel.reindex(sel.error.abs().sort_values(ascending=False).index)
            st.caption(f"{len(sel)} de {len(test)} CVs de test · ordenados de mayor a menor error de {ETIQUETA[m]}. "
                       "Selecciona una fila para ver el detalle.")
            if sel.empty:
                st.info("Ningún CV cumple los filtros.")
            else:
                elegido = st.dataframe(
                    sel[["id", "ocupacion", "y", f"p_{m}", "error"]], hide_index=True, width="stretch", height=240,
                    on_select="rerun", selection_mode="single-row", key="tabla_test",
                    column_config={"id": "ID", "ocupacion": "Ocupación",
                                   "y": st.column_config.NumberColumn("Real", format="%.2f"),
                                   f"p_{m}": st.column_config.NumberColumn("Predicho", format="%.2f"),
                                   "error": st.column_config.NumberColumn("Error", format="%+.2f")})
                filas = elegido.selection.rows if elegido and hasattr(elegido, "selection") else []
                fila = sel.iloc[filas[0]] if filas else sel.iloc[0]
                st.markdown(f"**CV {fila.id}** · {fila.ocupacion} · {num(fila.y, 2)} años reales")
                ui.kpis([("Real", num(fila.y, 1), "años")]
                        + [(ETIQUETA[k], num(fila[f"p_{k}"], 1),
                            f"error {fila[f'p_{k}'] - fila.y:+.1f}".replace(".", ","))
                           for k in MODELOS])
                with st.expander("Texto OCR que vieron los modelos (sin datos personales)", expanded=False):
                    st.code(fila.texto_final, language=None, wrap_lines=True)

            st.subheader("Buenas y malas predicciones de BERT")
            test["abs_bert"] = (test.p_bert - test.y).abs()
            c1, c2 = st.columns(2, gap="large")
            for col, titulo, bueno, filas_ej in (
                    (c1, "Aciertos (menor error)", True, test.nsmallest(3, "abs_bert")),
                    (c2, "Errores grandes", False, test.nlargest(3, "abs_bert"))):
                with col:
                    st.markdown(f"**{titulo}**")
                    for r in filas_ej.itertuples():
                        with st.expander(f"{r.id} · {r.ocupacion} · real {num(r.y, 1)} · BERT {num(r.p_bert, 1)}"):
                            st.markdown(explicar_ejemplo(r, bueno))
                            st.code(r.texto_final[:1200], language=None, wrap_lines=True)
