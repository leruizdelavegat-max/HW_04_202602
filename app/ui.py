"""Componentes visuales comunes de la app: estilo, encabezados, franjas de indicadores, notas y gráficos Altair."""
import html
import re

import altair as alt
import pandas as pd
import streamlit as st

ETIQUETA = {"mediana": "Mediana", "tfidf_ridge": "TF-IDF + Ridge", "embeddings_ridge": "Embeddings + Ridge",
            "bert": "DistilBERT (fine-tuned)", "regla_fechas": "Regla de fechas (bonus)"}
REQUERIDOS = ["mediana", "tfidf_ridge", "embeddings_ridge", "bert"]       # los 4 modelos del enunciado
MODELOS = REQUERIDOS + ["regla_fechas"]
COLOR = {"mediana": "#9AA3AF", "tfidf_ridge": "#4E79A7", "embeddings_ridge": "#59A14F",
         "bert": "#D9762B", "regla_fechas": "#7D6BA6"}
ESCALA_MODELOS = alt.Scale(domain=[ETIQUETA[m] for m in MODELOS], range=[COLOR[m] for m in MODELOS])


def escala(etiquetas):
    """Escala de color con solo los modelos mostrados (la leyenda no lista modelos ausentes)."""
    claves = [k for k in MODELOS if ETIQUETA[k] in etiquetas]
    return alt.Scale(domain=[ETIQUETA[k] for k in claves], range=[COLOR[k] for k in claves])
AZUL, GRIS, ROJO = "#1F4E79", "#6B7480", "#B23B3B"

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1 {font-size: 1.95rem !important; font-weight: 700; color: #14263D; letter-spacing: -0.01em; margin-bottom: .2rem;}
h3 {font-size: 1.22rem !important; font-weight: 650; color: #14263D; margin-top: 1.1rem !important;
    padding-bottom: .3rem !important; border-bottom: 1px solid #E3E7ED;}
.stTabs [data-baseweb="tab-list"] {gap: .2rem; border-bottom: 1px solid #DDE2E8;}
.stTabs [data-baseweb="tab"] {font-weight: 600; padding: .55rem .9rem;}
.subtitulo {color: #4A5361; font-size: 1rem; margin: 0 0 1rem;}
.nota {color: #4A5361; font-size: .9rem; line-height: 1.5; margin: .2rem 0 .9rem;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border: 1px solid #E3E7ED;
       border-radius: 10px; background: #F8FAFC; margin: .4rem 0 1rem; overflow: hidden;}
.kpi {padding: .75rem 1rem; border-right: 1px solid #E3E7ED;}
.kpi:last-child {border-right: none;}
.kpi .l {font-size: .74rem; color: #5B6573; text-transform: uppercase; letter-spacing: .04em; font-weight: 600;}
.kpi .v {font-size: clamp(1.1rem, 2.2vw, 1.45rem); font-weight: 700; color: #14263D; line-height: 1.25;
         margin: .15rem 0; overflow-wrap: anywhere;}
.kpi .n {font-size: .78rem; color: #6B7480; line-height: 1.35;}
.kpi.pendiente .v {color: #9AA3AF;}
.pasos {display: flex; flex-wrap: wrap; gap: .35rem; margin: .3rem 0 1rem; align-items: stretch;}
.paso {flex: 1 1 140px; border: 1px solid #E3E7ED; border-radius: 8px; padding: .5rem .7rem; background: #FFF;}
.paso .t {font-weight: 600; font-size: .86rem; color: #14263D;}
.paso .e {font-size: .76rem; margin-top: .15rem;}
.ok {color: #2E7D32;} .pend {color: #B26A00;}
.estimacion {border: 1px solid #E3E7ED; border-left: 4px solid #D9762B; border-radius: 8px; padding: .8rem 1rem;
             background: #FFF8F2; margin-bottom: .8rem;}
.estimacion .v {font-size: 2.1rem; font-weight: 700; color: #14263D; line-height: 1.1;}
.estimacion .l {font-size: .8rem; color: #5B6573; text-transform: uppercase; letter-spacing: .04em; font-weight: 600;}
.estimacion .n {font-size: .85rem; color: #4A5361; margin-top: .25rem;}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem;}
  .kpi {border-right: none; border-bottom: 1px solid #E3E7ED;}
  h1 {font-size: 1.5rem !important;}
}
</style>
"""


def estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def nota(texto):
    """Explicación breve bajo un gráfico o tabla (Markdown en línea permitido)."""
    texto = re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)       # dentro de HTML el Markdown no se procesa
    st.markdown(f'<div class="nota">{texto}</div>', unsafe_allow_html=True)


def kpis(items):
    """Franja de indicadores: [(etiqueta, valor, nota)]. Valor None = pendiente (no se inventa un número)."""
    celdas = []
    for etiqueta, valor, pie in items:
        clase = "kpi pendiente" if valor is None else "kpi"
        v = "Pendiente" if valor is None else html.escape(str(valor))
        celdas.append(f'<div class="{clase}"><div class="l">{html.escape(etiqueta)}</div><div class="v">{v}</div>'
                      f'<div class="n">{html.escape(pie or "")}</div></div>')
    st.markdown(f'<div class="kpis">{"".join(celdas)}</div>', unsafe_allow_html=True)


def pasos(items):
    """Estado del pipeline: [(nombre, hecho: bool, detalle)]."""
    celdas = []
    for nombre, hecho, detalle in items:
        estado = '<span class="ok">✔ Completado</span>' if hecho else '<span class="pend">● Pendiente</span>'
        celdas.append(f'<div class="paso"><div class="t">{html.escape(nombre)}</div><div class="e">{estado}</div>'
                      f'<div class="e" style="color:#6B7480">{html.escape(detalle)}</div></div>')
    st.markdown(f'<div class="pasos">{"".join(celdas)}</div>', unsafe_allow_html=True)


def estimacion(etiqueta, valor, pie):
    st.markdown(f'<div class="estimacion"><div class="l">{html.escape(etiqueta)}</div>'
                f'<div class="v">{html.escape(valor)}</div><div class="n">{pie}</div></div>', unsafe_allow_html=True)


def falta(recurso, como):
    st.info(f"**Resultado no disponible:** falta `{recurso}`. {como}", icon=":material/hourglass_empty:")


def num(x, d=2):
    """Número con coma decimal y espacio de miles (formato académico en español)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",")


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else num(100 * x, d) + " %"


def grafico(chart):
    st.altair_chart(chart.configure_axis(labelFontSize=11, titleFontSize=12, grid=True, gridColor="#EEF1F4")
                    .configure_view(strokeWidth=0).configure_legend(labelFontSize=11, titleFontSize=11, orient="bottom", columns=3,
                                                       labelLimit=200),
                    width="stretch")
