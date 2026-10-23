"""Limpieza del texto OCR, eliminación de PII y enmascaramiento de fugas del target.

Regla del issue: se ocultan la experiencia TOTAL declarada, la edad y la fecha de nacimiento.
Las fechas de cada empleo y los años de graduación SE CONSERVAN: son la señal legítima."""
import re

import numpy as np

MASK = "[MASK]"

# ── PII ─────────────────────────────────────────────────────────────────────────
RE_EMAIL = re.compile(r"\b[\w.+-]+\s?@\s?[\w-]+(?:\.[\w-]+)+\b")
RE_URL = re.compile(r"\b(?:https?://|www\.)\S+|\blinkedin\.com/\S*", re.I)
RE_TEL = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
# Calle: número + palabras + sufijo típico de EE. UU.
RE_CALLE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.]*\s+){0,4}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|"
    r"Ct|Court|Way|Pl|Place|Pkwy|Parkway|Hwy|Highway|Ter|Terrace|Cir|Circle)\b\.?(?:,?\s*(?:Apt|Suite|Unit|#)\s*\w+)?",
    re.I)
# "City, ST 12345" o "City, ST" — solo con códigos de estado reales, para no borrar "Nurse, RN"
ESTADOS = ("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
           "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC").split()
RE_CIUDAD = re.compile(rf"\b[A-Z][a-zA-Z.]*(?:\s[A-Z][a-zA-Z.]*){{0,3}},\s*(?:{'|'.join(ESTADOS)})\b(?:\s+\d{{5}}(?:-\d{{4}})?)?")

# ── Fugas del target ────────────────────────────────────────────────────────────
NUM = r"(?:\d{1,2}(?:\.\d)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|" \
      r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)"
RE_FUGAS = {
    # "12 years of experience", "over 5+ years' experience", "8 yrs experience in ..."
    "experiencia_total": re.compile(
        rf"\b(?:over|more than|nearly|almost|about|around|approximately)?\s*{NUM}\s*\+?\s*"
        rf"(?:years?|yrs?)['’]?\s+(?:of\s+)?(?:\w+\s+){{0,3}}?(?:experience|exp\b|career)", re.I),
    # "Total experience: 7 years", "Experience: 7 yrs"
    "experiencia_campo": re.compile(rf"\b(?:total\s+)?(?:work\s+)?experience\s*[:\-]\s*{NUM}\s*\+?\s*(?:years?|yrs?)", re.I),
    "edad": re.compile(rf"\b(?:age\s*[:\-]?\s*\d{{2}}|\d{{2}}\s*(?:years?|yrs?)\s*old)\b", re.I),
    "nacimiento": re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+on|\s+in)?|birth\s*date)\s*[:\-]?\s*"
        r"[\w ,./-]{0,25}?\b(?:19|20)\d{2}\b", re.I),
}

RE_ANIO = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")


def quitar_pii(texto, nombres=()):
    """Borra correo, URL, teléfono, dirección y el nombre propio del CV (si se conoce)."""
    t = RE_EMAIL.sub(" ", texto)
    t = RE_URL.sub(" ", t)
    t = RE_TEL.sub(" ", t)
    t = RE_CALLE.sub(" ", t)
    t = RE_CIUDAD.sub(" ", t)
    for n in nombres:
        if n and len(str(n)) > 1:
            t = re.sub(rf"\b{re.escape(str(n))}\b", " ", t, flags=re.I)
    return t


def enmascarar(texto):
    """Reemplaza las fugas por [MASK]. Devuelve (texto, {tipo: n_reemplazos})."""
    conteo = {}
    for tipo, patron in RE_FUGAS.items():
        texto, n = patron.subn(MASK, texto)
        conteo[tipo] = n
    return texto, conteo


def unir_lineas(texto):
    """Repara palabras partidas con guion al final de línea y colapsa espacios."""
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def limpiar(lineas, umbral, nombres=()):
    """Pipeline completo sobre las líneas de un JSON de OCR (ya en orden de lectura).
    Devuelve (texto_limpio, conteo_de_mascaras)."""
    crudo = "\n".join(l["text"] for l in lineas if l["conf"] >= umbral)
    t = unir_lineas(crudo)
    t = quitar_pii(t, nombres)
    t, conteo = enmascarar(t)
    return _compactar(t), conteo


def _compactar(t):
    """Colapsa espacios y quita líneas que quedaron sin letras ni dígitos (restos como ", |" tras borrar PII)."""
    lineas = (re.sub(r"[ \t]+", " ", s).strip(" ,|·•-") for s in t.splitlines())
    return "\n".join(s for s in lineas if re.search(r"[A-Za-z0-9]", s))


def limpiar_texto_plano(texto, nombres=()):
    """Misma limpieza para un texto que no viene del OCR (referencia `text` o PDF con capa de texto)."""
    t = unir_lineas(texto)
    t = quitar_pii(t, nombres)
    t, conteo = enmascarar(t)
    return _compactar(t), conteo


def anios(texto):
    return [int(a) for a in RE_ANIO.findall(texto or "")]


def recall_anios(referencia, ocr):
    """Proporción de años de la referencia que aparecen en el OCR (contando repeticiones)."""
    ref, hyp = anios(referencia), anios(ocr)
    if not ref:
        return np.nan
    disponibles = {}
    for a in hyp:
        disponibles[a] = disponibles.get(a, 0) + 1
    acierto = 0
    for a in ref:
        if disponibles.get(a, 0) > 0:
            acierto += 1
            disponibles[a] -= 1
    return acierto / len(ref)


def normalizar_para_cer(texto):
    return " ".join((texto or "").lower().split())
