"""Anonimización de un CV leído por OCR: qué líneas son datos personales, texto sin ellas e imagen protegida.

Se trabaja por LÍNEA de OCR (cada una con su caja), así la misma decisión sirve para el texto y para la imagen.
Es conservador a propósito: ante la duda una línea del encabezado se trata como personal.

Imagen protegida: no se tapa el original, se reconstruye sobre un lienzo blanco pegando SOLO las cajas de texto
no personales. Lo que el OCR no detectó (fotos, firmas, logos, texto ilegible) nunca llega a mostrarse."""
import re

from . import texto

# Títulos de sección en inglés (PopResume) y en español (CVs subidos)
SECCIONES = (
    r"work history|work experience|professional experience|experience|employment(?: history)?|education|skills|"
    r"summary|profile|objective|certifications?|projects?|languages|awards|publications|volunteer(?:ing)?|"
    r"experiencia(?: laboral| profesional)?|educaci[oó]n|formaci[oó]n(?: acad[eé]mica)?|habilidades|perfil|"
    r"resumen|objetivo|proyectos?|idiomas|certificaciones|cursos|logros|actividades(?: extra ?curriculares)?|"
    r"voluntariado|publicaciones")
RE_SECCION = re.compile(rf"^\s*(?:{SECCIONES})\s*:?\s*$", re.I)
# Secciones cuyo contenido completo es personal (barras laterales de contacto)
RE_SECCION_CONTACTO = re.compile(
    r"^\s*(?:contact(?:o| information| details)?|datos personales|personal (?:information|details|data)|"
    r"informaci[oó]n personal|references|referencias)\s*:?\s*$", re.I)

# Identificadores que no cubre texto.quitar_pii: dominios con ruta (github.com/usuario), documentos, nacimiento
RE_DOMINIO = re.compile(r"\b[\w-]+\.(?:com|org|net|io|edu|dev|me|pe|es|mx|co)(?:\.[a-z]{2})?/\S*", re.I)
RE_DOCUMENTO = re.compile(r"\b(?:dni|ssn|pasaporte|passport|c[eé]dula|rut|curp|nie|nif)\b\s*[:#n°º.]*\s*\w*", re.I)
RE_NACIMIENTO = re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born|fecha de nacimiento|nacid[oa])\b", re.I)
RE_DIRECCION = re.compile(r"\b(?:address|direcci[oó]n|domicilio|calle|av\.|avenida|jr\.|jir[oó]n|mz\.|urb\.)\s", re.I)
RE_SOLO_LETRAS = re.compile(r"^[A-Za-zÀ-ÿ'’.\- ]+$")
NO_NOMBRE = re.compile(r"\b(?:curriculum|vitae|resume|résumé|cv|hoja de vida)\b", re.I)

ETIQUETA = "[DATOS PERSONALES OCULTOS]"


def _es_telefono(s):
    """Secuencias de ≥ 9 dígitos con separadores (+51 925 451 276, 925451276). Se descartan las fechas
    ('2019.11–2023.01', '2012 - 2016'): grupos que son todos años o meses."""
    for m in re.finditer(r"\+?\d[\d\s().-]{7,}\d", texto.reparar_fechas(s)):
        grupos = re.findall(r"\d+", m.group(0))
        if sum(map(len, grupos)) >= 9 and not all(re.fullmatch(r"(?:19|20)\d\d|\d{1,2}", g) for g in grupos):
            return True
    return False


def tiene_pii(s):
    """¿La línea contiene un identificador directo (correo, URL, teléfono, dirección, documento…)?"""
    return bool(texto.RE_EMAIL.search(s) or texto.RE_URL.search(s) or texto.RE_TEL.search(s)
                or texto.RE_CALLE.search(s) or texto.RE_CIUDAD.search(s) or texto.RE_ESTADO_USA.search(s)
                or RE_DOMINIO.search(s) or RE_DOCUMENTO.search(s) or RE_NACIMIENTO.search(s)
                or RE_DIRECCION.search(s) or _es_telefono(s))


def _parece_nombre(s):
    s = s.strip()
    palabras = s.split()
    return (2 <= len(palabras) <= 7 and RE_SOLO_LETRAS.match(s) is not None
            and not RE_SECCION.match(s) and not NO_NOMBRE.search(s))


def clasificar(lineas):
    """Lista de booleanos (True = personal) para líneas de OCR ya en orden de lectura.

    1. Cualquier línea con un identificador directo.
    2. El encabezado (antes del primer título de sección): el nombre y todo lo que no sea un párrafo largo
       de resumen. Si no hay títulos, se toman las 3 primeras líneas como encabezado.
    3. Secciones de contacto o referencias completas.
    4. Cualquier otra línea que repita el nombre detectado (firmas, pies de página)."""
    textos = [l["text"] if isinstance(l, dict) else str(l) for l in lineas]
    personales = [tiene_pii(t) for t in textos]
    primera_seccion = next((i for i, t in enumerate(textos) if RE_SECCION.match(t)), None)
    fin_encabezado = primera_seccion if primera_seccion is not None else min(3, len(textos))

    nombre = None
    for i in range(fin_encabezado):
        if nombre is None and _parece_nombre(textos[i]):
            nombre = textos[i]
            personales[i] = True
        elif len(textos[i].split()) <= 12:          # líneas cortas del encabezado: ubicación, cargo + contacto…
            personales[i] = True

    en_contacto = False
    for i, t in enumerate(textos):
        if RE_SECCION_CONTACTO.match(t):
            en_contacto = True
        elif RE_SECCION.match(t):
            en_contacto = False
        if en_contacto:
            personales[i] = True

    if nombre:
        partes = {p.lower() for p in nombre.split() if len(p) >= 3}
        for i, t in enumerate(textos):
            if len(partes & {p.lower().strip(".,|") for p in t.split()}) >= min(2, len(partes)):
                personales[i] = True
    return personales


def texto_visible(lineas, personales):
    """Texto en orden de lectura con las líneas personales sustituidas por una etiqueta (para mostrar)."""
    out, previa = [], False
    for l, p in zip(lineas, personales):
        if p:
            if not previa:
                out.append(ETIQUETA)
            previa = True
        else:
            out.append(l["text"] if isinstance(l, dict) else str(l))
            previa = False
    return texto.reparar_fechas("\n".join(out))


def anonimizar_texto(texto_ocr):
    """Versión para un texto ya unido (una línea por renglón), sin cajas."""
    lineas = [s for s in (texto_ocr or "").splitlines() if s.strip()]
    return texto_visible(lineas, clasificar(lineas))


def _color_conf(c):
    return "#2E7D32" if c >= 0.9 else "#E0A100" if c >= 0.7 else "#C62828"


def imagen_protegida(imagen, lineas, personales, cajas=True):
    """Reconstruye la página con solo las cajas no personales. Las personales se dibujan como barras oscuras.
    Con `cajas=True` añade el contorno de cada caja (color = confianza) y su número de orden de lectura."""
    from PIL import Image, ImageDraw
    ancho, alto = imagen.size
    lienzo = Image.new("RGB", (ancho, alto), "white")
    rect = []
    for l in lineas:
        x0, y0, x1, y1 = l["box"]
        rect.append((max(0, x0), max(0, y0), min(ancho, x1), min(alto, y1)))
    for r, p in zip(rect, personales):
        if not p and r[2] > r[0] and r[3] > r[1]:
            lienzo.paste(imagen.crop(r), r[:2])
    d = ImageDraw.Draw(lienzo)
    grosor = max(1, round(ancho / 600))
    for k, (r, p, l) in enumerate(zip(rect, personales, lineas)):
        if p:                                       # las barras van al final: tapan cualquier solape
            d.rectangle((r[0] - 2, r[1] - 2, r[2] + 2, r[3] + 2), fill="#2B2F36")
        elif cajas:
            d.rectangle(r, outline=_color_conf(l.get("conf", 1.0)), width=grosor)
            d.text((r[0], max(0, r[1] - 12 * grosor)), str(k), fill="#1F4E79")
    if rect:                                        # recorta al área con texto: la vista se lee mejor
        m = 30
        lienzo = lienzo.crop((max(0, min(r[0] for r in rect) - m), max(0, min(r[1] for r in rect) - m),
                              min(ancho, max(r[2] for r in rect) + m), min(alto, max(r[3] for r in rect) + m)))
    return lienzo
