"""Regla de conteo de fechas (baseline bonus) y perturbaciones contrafactuales (Pregunta 1).

Formato de PopResume (visto al leer 50 textos OCR): "Cargo, Empresa, 2019.11–2023.01", con cargo, empresa y
fechas en la misma línea, y años sueltos en EDUCATION ("Bachelor's, …, 2015"). También se aceptan los
formatos habituales ("Mar 2014 – Jun 2018", "03/2012 - 2016", "2015 - Present") para CVs subidos a la app."""
import re

from .texto import reparar_fechas

MESES = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
MES = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"


def _fecha(k):
    # "Mar 2014" | "03/2014" | "2014" | "2014.03" (PopResume)
    return (rf"(?:(?P<m{k}>{MES})\s*,?\s*|(?P<n{k}>\d{{1,2}})\s*/\s*)?"
            rf"(?P<y{k}>(?:19|20)\d{{2}})(?:\.(?P<p{k}>\d{{1,2}}))?")


PRESENTE = r"(?P<pres>present|current|now|today|ongoing)"
RE_RANGO = re.compile(
    _fecha(1) + r"\s*(?:-|–|—|to|until|~)\s*(?:" + PRESENTE + "|" + _fecha(2) + ")", re.I)


def _mes(nombre, numero, punto):
    """Mes 1–12 a partir de 'Mar', '03/' o '.03'. Sin mes: None. '.0' (OCR truncó '.01') cuenta como enero."""
    if nombre:
        return MESES.get(nombre[:3].lower(), 6)
    for v in (numero, punto):
        if v:
            return min(max(int(v), 1), 12)
    return None


def _a_anios(m, anio_actual):
    """(inicio, fin) de un match de RE_RANGO en años decimales. Sin mes: inicio a mitad de año."""
    ini = int(m["y1"]) + ((_mes(m["m1"], m["n1"], m["p1"]) or 6) - 1) / 12
    if m["pres"]:
        fin = anio_actual + 0.5
    else:
        fin = int(m["y2"]) + ((_mes(m["m2"], m["n2"], m["p2"]) or 6) - 1) / 12
    return ini, fin


def rangos(texto, anio_actual):
    """Lista de (inicio, fin) en años decimales de todos los rangos del texto."""
    out = []
    for m in RE_RANGO.finditer(reparar_fechas(texto)):
        ini, fin = _a_anios(m, anio_actual)
        if 0 <= fin - ini <= 50:
            out.append((ini, fin))
    return out


def fusionar(intervalos):
    """Une solapamientos (dos empleos a la vez no cuentan doble)."""
    total, actual = 0.0, None
    for ini, fin in sorted(intervalos):
        if actual is None or ini > actual[1]:
            if actual:
                total += actual[1] - actual[0]
            actual = [ini, fin]
        else:
            actual[1] = max(actual[1], fin)
    if actual:
        total += actual[1] - actual[0]
    return total


# Solo palabras de TÍTULO académico: "school", "university" o "associate" también son empleadores
# ("Lincoln Middle School", "University Hospital") o cargos ("Associate Accountant").
EDUCACION = ("bachelor", "master of", "master's", "degree", "b.s.", "b.a.", "m.s.", "m.a.", "ph.d", "diploma",
             "associate of", "associate's", "associate degree", "gpa", "graduated", "coursework", "mba")
RE_EDUCACION = re.compile(r"\b(?:" + "|".join(map(re.escape, EDUCACION)) + ")", re.I)


def es_educativo(contexto):
    return bool(RE_EDUCACION.search(contexto))


# Secciones: un rango solo es laboral si está bajo un título de experiencia. Sin esto, un estudiante que lista
# actividades fechadas ("President, Debate Club, 2019–2022") recibía años de experiencia que no tiene.
# Los títulos se reconocen solos en su línea; "volunteer experience" o "leadership experience" NO son laborales.
LABORALES = (r"work history|work experience|professional experience|relevant experience|experience|employment"
             r"(?: history)?|career history|internships?|experiencia(?: laboral| profesional)?|"
             r"historial laboral|pr[aá]cticas(?: pre ?profesionales| profesionales)?")
OTRAS = (r"education|academic background|skills|summary|profile|objective|certifications?|projects?|languages|"
         r"awards|honou?rs|publications|volunteer(?:ing| experience| work)?|(?:extra ?curricular )?activities|"
         r"extra ?curriculars?|leadership(?: experience)?|courses|relevant coursework|coursework|interests|"
         r"references|contact|educaci[oó]n|formaci[oó]n(?: acad[eé]mica)?|habilidades|perfil|resumen|objetivo|"
         r"proyectos?|idiomas|certificaciones|cursos|logros|premios|actividades(?: extra ?curriculares)?|"
         r"voluntariado|publicaciones|intereses|referencias")
RE_TITULO_LABORAL = re.compile(rf"^\s*(?:{LABORALES})\s*:?\s*$", re.I)
RE_TITULO_OTRO = re.compile(rf"^\s*(?:{OTRAS})\s*:?\s*$", re.I)


def _seccion(linea):
    """'laboral', 'otra' o None si la línea no es un título de sección."""
    if RE_TITULO_LABORAL.match(linea):
        return "laboral"
    if RE_TITULO_OTRO.match(linea):       # no se usa "todo en mayúsculas": muchos CVs escriben así los cargos
        return "otra"
    return None


def tiene_seccion_laboral(texto):
    """True/False si el CV tiene o no un título de experiencia; None si no se reconoce ningún título."""
    secciones = {_seccion(l) for l in reparar_fechas(texto).splitlines()} - {None}
    return ("laboral" in secciones) if secciones else None


def _rangos_laborales(lineas):
    """[(índice_de_línea, match)] de los rangos laborales: dentro de una sección de experiencia y sin palabras de
    estudios (contexto: su línea y la anterior). Si el texto no tiene ningún título reconocible, se usan todos los
    rangos no educativos (no hay forma de saber en qué sección están)."""
    secciones = [_seccion(l) for l in lineas]
    con_titulos = any(secciones)
    out, actual = [], None
    for i, linea in enumerate(lineas):
        actual = secciones[i] or actual
        if con_titulos and actual != "laboral":
            continue
        for m in RE_RANGO.finditer(linea):
            if not es_educativo("\n".join(lineas[max(0, i - 1): i + 1])):
                out.append((i, m))
    return out


def estimar(texto, anio_actual, tope):
    """Años de experiencia = suma de rangos laborales fusionados (excluye rangos educativos)."""
    lineas = reparar_fechas(texto).splitlines()
    lab = [_a_anios(m, anio_actual) for _, m in _rangos_laborales(lineas)]
    lab = [(a, b) for a, b in lab if 0 <= b - a <= 50]
    return min(max(fusionar(lab), 0.0), tope)


def detalle(texto, anio_actual):
    """Rangos laborales que usa `estimar`, para mostrarlos: [{rango, inicio, fin, anios}] (años decimales)."""
    lineas = reparar_fechas(texto).splitlines()
    out = []
    for _, m in _rangos_laborales(lineas):
        ini, fin = _a_anios(m, anio_actual)
        if 0 <= fin - ini <= 50:
            out.append({"rango": m.group(0), "inicio": ini, "fin": fin, "anios": fin - ini})
    return out


# ── Perturbaciones para la Pregunta 1 ──────────────────────────────────────────
# Cada una tiene una respuesta "de reclutador" conocida, que se calcula con la regla sobre el texto perturbado.

def desplazar_fechas(texto, delta=5):
    """A · TODOS los años +5. La carrera ocurre 5 años más tarde pero dura lo mismo: un reclutador NO
    cambiaría su estimación. Un modelo que mira años absolutos ("2009" ⇒ senior) sí lo haría."""
    return re.sub(r"(?<!\d)((?:19|20)\d{2})(?!\d)", lambda m: str(int(m[1]) + delta), reparar_fechas(texto))


def adelantar_inicio(texto, delta=5):
    """B · El empleo más antiguo empieza `delta` años antes (solo cambia ese año de inicio). Un reclutador
    sumaría ≈ +5 años."""
    lineas = reparar_fechas(texto).splitlines()
    lab = _rangos_laborales(lineas)
    if not lab:
        return texto, False
    i, m = min(lab, key=lambda x: int(x[1]["y1"]))
    nuevo = str(int(m["y1"]) - delta)
    lineas[i] = lineas[i][:m.start("y1")] + nuevo + lineas[i][m.end("y1"):]
    return "\n".join(lineas), True


def quitar_empleo_mas_antiguo(texto):
    """C · Borra el bloque del empleo con el inicio más antiguo: su línea (cargo, empresa, fechas) y sus
    viñetas, hasta el siguiente rango o encabezado. Un reclutador restaría la duración de ese empleo."""
    lineas = reparar_fechas(texto).splitlines()
    lab = _rangos_laborales(lineas)
    if not lab:
        return texto, False
    i, m = min(lab, key=lambda x: int(x[1]["y1"]))
    # si las fechas van solas en su línea, el cargo/empresa está en la línea anterior
    ini = i if re.search(r"[A-Za-z]{3}", lineas[i][:m.start()]) else max(0, i - 1)
    fin = i + 1
    while fin < len(lineas) and not RE_RANGO.search(lineas[fin]) and not _es_encabezado(lineas[fin]):
        fin += 1
    if (fin < len(lineas) and RE_RANGO.search(lineas[fin]) and fin - 1 > i
            and not re.search(r"[A-Za-z]{3}", lineas[fin][:RE_RANGO.search(lineas[fin]).start()])):
        fin -= 1                      # el siguiente rango va solo: su cargo/empresa es la línea previa
    return "\n".join(lineas[:ini] + lineas[fin:]), True


def _es_encabezado(linea):
    s = linea.strip()
    return s.isupper() and 3 <= len(s) <= 30 and not any(ch.isdigit() for ch in s)
