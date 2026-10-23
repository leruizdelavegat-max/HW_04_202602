"""Regla de conteo de fechas (baseline bonus) y perturbaciones contrafactuales (Pregunta 1)."""
import re

MESES = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}
MES = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
def _fecha(k):
    return rf"(?:(?P<m{k}>{MES})\s*,?\s*|(?P<n{k}>\d{{1,2}})\s*/\s*)?(?P<y{k}>(?:19|20)\d{{2}})"


PRESENTE = r"(?P<pres>present|current|now|today|ongoing)"
RE_RANGO = re.compile(
    _fecha(1) + r"\s*(?:-|–|—|to|until|~)\s*(?:" + PRESENTE + "|" + _fecha(2) + ")", re.I)


def _mes(m, n):
    if m:
        return MESES.get(m[:3].lower(), 6)
    if n and 1 <= int(n) <= 12:
        return int(n)
    return None


def rangos(texto, anio_actual):
    """Lista de (inicio, fin) en años decimales. Sin mes: inicio a mitad de año."""
    out = []
    for m in RE_RANGO.finditer(texto or ""):
        y1 = int(m["y1"])
        m1 = _mes(m["m1"], m["n1"]) or 6
        if m["pres"]:
            fin = anio_actual + 0.5
        else:
            y2 = int(m["y2"])
            m2 = _mes(m["m2"], m["n2"]) or 6
            fin = y2 + (m2 - 1) / 12
        ini = y1 + (m1 - 1) / 12
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


def estimar(texto, anio_actual, tope):
    """Años de experiencia = suma de rangos laborales fusionados (excluye rangos educativos)."""
    lineas = (texto or "").splitlines()
    lab = []
    for i, linea in enumerate(lineas):
        # contexto = la línea del rango y la anterior (cargo/empresa o título/universidad)
        if not RE_RANGO.search(linea) or es_educativo("\n".join(lineas[max(0, i - 1): i + 1])):
            continue
        lab.extend(rangos(linea, anio_actual))
    return min(max(fusionar(lab), 0.0), tope)


# ── Perturbaciones para la Pregunta 1 ──────────────────────────────────────────

def desplazar_fechas(texto, delta=-5):
    """Corre TODOS los años `delta` años (por defecto 5 hacia atrás). Con un empleo actual
    ("– Present") la carrera queda 5 años más larga: un reclutador sumaría ≈ +5 años."""
    return re.sub(r"(?<!\d)((?:19|20)\d{2})(?!\d)", lambda m: str(int(m[1]) + delta), texto)


def quitar_empleo_mas_antiguo(texto):
    """Elimina el bloque del empleo con la fecha de inicio más antigua: desde la línea anterior
    a su rango de fechas (cargo/empresa) hasta antes del siguiente rango o encabezado.
    Un reclutador debería restar la duración de ese empleo."""
    lineas = texto.splitlines()
    candidatos = []
    for i, l in enumerate(lineas):
        m = RE_RANGO.search(l)
        if m and not es_educativo("\n".join(lineas[max(0, i - 1): i + 1])):
            candidatos.append((int(m["y1"]), i))
    if not candidatos:
        return texto, False
    _, i = min(candidatos)
    ini = max(0, i - 1)
    fin = i + 1
    while fin < len(lineas) and not RE_RANGO.search(lineas[fin]) and not _es_encabezado(lineas[fin]):
        fin += 1
    if fin < len(lineas) and RE_RANGO.search(lineas[fin]) and fin - 1 > i:
        fin -= 1                      # la línea previa al siguiente rango es su cargo/empresa
    return "\n".join(lineas[:ini] + lineas[fin:]), True


def _es_encabezado(linea):
    s = linea.strip()
    return s.isupper() and 3 <= len(s) <= 30 and not any(ch.isdigit() for ch in s)
