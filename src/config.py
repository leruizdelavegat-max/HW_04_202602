"""Constantes compartidas por notebooks y app. Cambiar algo aquí cambia todo el pipeline."""
from pathlib import Path

# ── Reproducibilidad ────────────────────────────────────────────────────────────
SEED = 42

# ── Dataset ─────────────────────────────────────────────────────────────────────
HF_DATASET = "sumin-yu/PopResume"           # gated: aceptar condiciones en Hugging Face
N_POR_OCUPACION = 800                        # 5 × 800 = 4 000 CVs (lo que pide el issue)
TARGET = "exp_year"

# Columnas que JAMÁS entran como feature (solo auditoría de equidad o limpieza de PII)
PROHIBIDAS = ["age", "sex", "race", "first_name", "last_name", "state", "region",
              "first_name_sex", "first_name_age_group", "surname_race_label"]

# ── Target ──────────────────────────────────────────────────────────────────────
TOPE_BASE = 20                 # recorte 0–20 que pide el issue
TOPES_ALTERNATIVOS = (25, 30)  # solo si >15 % de la muestra queda en el tope
UMBRAL_TOPE = 0.15

# Bandas para la evaluación por clases (el límite superior de la última se ajusta al tope)
BANDAS = [(0, 2), (3, 5), (6, 10), (11, 20)]
NOMBRES_BANDAS = ["0–2", "3–5", "6–10", "11–20"]

# ── OCR (PaddleOCR 3.x) ─────────────────────────────────────────────────────────
# Mobile det + reconocedor específico de inglés: los CVs son renders limpios en inglés,
# y la app vuelve a correr el MISMO modelo en CPU (sin GPU), así que debe ser liviano.
OCR_DET = "PP-OCRv5_mobile_det"
OCR_REC = "en_PP-OCRv5_mobile_rec"
OCR_OPCIONES = dict(
    use_doc_orientation_classify=False,   # renders ya derechos
    use_doc_unwarping=False,              # no hay hojas dobladas
    use_textline_orientation=False,       # todas las líneas son horizontales
)

# ── Rutas (relativas a la raíz del repo; en Colab DATOS apunta a Drive) ────────
RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
SPLITS = DATA / "splits"
PROCESADOS = DATA / "processed"
OUTPUT = RAIZ / "output"
TABLAS = OUTPUT / "tables"
FIGURAS = OUTPUT / "figures"
MODELOS = RAIZ / "models"            # en .gitignore: los pesos no van al repo

for _p in (SPLITS, PROCESADOS, TABLAS, FIGURAS, MODELOS):
    _p.mkdir(parents=True, exist_ok=True)


def banda(anios):
    """Índice de banda (0..3). Las predicciones continuas se redondean a la banda más cercana
    (2.4 → «0–2», 2.6 → «3–5»); la última banda absorbe todo lo que pase de 10."""
    for i, (_, hasta) in enumerate(BANDAS[:-1]):
        if anios < hasta + 0.5:
            return i
    return len(BANDAS) - 1
