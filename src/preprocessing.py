"""Etapa 2 · Limpieza y enmascarado del texto OCR.

Punto de entrada con el nombre que pide el enunciado; la implementación está en `texto.py` (pipeline de los
notebooks) y en `anonimizar.py` (CVs subidos a la app). Pasos de `limpiar`:
1. descarta líneas bajo el umbral de confianza (0,0: ninguna línea baja de 0,9, ver `umbral_confianza.csv`);
2. une palabras partidas con guion y repara fechas pegadas por el OCR («2019.112023.01» → «2019.11–2023.01»);
3. borra nombre, correo, teléfono, URL y dirección;
4. enmascara con [MASK] la experiencia total declarada, la edad y la fecha de nacimiento.
Las fechas de cada empleo y los años de graduación se conservan: son la evidencia del modelo."""
from .anonimizar import anonimizar_texto, clasificar, imagen_protegida, texto_visible, tiene_pii
from .texto import (MASK, enmascarar, limpiar, limpiar_texto_plano, quitar_pii, reparar_fechas, unir_lineas)


def contar_enmascarados(conteos):
    """Nº de CVs con al menos una máscara, a partir de la lista de conteos que devuelve `limpiar`."""
    return sum(1 for c in conteos if sum(c.values()) > 0)


__all__ = ["MASK", "anonimizar_texto", "clasificar", "contar_enmascarados", "enmascarar", "imagen_protegida",
           "limpiar", "limpiar_texto_plano", "quitar_pii", "reparar_fechas", "texto_visible", "tiene_pii",
           "unir_lineas"]
