"""Punto de entrada que pide el enunciado:  streamlit run app.py
La app vive en app/streamlit_app.py (4 pestañas) y sus componentes visuales en app/ui.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"), run_name="__main__")
