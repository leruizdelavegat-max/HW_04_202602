# HW_04_202602

**Estimar los años de experiencia laboral a partir de la imagen de un CV**, con PaddleOCR para leer y
cuatro modelos para estimar: mediana, TF-IDF + Ridge, embeddings + Ridge y DistilBERT ajustado como regresor
(más una regla de conteo de fechas como bonus).

> Las cifras marcadas con ⟨…⟩ se completan con las salidas de los notebooks (`output/`).

## 1 · Qué hace y para quién

Lee la **imagen** de un currículum, extrae el texto con OCR y predice `exp_year` (0–20 años). No basta con
buscar una frase: el modelo tiene que combinar fechas de empleo, número de puestos, progresión de cargos y fin
de estudios.

**Usuarios posibles:** bolsas de empleo y servicios públicos de empleo que reciben CVs escaneados y necesitan
pre-llenar el campo «años de experiencia» de su formulario; equipos de RR. HH. que quieren **verificar** lo que
declara un postulante; agencias que ordenan grandes volúmenes para revisión humana. **Nunca** para decidir
una contratación (ver §9).

## 2 · Datos y muestreo

- **PopResume** (Yu, Park & Moon, EMNLP 2026; CC BY 4.0, `sumin-yu/PopResume` en Hugging Face): 60 884 CVs
  sintéticos de 5 ocupaciones, construidos a partir de estadísticas de EE. UU.
- **Muestra:** 800 CVs por ocupación = **4 000**, semilla **42**. Dentro de cada ocupación se estratifica por
  banda de experiencia (asignación proporcional) para conservar la distribución del target. IDs en
  [`data/sample_ids.csv`](data/sample_ids.csv).

| Ocupación | Población | Muestra |
|---|---:|---:|
| Enfermeros/as registrados | 17 632 | 800 |
| Maestros de primaria/secundaria | 16 188 | 800 |
| Desarrolladores de software | 13 370 | 800 |
| Contadores y auditores | 7 784 | 800 |
| Trabajadores de construcción | 5 910 | 800 |

- Solo imágenes **sin foto** (`has_photo == False`). Las features salen **solo del OCR**; la columna `text`
  solo se usa para medir el OCR (y, en la Pregunta 2, para medir su costo).
- **Nunca** son features: `age`, `sex`, `race`, nombres, `state`, `region`. `sex` y `race` solo se usan en la
  comprobación de equidad.
- **Target:** `exp_year` recortado a 0–20. Con tope 20 queda en el tope el ⟨x⟩ % de la muestra (< 15 %, así que
  se mantiene 20) ⟨o: «se sube a 25 porque…»⟩. Distribución por banda: [`output/tables/muestra_por_banda.csv`](output/tables/muestra_por_banda.csv).

## 3 · OCR (PaddleOCR) y orden de lectura

- **PaddleOCR 3.7.0 + PaddlePaddle-GPU 3.3.0** (CUDA 12.6) en Colab T4; PaddlePaddle 3.3.0 CPU en la app.
- **Modelos:** `PP-OCRv5_mobile_det` + `en_PP-OCRv5_mobile_rec`. Los CVs son renders digitales limpios en
  inglés, así que no hacen falta los modelos *server* (que rinden mejor en manuscritos y fotos). El reconocedor
  solo para inglés evita confusiones con otros alfabetos. Además la app vuelve a correr **el mismo** modelo
  en CPU, lo que evita un cambio de distribución entre entrenamiento y uso.
- Se apagan la clasificación de orientación, el *unwarping* y la orientación de línea (los renders vienen derechos).
- **Salida:** un JSON por CV con `text`, `conf` y `box` por línea (`data/raw/ocr_json/`, en Drive). El
  consolidado está en `data/processed/ocr.parquet`.
- **Orden de lectura** (`src/ocr.py`):
  1. Se busca un canal vertical entre el 25 % y el 75 % del ancho que cruce ≤ 5 % de las cajas y deje ≥ 20 % de las líneas a cada lado. Si existe, la página es a dos columnas.
  2. Las líneas que cruzan el canal (nombre, títulos a todo el ancho) cortan la página en franjas.
  3. En cada franja se lee primero la columna izquierda completa y luego la derecha.
  4. Las cajas a menos de media altura de distancia forman una fila, que se lee de izquierda a derecha.
- **Rendimiento:** ⟨s⟩ s por CV en ⟨GPU⟩, confianza media ⟨c⟩, ⟨n⟩ imágenes fallidas, ⟨p⟩ % de páginas a dos columnas
  ([`output/tables/ocr_resumen.json`](output/tables/ocr_resumen.json)).

## 4 · Calidad del OCR, limpieza y enmascaramiento

- **CER** ⟨x⟩ % y **WER** ⟨x⟩ % frente a `text`, desglosados por ocupación en
  [`output/tables/ocr_calidad_por_ocupacion.csv`](output/tables/ocr_calidad_por_ocupacion.csv).
- **Años de referencia leídos correctamente:** ⟨x⟩ %.
- **Umbral de confianza:** ⟨u⟩, el que minimiza el CER medio entre {0, 0.3, …, 0.9}
  ([`umbral_confianza.csv`](output/tables/umbral_confianza.csv)).
- **Limpieza:** se descartan las líneas bajo el umbral, se unen palabras partidas por guion y se borran correo,
  URL, teléfono, dirección y «Ciudad, ST» (solo con códigos de estado reales) y el nombre del propio CV.
- **Enmascaramiento** (`src/texto.py`): se reemplazan por `[MASK]` la experiencia total declarada, la edad y
  la fecha de nacimiento. Las fechas de empleo y los años de graduación se conservan. Se leyeron 50 textos OCR
  ([`output/ocr_muestra_50.txt`](output/ocr_muestra_50.txt)). Resultado: **⟨n⟩ CVs con alguna máscara**
  ([`enmascaramiento.json`](output/tables/enmascaramiento.json)).

## 5 · Split

70/15/15 por ID, estratificado por **ocupación × banda** (0–2, 3–5, 6–10, 11–20), con semilla 42.
Los IDs están en [`data/splits/`](data/splits/). Todos los modelos leen ese mismo split, y el test solo se abre
en el notebook 05.

## 6 · Modelos

| Modelo | Cómo se construye | Ajuste (solo con val) |
|---|---|---|
| Mediana | Mediana de `y` en train | — |
| TF-IDF + Ridge | Palabras 1–2 (+ caracteres 3–5 opcional), `sublinear_tf`, `min_df=2`, **con dígitos** (los años son la señal) | `alpha` y uso de caracteres |
| Embeddings + Ridge | `all-MiniLM-L6-v2` congelado. El CV se corta en trozos de 180 palabras con solape de 30 (dentro de su límite de 256 tokens) y se promedian los embeddings normalizados | `alpha` |
| **DistilBERT** | `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`, pérdida MSE, target ÷ 20, predicción ×20 recortada a [0, 20] | Early stopping por MAE de val |
| Regla de fechas (bonus) | Suma rangos de fechas laborales fusionando solapamientos | Año de «Present», calibrado en train |

**BERT:** lr 3e-5, batch 16, máximo 8 épocas (paciencia 2), longitud 512, fp16, semilla 42, en ⟨GPU⟩ durante ⟨min⟩ min.
- **CVs largos:** ⟨x⟩ % supera los 510 tokens. Se usa **cabeza + cola**: los primeros 382 tokens (historial
  laboral) más los últimos 128 (educación y año de graduación).
- Sobreviven el ⟨x⟩ % de los años, frente a ⟨y⟩ % si solo se usa la cabeza
  ([`truncamiento_sobrevivencia.csv`](output/tables/truncamiento_sobrevivencia.csv)).

## 7 · Resultados (test, n = 600)

Tabla completa: [`output/tables/model_comparison.csv`](output/tables/model_comparison.csv).

| Modelo | MAE | RMSE | ±1 | ±2 | ±3 | Acc. banda | Macro-F1 | R² | Entrenamiento | Inferencia/CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Mediana | ⟨⟩ | | | | | | | | | |
| TF-IDF + Ridge | ⟨⟩ | | | | | | | | | |
| Embeddings + Ridge | ⟨⟩ | | | | | | | | | |
| Regla de fechas | ⟨⟩ | | | | | | | | | |
| DistilBERT | ⟨⟩ | | | | | | | | | |

**Ganancia de BERT**, como MAE(baseline) − MAE(BERT), con IC 95 % por bootstrap pareado de 2 000
remuestreos ([`ganancia_bert_ic.csv`](output/tables/ganancia_bert_ic.csv)): ⟨…⟩.

Figuras: [predicho vs real](output/figures/pred_vs_real.png) · [MAE por banda y ocupación](output/figures/mae_banda_ocupacion.png) ·
[matrices de confusión](output/figures/matrices_confusion.png). Los 10 mayores errores de BERT, con su causa,
están en [`top10_errores_bert.csv`](output/tables/top10_errores_bert.csv).

**Mejoras propuestas:** ⟨…⟩

## 8 · Preguntas analíticas

### P1 · ¿Qué aprende BERT que los baselines no?
⟨Reducción de MAE frente a cada baseline y si es significativa. En qué bandas está la mejora. Prueba
contrafactual con 20 CVs de test ([`q1_perturbaciones.csv`](output/tables/q1_perturbaciones.csv)):
- **(A)** todas las fechas 5 años antes: un reclutador sumaría unos 5 años.
- **(B)** se quita el empleo más antiguo: un reclutador restaría su duración.

¿Qué modelo reacciona como un reclutador?⟩

### P2 · ¿Cuánto cuesta el ruido del OCR?
⟨CER/WER y dónde falla. MAE sobre el OCR frente al texto de referencia, por modelo
([`q2_ruido_ocr.csv`](output/tables/q2_ruido_ocr.csv)) y por cuartil de CER. Qué modelo es más robusto y por qué.⟩

### P3 · Sensibilidad metodológica
⟨Comparaciones ([`q3_sensibilidad.csv`](output/tables/q3_sensibilidad.csv)):
- truncamiento: cabeza vs cabeza + cola;
- pérdida de BERT: MSE vs Huber;
- dígitos en TF-IDF: con vs sin;
- limpieza del OCR: cruda, limpia o sin líneas de baja confianza.

Cuánto cambia el MAE de cada modelo.⟩

### P4 · Valor de negocio, equidad y ética
⟨Usuarios y tarea. Costo de errores de ±2 y ±5 años, y cuál es peor: sobreestimar puede colocar a un junior
en un puesto senior, subestimar puede descartar a alguien calificado. MAE y error medio de BERT por `sex` y
`race` ([`q4_equidad.csv`](output/tables/q4_equidad.csv)) y su posible explicación.⟩

## 9 · Limitaciones

- Los CVs son **sintéticos** y salen de plantillas: el OCR real de CVs escaneados será peor.
- Solo hay 5 ocupaciones, estadísticas de EE. UU. y CVs en inglés.
- El target viene del generador, no de una verificación laboral.
- El estimador **no debe decidir contrataciones**:
  - no ve calidad ni pertinencia de la experiencia;
  - hereda los sesgos del generador;
  - un error de ±5 años cambia la carrera de una persona.

## 10 · Reproducción paso a paso

**Requisitos:**
- Cuenta de Hugging Face, con las condiciones de `sumin-yu/PopResume` aceptadas y un token *Read*.
- Cuenta de Google con unos 5 GB libres en Drive.

| # | Notebook (Google Colab) | Entorno | Qué produce |
|---|---|---|---|
| 1 | `notebooks/01_muestreo_ocr.ipynb` | GPU T4 | Muestra, imágenes (en Drive), JSON de OCR, `ocr.parquet` |
| 2 | `notebooks/02_calidad_limpieza_split.ipynb` | CPU basta | CER/WER, textos limpios, máscara, `data/splits/` |
| 3 | `notebooks/03_baselines.ipynb` | GPU T4, **entorno nuevo** | Baselines y sensibilidad |
| 4 | `notebooks/04_bert_regresor.ipynb` | GPU T4 | DistilBERT y sensibilidad |
| 5 | `notebooks/05_evaluacion_preguntas.ipynb` | GPU T4 | Tablas, figuras, P1–P4 |

**Cómo correrlos:**
1. En Colab: *Archivo → Abrir notebook → GitHub* y pega la URL de este repo.
2. En 🔑 Secretos, crea `HF_TOKEN`. Si usas un fork, cambia `REPO_URL` en la celda de entorno.
3. Cada notebook clona el repo en `MyDrive/HW_04_202602` y retoma si Colab corta la sesión. Al final
   descarga un zip con las salidas livianas: descomprímelo en la raíz del repo local.
4. Empieza el notebook 3 en un entorno nuevo (*Desconectar y borrar entorno*), porque PaddlePaddle-GPU y PyTorch
   traen librerías CUDA distintas.

## 11 · Ejecutar la app

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

La app lee `output/` y `data/processed/`. El estimador en vivo necesita además los modelos:
1. Descarga `modelos_para_app.zip`, que genera el notebook 04.
2. Descomprímelo de modo que quede `models/bert_regresor/`, `models/tfidf_ridge.joblib`, etc.

La carpeta `models/` está en `.gitignore`: los pesos no van al repo.

Pestañas:
1. Problema y metodología.
2. OCR y análisis exploratorio.
3. Resultados de modelos.
4. Estimador interactivo: sube una imagen o un PDF, o explora los CVs de test.

## 12 · Video

Enlace en [`video/link.txt`](video/link.txt), de 5 minutos como máximo.

## Estructura

```
src/          config, data (muestreo/descarga), ocr (PaddleOCR + orden de lectura), texto (limpieza/máscara),
              reglas (fechas + perturbaciones), metricas (bootstrap), modelos (baselines + BERT)
notebooks/    01–05 (Colab)
app/          streamlit_app.py (4 pestañas)
data/         sample_ids.csv, splits/, processed/   (sin imágenes)
output/       tables/, figures/, predicciones/
video/        link.txt
```
