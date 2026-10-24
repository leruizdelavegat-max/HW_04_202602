# HW_04_202602 · Años de experiencia desde la imagen de un CV

**Estimar los años de experiencia laboral a partir de la imagen de un CV**, con PaddleOCR para leer y
cuatro modelos para estimar: mediana, TF-IDF + Ridge, embeddings + Ridge y DistilBERT ajustado como regresor
(más una regla de conteo de fechas como bonus).

> **En una línea:** DistilBERT logra **MAE 0,26 años** (99 % de los CVs a ±1 año) y supera con significancia a
> todos los baselines, pero una **regla que suma rangos de fechas logra MAE 0,02**. En PopResume la
> experiencia *es* la suma de las fechas: el modelo caro no se justifica aquí; la regla sí.

## 1 · Qué hace y para quién

Lee la **imagen** de un currículum, extrae el texto con OCR y predice `exp_year` (0–20 años). No es buscar
una frase: hay que combinar fechas de empleo, número de puestos, progresión de cargos y fin de estudios.

**Usuarios posibles:** bolsas de empleo y servicios públicos de empleo que reciben CVs escaneados y necesitan
**pre-llenar** el campo «años de experiencia»; equipos de RR. HH. que quieren **verificar** lo declarado por un
postulante; agencias que ordenan grandes volúmenes para revisión humana. **Nunca** para decidir una
contratación (§9).

## 2 · Datos y muestreo

- **PopResume** (Yu, Park & Moon, EMNLP 2026; CC BY 4.0; `sumin-yu/PopResume`): 60 884 CVs sintéticos de 5
  ocupaciones construidos con estadísticas de EE. UU.
- **Muestra: 800 CVs por ocupación = 4 000**, semilla **42**, estratificada dentro de cada ocupación por banda
  de experiencia (asignación proporcional). IDs en [`data/sample_ids.csv`](data/sample_ids.csv).

| Ocupación | Población | Muestra | 0–2 | 3–5 | 6–10 | 11–20 |
|---|---:|---:|---:|---:|---:|---:|
| Contadores y auditores | 7 784 | 800 | 119 | 222 | 275 | 184 |
| Trabajadores de construcción | 5 910 | 800 | 223 | 227 | 211 | 139 |
| Maestros de primaria/secundaria | 16 188 | 800 | 138 | 229 | 274 | 159 |
| Enfermeros/as registrados | 17 632 | 800 | 145 | 249 | 275 | 131 |
| Desarrolladores de software | 13 370 | 800 | 151 | 245 | 274 | 130 |
| **Total** | 60 884 | **4 000** | 776 | 1 172 | 1 309 | 743 |

- Solo imágenes **sin foto** (`has_photo == False`). Features **solo del OCR**; `text` se usa únicamente para
  medir el OCR (y su costo, Pregunta 2).
- **Nunca** son features: `age`, `sex`, `race`, nombres, `state`, `region`. `sex` y `race` viven aparte en
  `data/processed/auditoria.parquet` y solo se usan en la comprobación de equidad.
- **Target:** `exp_year` (continuo, en años) recortado a 0–20. Solo el **0,8 %** queda en el tope (< 15 %),
  así que se mantiene 20.

## 3 · OCR (PaddleOCR) y orden de lectura

- **PaddleOCR 3.7.0 + PaddlePaddle-GPU 3.3.0** (CUDA 12.6) en una **Tesla T4** de Colab; PaddlePaddle 3.3.0 CPU
  en la app.
- **Modelos:** `PP-OCRv5_mobile_det` + `en_PP-OCRv5_mobile_rec`. Los CVs son renders digitales limpios en
  inglés, donde *server* no aporta (su ventaja está en manuscritos y fotos); el reconocedor solo-inglés evita
  confusiones con otros alfabetos; y la app vuelve a correr **el mismo** modelo en CPU, sin cambio de
  distribución entre entrenamiento y uso.
- Salida: un JSON por CV (`text`, `conf`, `box` por línea) en Drive; consolidado en `data/processed/ocr.parquet`.
- **Orden de lectura** (`src/ocr.py`): se busca un canal vertical entre el 25 % y el 75 % del ancho que crucen
  ≤ 5 % de las cajas, con ≥ 20 % de las líneas a cada lado → dos columnas; las líneas que cruzan el canal
  parten la página en franjas; en cada franja, columna izquierda completa y luego derecha; dentro de una
  columna, filas (centros a < ½ altura) de izquierda a derecha. **En PopResume el 0 % de las páginas resultó a
  dos columnas**: las plantillas son de una columna, y el orden por filas basta.
- **Rendimiento:** **0,26 s por CV** (mediana 0,23) en la T4 · confianza media **0,991** · **0 imágenes
  fallidas** · 14,4 líneas por CV ([`ocr_resumen.json`](output/tables/ocr_resumen.json)).

## 4 · Calidad del OCR, limpieza y enmascaramiento

| Ocupación | CER | WER | Años de la referencia leídos |
|---|---:|---:|---:|
| Contadores y auditores | 1,01 % | 7,96 % | 100 % |
| Construcción | 0,87 % | 6,67 % | 100 % |
| Maestros | 0,91 % | 7,05 % | 100 % |
| Enfermería | 1,06 % | 7,78 % | 100 % |
| Software | 1,17 % | 8,56 % | 100 % |
| **Total** | **1,00 %** | **7,60 %** | **100 %** |

([`ocr_calidad_por_ocupacion.csv`](output/tables/ocr_calidad_por_ocupacion.csv), medido con `jiwer` tras minúsculas y espacios colapsados.)

- **Dónde falla el OCR:** casi todo el WER viene de **fechas pegadas**: el formato es
  `Cargo, Empresa, 2019.11–2023.01` y en el **71 %** de los CVs el OCR pierde la raya y lee `2019.112023.01`.
  Sin reparar, solo el **82 %** de los años de la referencia aparecía como año aislado; tras reinsertar la raya
  (`src/texto.py::reparar_fechas`), el **100 %**. Hallazgo sacado de leer los textos, no de las métricas.
- **Umbral de confianza: 0,0.** Se probaron 0–0,9 y el CER no cambia: ninguna línea baja de 0,9 de confianza,
  así que descartar líneas no aporta ([`umbral_confianza.csv`](output/tables/umbral_confianza.csv)).
- **Limpieza:** reparar fechas pegadas, unir palabras partidas por guion y borrar PII — correo, URL, teléfono,
  dirección, **estado** (`TX, USA`) y el nombre del propio CV (los nombres se usan solo para borrarlos).
- **Enmascaramiento:** se reemplazan por `[MASK]` la experiencia total declarada («12 years of experience»), la
  edad y la fecha de nacimiento; las fechas de empleo y los años de graduación se conservan. Se leyeron los 50
  textos de [`output/ocr_muestra_50.txt`](output/ocr_muestra_50.txt): **ninguno declara su experiencia total,
  edad ni fecha de nacimiento**, y los patrones aplicados a los 4 000 CVs encontraron **0 CVs afectados**
  ([`enmascaramiento.json`](output/tables/enmascaramiento.json)). La regla queda activa (la app la aplica a CVs reales).

## 5 · Split

70/15/15 por ID (**2 800 / 600 / 600**), estratificado por ocupación × banda (0–2, 3–5, 6–10, 11–20), semilla 42.
IDs en [`data/splits/`](data/splits/); todos los modelos leen el mismo split y el test solo se abre en el
notebook 05.

## 6 · Modelos

| Modelo | Construcción | Ajuste (solo validación) |
|---|---|---|
| Mediana | mediana de `y` en train | — |
| TF-IDF + Ridge | palabras 1–2 **+ caracteres 3–5**, `sublinear_tf`, `min_df=2`, **con dígitos** (los años son la señal) | `alpha = 0,3` (grid 0,3–10 × con/sin caracteres) |
| Embeddings + Ridge | `all-MiniLM-L6-v2` congelado; trozos de 180 palabras (solape 30) promediados | `alpha = 0,1` |
| **DistilBERT** | `distilbert-base-uncased`, `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`, MSE, target ÷ 20, salida × 20 recortada a [0, 20] | early stopping por MAE val |
| Regla de fechas (bonus) | suma de rangos laborales con solapamientos fusionados | — |

**BERT:** lr 3e-5 (AdamW, decaimiento 0,01, warmup 6 %), batch 16, máx. 8 épocas (paciencia 2), longitud 512,
fp16, semilla 42, **Tesla T4, 2,1 min**. Mejor MAE de validación: 0,248 (época 8: el early stopping no llegó a
activarse). **CVs largos:** con el tokenizador de DistilBERT los CVs miden **40–150 tokens**; **ninguno supera
512**, así que la estrategia cabeza + cola (382 + 128 tokens) no recorta nada y el 100 % de los años llega al
modelo ([`truncamiento_sobrevivencia.csv`](output/tables/truncamiento_sobrevivencia.csv), [`tokens_por_cv.png`](output/figures/tokens_por_cv.png)).

## 7 · Resultados (test, n = 600)

| Modelo | MAE | RMSE | ±1 | ±2 | ±3 | Acc. banda | Macro-F1 | R² | Entren. | Inferencia/CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mediana | 3,50 | 4,52 | 19 % | 33 % | 51 % | 33 % | 0,12 | −0,04 | 0 s | 0 ms |
| TF-IDF + Ridge | 0,90 | 1,18 | 64 % | 93 % | 98 % | 81 % | 0,81 | 0,93 | 2,8 s | 0,98 ms |
| Embeddings + Ridge | 1,33 | 1,73 | 46 % | 79 % | 91 % | 74 % | 0,75 | 0,85 | 4,6 s | 1,14 ms |
| **DistilBERT** | **0,26** | **0,35** | **99 %** | **100 %** | 100 % | **93 %** | **0,94** | **0,994** | 126 s | 1,16 ms |
| Regla de fechas (bonus) | 0,02 | 0,04 | 100 % | 100 % | 100 % | 99 % | 0,99 | 0,9999 | 0 s | 0,19 ms |

([`model_comparison.csv`](output/tables/model_comparison.csv); tiempos en la T4 de Colab.)

**Ganancia de BERT** = MAE(baseline) − MAE(BERT), IC 95 % por bootstrap pareado (2 000 remuestreos)
([`ganancia_bert_ic.csv`](output/tables/ganancia_bert_ic.csv)):

| frente a | ganancia (años) | IC 95 % | ¿significativa? |
|---|---:|---|---|
| Mediana | +3,24 | [3,00; 3,49] | sí |
| TF-IDF + Ridge | **+0,63** | [0,58; 0,70] | sí |
| Embeddings + Ridge | +1,07 | [0,98; 1,16] | sí |
| Regla de fechas | **−0,24** | [−0,26; −0,22] | sí (la regla es mejor) |

- **Predicho vs real** ([`pred_vs_real.png`](output/figures/pred_vs_real.png)): TF-IDF y embeddings muestran
  **regresión a la media** — sobreestiman a los juniors y se aplanan por encima de ~15 años (MAE 1,37 y 1,88 en
  la banda 11–20). BERT sigue la diagonal en todo el rango; la regla, casi perfecta.
- **MAE por banda** ([`mae_por_banda.csv`](output/tables/mae_por_banda.csv)): BERT 0,23 / 0,30 / 0,22 / 0,31; su
  confusión principal es 6–10 → 3–5 (19 CVs) en los bordes de banda ([`matrices_confusion.png`](output/figures/matrices_confusion.png)).
- **MAE por ocupación** ([`mae_por_ocupacion.csv`](output/tables/mae_por_ocupacion.csv)): BERT 0,23–0,29 en todas;
  TF-IDF sufre más en maestros (1,06) y enfermería (1,03).
- **Los 10 mayores errores de BERT** ([`top10_errores_bert.csv`](output/tables/top10_errores_bert.csv)) son todos
  de ~1 año y en ninguno falla el OCR (CER 1 %, 100 % de años leídos, una columna). 4 son CVs con **hueco
  laboral** (BERT parece estimar el lapso total y no la suma de empleos) y 6 «otra», casi todos
  sobreestimaciones de ~+1 año (3 carreras de 18+ años y 3 juniors de menos de 4 años) — imprecisión de la
  aritmética de fechas, no del OCR: la regla acierta en los 10 casos con error < 0,05.

**Mejoras propuestas:** (1) en producción, usar la **regla de fechas** como estimador principal y BERT solo
como respaldo cuando la regla no encuentre rangos (formatos raros, fechas ilegibles); (2) entrenar más épocas
(el early stopping no se activó); (3) dar a BERT las duraciones ya calculadas como texto («[3,2 años]») para
no exigirle aritmética; (4) aumentar con plantillas de CV reales y otros formatos de fecha antes de desplegar.

## 8 · Preguntas analíticas

### P1 · ¿Qué aprende BERT que los baselines no?

BERT reduce el MAE en **0,63 años frente a TF-IDF** y **1,07 frente a embeddings**, ambas diferencias
significativas, y la mejora se concentra en los **extremos**: en 11–20 años pasa de 1,37 (TF-IDF) a 0,31,
sin la regresión a la media de los baselines. Lee *estructura*: cuántos empleos hay, cuándo, y qué tan
senior suenan los cargos.

Pero la **prueba contrafactual** (20 CVs de test con 2+ empleos, [`q1_perturbaciones.csv`](output/tables/q1_perturbaciones.csv))
muestra que **no razona como un reclutador**:

| Perturbación | Reclutador | TF-IDF | Embeddings | **BERT** | Regla |
|---|---:|---:|---:|---:|---:|
| A · todas las fechas +5 años (mismas duraciones) | 0,00 | −1,61 | +0,11 | **−2,77** | 0,00 |
| B · el primer empleo empieza 5 años antes | +4,39 | +0,37 | +0,11 | **+0,36** | +4,39 |
| C · sin el empleo más antiguo | −3,01 | −1,77 | −2,78 | **−3,14** | −3,01 |

- **C:** BERT acierta (−3,14 frente a −3,01, 100 % en la dirección correcta): cuenta empleos y sus bloques.
- **A:** BERT **cae 2,8 años** cuando las mismas carreras se corren al futuro: usa **años absolutos** como atajo
  («2009» ⇒ senior), no duraciones.
- **B:** BERT **casi no reacciona** (+0,36 de +4,4 esperados): no está restando fechas.

Conclusión: BERT aprende correlaciones muy buenas *dentro de la distribución* (cuántos empleos, qué años, qué
verbos —«Directed» frente a «Assisted with»—), pero no la aritmética que define el target. La regla sí la hace
por construcción.

### P2 · ¿Cuánto cuesta el ruido del OCR?

El OCR es muy bueno (CER 1,0 %, WER 7,6 %, confianza 0,99) y, **tras reparar las fechas pegadas**, recupera el
100 % de los años. Por eso el costo del ruido es **prácticamente nulo** ([`q2_ruido_ocr.csv`](output/tables/q2_ruido_ocr.csv)):

| Modelo | MAE con OCR | MAE con `text` | brecha |
|---|---:|---:|---:|
| TF-IDF + Ridge | 0,898 | 0,905 | −0,01 |
| Embeddings + Ridge | 1,330 | 1,355 | −0,03 |
| DistilBERT | 0,263 | 0,266 | −0,00 |
| Regla de fechas | 0,023 | 0,021 | +0,00 |

La brecha incluso es levemente negativa para los modelos aprendidos: se entrenaron con texto OCR y la
referencia les resulta un poco «fuera de distribución» (otro separador de fechas, viñetas). El MAE tampoco
crece en el cuartil de peor CER ([`q2_mae_por_cuartil_cer.csv`](output/tables/q2_mae_por_cuartil_cer.csv)).
**El costo real del OCR estaba en un solo error sistemático** — la raya perdida en el 71 % de los CVs— y se
neutralizó con una línea de post-proceso; sin esa reparación, la regla no encontraba ninguna fecha. El más
robusto es la **regla** (lee solo fechas, y estas se leen al 100 %); entre los aprendidos, **BERT**.

### P3 · Sensibilidad metodológica

([`q3_sensibilidad.csv`](output/tables/q3_sensibilidad.csv), MAE de test.)

| Decisión | Base | Alternativa | Efecto |
|---|---|---|---|
| Dígitos en TF-IDF | con dígitos: **0,898** | sin dígitos (`#`): **1,862** | **+107 %**: la mitad de la señal de TF-IDF son los años |
| Limpieza del OCR (TF-IDF) | limpio: 0,898 | crudo (sin reparar fechas ni PII): 0,959 | +7 % |
| Limpieza del OCR (embeddings) | limpio: 1,330 | crudo: 1,492 | +12 % |
| Limpieza (umbral de confianza) | 0,0 | 0,3–0,9 | sin cambio (ninguna línea < 0,9) |
| Truncamiento BERT | cabeza + cola: 0,263 | cabeza: 0,262 | ninguno (ningún CV supera 512 tokens) |
| Pérdida BERT | MSE: 0,263 | Huber (δ = 1 año): 0,266 | ninguno |

Las decisiones que importan son **de datos** (conservar los años, reparar las fechas), no de arquitectura.

### P4 · Valor de negocio, equidad y ética

- **Quién y para qué:** un servicio público de empleo o una bolsa de trabajo que digitaliza CVs escaneados
  podría **pre-llenar** «años de experiencia» y **marcar discrepancias** con lo declarado, siempre con revisión
  humana.
- **Costo de los errores:** ±2 años casi nunca cambia de banda salarial ni de requisito («3+ años»); ±5 años
  sí (junior ↔ senior). **Sobreestimar es peor** en un filtro de requisitos: deja pasar a quien no cumple y
  desplaza a quien sí cumple. Subestimar descarta en silencio a personas calificadas —también grave, y más
  difícil de detectar porque nadie revisa a los rechazados—. Con BERT, el 100 % de los errores de test está
  dentro de ±2 años; con TF-IDF, el 7 % no.
- **Equidad** ([`q4_equidad.csv`](output/tables/q4_equidad.csv), BERT, test):

| Grupo | n | MAE [IC 95 %] | error medio (pred − real) | años reales (media) |
|---|---:|---|---:|---:|
| Female | 303 | 0,263 [0,239; 0,289] | −0,03 | 6,8 |
| Male | 297 | 0,264 [0,237; 0,291] | −0,01 | 6,3 |
| Asian/Pacific Islander | 88 | 0,251 [0,212; 0,290] | −0,02 | 6,8 |
| Black | 50 | **0,348** [0,276; 0,419] | +0,07 | **8,0** |
| White | 462 | 0,257 [0,237; 0,277] | −0,03 | 6,3 |

  Por **sexo** no hay diferencia. Por **raza**, el MAE del grupo Black es mayor (0,35 frente a 0,26), con IC que
  apenas se tocan y n = 50. La explicación más probable no es la raza en sí —no es feature y los nombres se
  borran—, sino que este grupo tiene **más experiencia promedio** (8,0 años), y BERT se equivoca más en carreras
  largas (MAE 0,31 en 11–20). Igual debe monitorearse con más datos; un sesgo de error medio de +0,07 años es
  operacionalmente despreciable.
- **Limitaciones:** ver §9.

## 9 · Limitaciones

CVs **sintéticos** de plantilla (el OCR real de escaneos será peor; ningún CV a dos columnas), 5 ocupaciones,
estadísticas de EE. UU., solo inglés, un único formato de fecha. El target lo genera el mismo sistema que las
fechas (por eso la regla es casi perfecta); en CVs reales la experiencia declarada y la real divergen. **Nunca
debe decidir una contratación**: no mide calidad ni pertinencia de la experiencia, hereda los sesgos del
generador y un error de ±5 años cambia la carrera de una persona. Uso aceptable: pre-llenado y alerta, con una
persona que decide.

## 10 · Reproducción paso a paso

Requisitos: cuenta de Hugging Face con las condiciones de `sumin-yu/PopResume` aceptadas y un token *Read*;
cuenta de Google con ~5 GB en Drive.

| # | Notebook (Google Colab) | Entorno | Produce |
|---|---|---|---|
| 1 | `notebooks/01_muestreo_ocr.ipynb` | GPU T4 | muestra, imágenes (Drive), JSON de OCR, `ocr.parquet` |
| 2 | `notebooks/02_calidad_limpieza_split.ipynb` | CPU basta | CER/WER, textos limpios, máscara, `data/splits/` |
| 3 | `notebooks/03_baselines.ipynb` | GPU recomendable, **entorno nuevo** | baselines + sensibilidad |
| 4 | `notebooks/04_bert_regresor.ipynb` | **GPU T4** | DistilBERT + sensibilidad |
| 5 | `notebooks/05_evaluacion_preguntas.ipynb` | GPU T4 | tablas, figuras, P1–P4 |

1. Colab → *Archivo → Abrir notebook → GitHub* → este repo, rama `feature/pipeline` (o `main` tras el merge; cambiar `RAMA`).
2. 🔑 Secretos: `HF_TOKEN` (solo lo usa el notebook 01).
3. Cada notebook clona el repo en `MyDrive/HW_04_202602`, actualiza **solo el código** al ejecutarse (datos y
   resultados en Drive no se tocan) y retoma si Colab corta la sesión. Al final descarga un zip con las salidas
   livianas.
4. El notebook 03 se abre en un **entorno nuevo**: PaddlePaddle-GPU (notebook 01) cambia la librería NCCL que usa
   PyTorch.
5. Verificación local sin GPU: `src/` es importable y los notebooks 02–05 corren en CPU (BERT tarda horas).

## 11 · Ejecutar la app

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

`app.py` (raíz) es el punto de entrada que pide el enunciado; la app está en `app/streamlit_app.py` y sus
componentes visuales en `app/ui.py`. No necesita GPU: OCR y modelos corren en CPU. Tras editar el código hay que
reiniciar la app (`.streamlit/config.toml` desactiva el vigilante de archivos).

La app lee `output/` y `data/processed/` (incluidos en el repo). Los indicadores se calculan sobre esos archivos; si
falta uno, la sección lo indica como pendiente en vez de mostrar un valor. El **estimador en vivo** necesita además los
modelos: descarga `models/` desde `MyDrive/HW_04_202602/models/` (o `modelos_para_app.zip` del notebook 04)
para que queden `models/bert_regresor/`, `models/tfidf_ridge.joblib`, etc. `models/` está en `.gitignore`:
los pesos no van al repo.

Pestañas: (1) Problema y metodología · (2) OCR y análisis exploratorio · (3) Resultados de modelos ·
(4) Estimador interactivo (subir imagen o PDF, o explorar CVs de test por ocupación, años reales y tamaño del error).

**Privacidad del estimador** (`src/anonimizar.py`): cada línea del OCR se clasifica como personal (nombre,
encabezado, correo, teléfono, URL, dirección, documento, secciones de contacto) o no. Los modelos solo reciben las
líneas no personales. La imagen se muestra **reconstruida** sobre un lienzo blanco con únicamente las cajas de texto
no personales; las personales son barras oscuras y lo que el OCR no detecta (fotos, firmas, logos) no aparece. El
archivo subido no se guarda (el PNG temporal del OCR se borra al terminar) y los errores no muestran texto del CV.

## 12 · Video

Enlace en [`video/link.txt`](video/link.txt) (≤ 5 minutos).

## Estructura

```
src/          config, data (muestreo/descarga), ocr (PaddleOCR + orden de lectura), texto (limpieza, fechas, máscara),
              reglas (regla de fechas + perturbaciones), metricas (bootstrap), modelos (baselines + BERT),
              anonimizar (líneas personales, imagen protegida), estimador (pipeline del CV subido)
notebooks/    01–05 (Colab)
app.py        punto de entrada:  streamlit run app.py
app/          streamlit_app.py (4 pestañas), ui.py (estilo y componentes)
data/         sample_ids.csv, splits/, processed/   (sin imágenes)
output/       tables/, figures/, predicciones/, ocr_muestra_50.txt
video/        link.txt
```
