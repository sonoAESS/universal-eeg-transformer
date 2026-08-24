# Exploración `universal_refs` (rama `explore/multi-heatmap`)

Modelo que convierte entre **7 referencias** — unipolar/Cz, linked
mastoides, linked lóbulos, bipolar (vecinos físicos), CAR, REST y
Laplaciano de superficie — siempre **intra-configuración**, sobre **cascos
100% reales** (subconjuntos exactos del canónico eegbci + bases BIDS
externas con sus electrodos nativos; sin geometrías simuladas). Núcleo
lineal instantáneo + **cabeza temporal de residuo** sobre ventanas
centradas offline.

| Archivo | Contenido |
|---|---|
| `01_exploracion_datos.ipynb` | Conformación/exploración: 7 referencias (trazas, operadores, potencial vs CSD en malla), estructura temporal que justifica la ventana dinámica, ingesta de bases externas. |
| `02_modelo_entrenamiento.ipynb` | Arquitectura híbrida (`build_multiconfig_model`, variante `universal_refs`), entrenamiento ventaneado (`train_variant`), evaluación 49 rutas vs línea base analítica, campo de superficie, diagnóstico espectral por bandas y **ablation** (`ABLATION = True`). |

**Flujo Colab:** abrir en Colab → Runtime > GPU (T4) → ejecutar en orden.
Caché y checkpoints persisten en Drive (`MiUnidad/universal_eeg_cache/`);
el primer arranque clona este repo (rama `explore/multi-heatmap`) e instala
dependencias.

**Configuración:** `config/universal_refs.yaml`.

* Cascos activos por defecto: `"10-20", "10-10", "canonical"`; para añadir
  una base externa real: censo con `tools/openneuro_census.py` → descarga
  BIDS → `data.external.build_external_dataset(...)` → añadir
  `"external:<etiqueta>"` a `mapping.configs`.
* `model.temporal_window = 0` desactiva la cabeza dinámica (modelo
  puramente instantáneo = ablation exacta).
* Reutiliza `runs/universal_refs/best.weights.h5` si existe;
  `FORCE = True` reentrena desde cero.
