# Exploración multi_heatmap (rama `explore/multi-heatmap`)

Estructura mínima para explorar esta variante con **Google Colab**:

| Archivo | Contenido |
|---|---|
| `01_exploracion_datos.ipynb` | Conformación y exploración de la data: descarga eegbci, dataset multi-referencia, insumos multi-configuración balanceados (19/64/128/256), operadores de referencia y campo de superficie. |
| `02_modelo_entrenamiento.ipynb` | Construcción del modelo (`build_multiconfig_model`), entrenamiento (`train_variant`), validación en test (`evaluate_multiconfig`, `evaluate_multiconfig_surface`) y figuras. |

**Flujo Colab:** abrir en Colab → Runtime > Change runtime type > GPU (T4) →
ejecutar en orden. La caché de datos y los checkpoints se persisten en
Google Drive (`MiUnidad/universal_eeg_cache/`); el primer arranque clona este
repo (rama `explore/multi-heatmap`) e instala dependencias.

La configuración vive en `config/multi_heatmap.yaml` (cambiar a
`multi_heatmap_v2.yaml` en la celda `CONFIG` cuando toque). Los notebooks
reutilizan `runs/multi_heatmap/best.weights.h5` si existe; `FORCE = True`
reentrena desde cero.
