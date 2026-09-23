# AGENTS.md — Universal EEG Transformer

Instrucciones para agentes de código (y humanos) que trabajan en este repositorio.

## Visión general

Autoencoder **lineal** multicabezal *All-to-All* que unifica la conversión entre
cuatro referencias de EEG — `unipolar`, `bipolar`, `CAR`, `REST` — mediante un
espacio latente central de dimensión C. Cada ruta efectiva es la matriz
`A_{s→d} = W^enc_s W^dec_d`. Sobre esa base existen variantes de unificación de
montajes (`montage_*`) y de entrenamiento **multi-configuración**
(`multi_montage`, `multi_heatmap`, `multi_heatmap_v2`): un solo modelo que
acepta 19/64/128/256 electrodos (10-20, canónico, densos simulados) y predice
las 4 referencias intra-configuración; `multi_heatmap` añade el campo de
superficie (heatmap) sobre una malla compartida como salida entrenada.

Física y metodología: `README.md` y `docs/guia_conceptual.md` (fuente de
verdad conceptual). Resultados comparativos: `docs/results_comparison.md`.

## Entorno y comandos

* Gestión con `uv`; la fuente de verdad de dependencias es `pyproject.toml`
  (NO usar `requirements.txt`, ya no existe). Python ≥ 3.12, TensorFlow/Keras 3.
* El venv del proyecto es `entorno/`. Todo se ejecuta con `PYTHONPATH=src`
  (layout `src/eeg_transform/`):

```bash
uv sync                                            # recrear entorno si hace falta
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q        # suite rápida (excluye payoff)
PYTHONPATH=src entorno/bin/python -m pytest tests/test_spec_XX_* -m payoff -q  # aceptación SDD
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli --help
eeg-transform -c config/smoke.yaml pipeline        # build + train + eval
eeg-transform experiment-extrapolacion             # experimento sintético (sin dataset/config)
PYTHONPATH=src entorno/bin/python notebooks/generate_notebooks.py   # regenerar notebooks
```

* Los tests usan datos sintéticos pequeños; deben pasar antes de cualquier
  commit. No requieren descargar eegbci.
* La primera ejecución de `build` descarga PhysioNet eegbci y cachea en
  `data/processed/`; los checkpoints viven en `runs/<variante>/`.

## Desarrollo guiado por especificación (SDD)

* Proceso en `docs/specs/spec-00-sdd-proceso.md`; specs semilla:
  `spec-01` (estabilizar `multi_heatmap_v2`), `spec-02` (reentrenar
  `universal_refs_{conv,gru}`), `spec-03` (gate de calidad en `topomap-video`),
  `spec-04` (reporte de rutas normalizado).
* Una spec vive en `docs/specs/spec-XX-<slug>.md` (secciones obligatorias:
  Contexto, Requisitos, Criterios de aceptación) y su test asociado en
  `tests/test_spec_<slug>.py`. La spec es la **fuente de verdad**; si la
  implementación contradice la spec, se reescribe la spec primero.
* Los **criterios de aceptación** que requieren artefactos de
  entrenamiento/evaluación van marcados `@pytest.mark.payoff` y se excluyen por
  defecto (`-m 'not payoff'`); se ejecutan explícitamente con `-m payoff` como
  verificación de aceptación.
* Prefijo de commit: `spec(XX): ...` (p. ej. `spec(01): ...`).

## Convenciones de código

* **Idioma:** español para docstrings, comentarios, markdown y mensajes de log;
  identificadores en inglés cuando el dominio ya los fija (p. ej. `leadfield`).
* **Convención matricial sagrada:** `X_ref = X @ M` con `X` de forma `(T, C)`;
  las matrices de referencia/proyección actúan **por filas** (`P_s`: C_s→C,
  `Q_s`: C→C_s, `S_s`: C_s→malla). Nunca transponer sin verificar esta regla.
* Estilo: type hints modernos (`list[str] | None`), `from __future__ import
  annotations`, docstrings NumPy-style, logging vía `logging_conf.get_logger`.
* Sin comentarios triviales; explicar el *porqué* físico/algebraico cuando aplique.
* Configuración: dataclasses en `config.py` + YAML en `config/`; una variante
  de modelo = un YAML + (normalmente) su notebook.

## Estructura relevante

```text
config/*.yaml                  # una configuración por variante/experimento
src/eeg_transform/
  references.py                # operadores uni/bip/CAR/REST (+ rest_rcond)
  leadfield.py                 # esfera 4 capas analítica (MNE)
  mapping.py                   # proyección spline/leadfield entre montajes, malla
  experiments/{montage,multi,hemisphere}.py  # observaciones por montaje, multi-config
                               #   balanceada y experimento sintético de extrapolación
  models/{universal_transformer,multi_montage,multi_heatmap}.py
  training/trainer.py          # bucles TF + build_multiconfig_model
  evaluation/{metrics,plots,topomap_video}.py  # métricas, renders y vídeo-topomapa
  nb.py                        # helpers compartidos de los notebooks (load_experiment,
                               #   train_variant, evaluate_*, plot_*)
  cli.py                       # build/train/eval/compare/pipeline/montage/
                               #   experiment-extrapolacion/topomap-video
config/topomap_video_*.yaml    # (re)entrenamiento de la familia para el vídeo-topomapa
tests/test_*.py                # physics, config_ds, montage, multi, temporal,
                               #   topomap_video, integration
notebooks/generate_notebooks.py + *.ipynb   # notebooks autogenerados (no editar a mano)
notebooks/exploraciones/       # topomap_refs/ y universal_refs_colab/ (editados a mano)
runs/<variante>/               # checkpoints, history.csv, métricas, figuras
```

## Notebooks

Se generan desde `notebooks/generate_notebooks.py` (nbformat): editar ahí la
lógica/regenerar, nunca el `.ipynb` directamente. Cada notebook reutiliza el
checkpoint de `runs/` salvo `FORCE = True`. Excepción: las exploraciones de
`notebooks/exploraciones/` (topomap_refs, universal_refs_colab) se editan a mano
y NO se regeneran.

## Rama actual: trabajos de topomapa / recurrencia (fusionados en `main`)

`explore/topomap-refs` se fusionó en `main` (`413d843`, con push a origin). Desde
esa base se construyó en `main` el bloque `feat(topomap_video)`. La rama aloja:

### 1. Topomapa / grilla universal (prueba conceptual, SOLO notebooks)

Exploración centrada en el enfoque **topomapa / grilla universal** (concepto de
`/home/aess/Proyectos/eeg_to_eccog_dl`): toda señal se representa como campo de
superficie interpolado y, desde ahí, se puede **derivar cualquier
distribución** (una 10-10 real → la 10-20, o desde otra distribución) y
**cambiar la referencia** de forma conjunta con el montaje. Objetivos:

* Montajes derivados por interpolación esférica (posiciones MNE
  `standard_1005`/`standard_1020`, `mapping.spherical_spline_matrix`); el
  canonical eegbci 64ch ya vive en `standard_1005` (loader.py).
* Topomapas estilo MNE (disco circular, medido negro / interpolado gris).
* Nueva referencia **`average_all`**: `M = I - w·1ᵀ` con `w = media sobre los
  píxeles del topomapa` (`mapping.scalp_grid_matrix`); hay más píxeles que
  canales, así que el promedio se acerca más al infinito físico
  (esperado CAR < avg_all < REST, sin invertir el leadfield).
* **Conversión montaje + referencia con un modelo** entrenado *dentro de los
  notebooks* (TF/Keras), p. ej. **10-10 bipolar → 10-20 monopolar Cz**,
  evaluada contra la línea base analítica (interpolación + re-referenciación).
* Probar con datos reales cacheados (eegbci 64ch, sujetos 1-2, 7 refs): sin
  descargas y sin Drive.

* **Solo notebooks**: NO se toca `src/`, NO hay scripts ni tests nuevos.
  Estructura: `notebooks/exploraciones/topomap_refs/`
  (`01_topomapas_y_montajes.ipynb`,
  `02_referencia_average_all.ipynb`, `03_modelo_montaje_referencia.ipynb`,
  `README.md`) — se editan a mano, NO se regeneran desde
  `generate_notebooks.py` (igual que `universal_refs_colab/`).
* Las referencias/montajes nuevos se definen DENTRO de los notebooks usando
  funciones existentes del paquete (`scalp_grid_matrix`, `linked_matrix`,
  `build_reference_matrix`, `inter_reference_matrix`, `spherical_spline_matrix`)
  respetando la convención matricial `X_ref = X @ M`.
* Caché/checkpoints: `data/processed/` (ya cacheado, incluye
  `mne_asa_montages.npz` con los grids ASA de MNE `standard_1005`/`standard_1020`)
  y `runs/topomap_refs/`. Para la grilla universal, los grids ASA se alinean al
  marco PhysioNet del canonical vía Procrustes por nombres compartidos (RMSE~0;
  las convenciones MNE y PhysioNet difieren solo por rotación/reflexión).
* Prefijo de commit del bloque notebook: `feat(topomap_refs): ...`.

### 2. Recurrencia en `universal_refs` (código en `src/`, esta misma rama)

La variante `universal_refs` (`MultiHeatmapTemporal`) añade a su núcleo lineal
instantáneo una **cabeza temporal de residuo** que predice la corrección
dinámica sobre una ventana de lo medido. `model.temporal_cell` selecciona la
arquitectura de la cabeza:

* `conv` (por defecto): bloques depthwise+pointwise sobre ventana **centrada**
  (`padding="same"`), acausal.
* `gru` / `lstm` / `rnn`: **una célula recurrente causal** (units =
  `temporal_channels`) — "una ventana de tiempo de lo que se está midiendo
  contra lo que se predice"; generaliza a secuencias más largas que la ventana
  de entrenamiento. `temporal_kernel` no aplica fuera de `conv`.

Puntos importantes:

* La evaluación estándar (`evaluate_multiconfig_routes`) alimenta tensores
  2-D y **desactiva** la cabeza. El beneficio temporal se mide con la
  **evaluación ventaneada** (`evaluate_multiconfig_windowed`,
  `predict_multiconfig_windowed`): ventanas causales deslizantes, salida del
  último paso, comparando los instantes con contexto completo. En el CLI se
  genera `metrics_windowed_test.csv` para `universal_refs` con
  `temporal_window > 0`; en notebooks vía `nb.evaluate_multiconfig_windowed`.
* Las cabezas se inicializan a cero: el arranque es idéntico al modelo
  lineal (ablation trivial) para cualquier `temporal_cell`.
* Variante de ejecución: `config/universal_refs_gru.yaml`
  (`temporal_cell: gru`, salidas en `runs/universal_refs_gru` — no está en esta
  máquina). Barrido comparativo de cabezas en `config/universal_refs_cmp_*.yaml`.
* Tests: `tests/test_temporal.py` (parametrizado por célula + evaluador
  ventaneado). Prefijo de commit de este bloque: `feat(universal_refs): ...`.

### 3. Vídeo-topomapa comparativo (código en `src/`, en `main`)

Bloque `feat(topomap_video)`: animación (GIF/MP4) que muestra, por ventana
temporal, el topomapa de la **referencia observada** y los de las **7 referencias
estimadas** por cada método, con métricas de cada ruta en la ventana causal
`[t-window+1, t]`. Todo vive en `src/eeg_transform/evaluation/topomap_video.py`
(registro `VIDEO_MODELS`, `SegmentData`, `predict_method`, `topomap_video_fig`,
`save_topomap_video`, `run_topomap_video`) y se lanza con
`eeg-transform topomap-video`.

* Métodos: línea base analítica `T_d·pinv(T_s)` (`inter_reference_matrix`)
  + modelos `free`, `multi_montage`, `multi_heatmap_v2`, `universal_refs_conv/gru`
  ((re)entrenados con `config/topomap_video_*.yaml`, sujetos 1-2, en
  `runs/topomap_video/<nombre>/`). `free` y `universal_refs` solo soportan la
  entrada `canonical` (64 ch); los `multi_*` también `10-20` (19 ch).
* Alineación: todas las predicciones quedan **alineadas a muestras** del
  segmento; los temporales (`_predict_causal`, `stride=1`) marcan `NaN` los
  instantes `< window-1`. Los marcos arrancan en `2·(window-1)` para que cada
  ventana causal tenga exactamente `window` muestras con predicción. Los
  `sliding_window_view` añaden la ventana como eje trailing → transponer a
  `(n_w, window, C)`.
* Salidas en `runs/topomap_video/figs/`: `topomap_video_{input}_{source}.{gif,mp4}`
  + `*_metrics.csv` (marco, instante_s, ruta, rmse_uV, r, ve, método).
* Tests: `tests/test_topomap_video.py` (alineación, métricas, figura — datos
  sintéticos, sin entrenamiento ni descargas). Prefijo de commit
  `feat(topomap_video): ...`.

## Trabajo con ramas y commits

* Mensajes de commit en español, imperativos, con el prefijo de la variante
  cuando aplique (p. ej. `feat(multi_heatmap_v2): ...`, `docs: ...`).
* Exploración de una variante = rama propia; los artefactos grandes
  (`runs/`, `data/`) no se versionan (ver `.gitignore`).
