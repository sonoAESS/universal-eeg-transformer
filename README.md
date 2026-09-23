# Universal EEG Transformer

Autoencoder **lineal** multicabezal *All-to-All* que unifica la conversión entre
cuatro referencias de EEG — `unipolar`, `bipolar`, `CAR` y `REST` — a través de
un espacio latente central de dimensión `C`. Al ser puramente lineal, el mapeo
efectivo de cada ruta es la matriz `A_{s→d} = W^enc_s W^dec_d` (C×C), lo que
respeta la estructura algebraica de los campos electrostáticos del cuero
cabelludo.

Sobre ese núcleo existen tres familias:

* **Canónica** (`free`): las 16 rutas entre las 4 referencias en el montaje
  nativo (64 ch).
* **Unificación de montajes** (`montage_leadfield`, `montage_heatmap`): estimar
  las 4 referencias canónicas de 64 ch desde una configuración de electrodos
  distinta (p. ej. 10-20 de 19 ch) mediante una proyección fija `P`.
* **Multi-configuración** (`multi_montage`, `multi_heatmap`, `multi_heatmap_v2`,
  `universal_refs`): **un solo modelo** que acepta 19/64/128/256 electrodos
  (10-20, canónico, densos simulados) y predice las 4 referencias
  intra-configuración; `multi_heatmap*` añade el campo de superficie (heatmap)
  sobre la malla compartida como salida entrenada, y `universal_refs` una cabeza
  temporal de residuo (conv/gru/lstm/rnn).

## Física y convenciones

* **Convención de matrices:** `X_ref = X @ M`, con `X` de forma `(T, C)` y `M` la
  matriz de referencia que actúa por filas.
* **`unipolar`:** resta del canal de referencia (`Cz`). **`bipolar`:** resta del
  canal siguiente (cadena, rango C−1). **`CAR`:** resta de la media de todos los
  canales. **`REST`:** paso a una referencia al infinito virtual, calculada con
  el *lead field* analítico.
* **Referencia del dato original:** `eegbci` graba contra la mastoides izquierda
  (`data.original_reference`). Se revisa y audita en el pipeline, pero los cuatro
  montajes son invariantes a ella: todos eliminan el offset constante instantáneo
  (diferencias en uni/bip, `W_avg` en CAR/REST), así que el entrenamiento no
  depende de la referencia de adquisición.
* **Lead field:** esfera concéntrica de 4 capas (cerebro/CSF/cráneo/piel) vía MNE
  (`make_sphere_model`), con radios relativos `[0.87, 0.9, 0.97, 1.0]` y
  conductividades `[0.33, 1.0, 0.0042, 0.33]` S/m, y grilla de dipolos de 10 mm.
* **Matriz REST (row-application):** `M_rest = W_avg (G (W_avg G)⁺)ᵀ`, con `G` el
  lead field y `W_avg = I − 11ᵀ/C`. Es exacta para fuentes en el *rowspace* de
  `W_avg G`; unipolar y CAR se recuperan de forma exacta, mientras que bipolar
  pierde la componente constante.
* **Pérdida:** MSE estandarizado con Z-score del objetivo por lote para equilibrar
  las rutas independientemente de su escala.
* **Inicialización lineal empírica:** antes de entrenar, `W^enc_s = ridge(X_s→z)`
  y `W^dec_d = ridge(z→X_d)` con `z = X_unipolar` (latente de dimensión `C`),
  calculadas sobre una submuestra de entrenamiento. Como las referencias provienen
  de la misma señal por operadores lineales, esta factorización deja cada ruta en
  `var_expl ≈ 1` desde la época 0; sin ella el gradiente queda atrapado en cuencas
  degeneradas del autoencoder lineal (producto Glorot) y el modelo colapsa a
  predecir la media (`r ≈ 0`).

## Instalación

Python 3.12+, gestión con [`uv`](https://docs.astral.sh/uv/); la fuente de verdad
de dependencias es `pyproject.toml` (ya no existe `requirements.txt`):

```bash
uv sync                          # crea entorno e instala dependencias
```

Con el entorno ya creado (`entorno/`), para ejecutar código y tests:

```bash
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli --help
```

Los tests usan datos sintéticos pequeños y pasan sin GPU ni descargas.

## Uso

La CLI (`eeg-transform`) tiene siete comandos; `--config` alude a un YAML (ver
`config/`):

```bash
eeg-transform -c config/smoke.yaml build        # descarga PhysioNet eegbci y cachea dataset + lead field
eeg-transform -c config/smoke.yaml train        # entrena el transformador lineal
eeg-transform -c config/smoke.yaml eval         # métricas de test y figuras
eeg-transform -c config/*.yaml pipeline         # build + train + eval
eeg-transform compare runs/multi_montage runs/multi_heatmap runs/universal_refs   # tabla comparativa de ejecuciones
eeg-transform montage [--config config/montage_heatmap.yaml] [--montages 10-20,10-10]  # reconstrucción de montajes
eeg-transform experiment-extrapolacion         # experimento sintético (no usa dataset ni config)
```

* `build`: descarga los sujetos/corridas indicados, filtra (bandpass y notch),
  marca artefactos y canales malos por MAD z-score, revisa la referencia del dato
  original (`original_reference`) y construye las 4 referencias alineadas
  (unipolar, bipolar, CAR, REST), guardando `data/processed/dataset_*.npz`.
  Splits por bloques temporales (`block`) o por sujeto (`subject`).
* `train`: escribe checkpoints en `runs/<nombre>/` (`best.weights.h5`,
  `model.keras`, `history.csv`).
* `eval`: imprime tabla de métricas por ruta (MSE/RMSE/MAE/r), compara las
  matrices efectivas con las analíticas y guarda figuras en `runs/<nombre>/figs/`.
* `experiment-extrapolacion`: estudio sintético de extrapolación de la spline
  esférica (y del lead field) a un hemisferio ciego; metodología en
  `docs/experimento_extrapolacion_hemisferio.md`.

## Resultados principales

Detalle completo y comparativa: `docs/results_comparison.md` (métricas de test,
sujetos 1–12 de `eegbci`, split temporal por bloques).

| Familia | Variante | rmse cross (µV) | r cross | nota |
|---|---|---|---|---|
| Canónica | `free` | 0.66 | 0.9997 | 16 rutas entre 4 referencias (64 ch) |
| Montaje | `montage_heatmap` | 16.26 | 0.746 | 10-20 (19 ch) → canónico 64, spline |
| Montaje | `montage_leadfield` | 18.70 | 0.623 | ídem, solución inversa |
| Multi | `multi_montage` | 7.6–14.5 | 0.59–0.94 | 10-20/canonical/dense-128/256 |
| Multi + campo | `multi_heatmap` | — | 0.93 (10-20) | igual + `surface_loss` sobre malla |

Conclusiones generales en `docs/results_comparison.md` y física/métricas en
`docs/guia_conceptual.md`.

## Notebooks por variante

Cada variante tiene un **notebook autocontenido** en `notebooks/` que explica la
metodología y la arquitectura (markdown), carga el dataset cacheado, entrena o
**reutiliza el checkpoint** existente, y muestra las mismas visualizaciones
(curvas de aprendizaje, heatmap de RMSE por ruta, trazas reales vs predichas)
directamente en las celdas. Reutilizan el checkpoint por defecto (segundos); pon
`FORCE = True` solo para reentrenar desde cero.

| Notebook | Configuración | Variante |
|---|---|---|
| `notebooks/montage_leadfield.ipynb` | `config/montage_leadfield.yaml` | unificación por solución inversa (lead field) |
| `notebooks/montage_heatmap.ipynb` | `config/montage_heatmap.yaml` | unificación por mapas de calor (spline) |
| `notebooks/multi_montage.ipynb` | `config/multi_montage.yaml` | entrenamiento conjunto y balanceado sobre 19/64/128/256 electrodos |
| `notebooks/multi_heatmap.ipynb` | `config/multi_heatmap.yaml` | igual + campo de superficie (el topomapa es una salida entrenada) |
| `notebooks/multi_heatmap_v2.ipynb` | `config/multi_heatmap_v2.yaml` | igual + campo aprendible y consistencia física cruzada |
| `notebooks/universal_refs.ipynb` | `config/universal_refs.yaml` | igual + cabeza temporal de residuo (conv/gru/lstm/rnn) |

Además hay dos exploraciones que **se editan a mano** (no se regeneran desde
`generate_notebooks.py`) en `notebooks/exploraciones/`:

* `topomap_refs/` — enfoque **topomapa / grilla universal**: toda señal como
  campo de superficie interpolado para derivar cualquier distribución y cambiar
  la referencia de forma conjunta con el montaje (incluye la referencia
  `average_all`). Ver su `README.md`.
* `universal_refs_colab/` — notebooks de origen de la variante `universal_refs`
  (cabeza temporal), ejecutados en Colab.

Los notebooks de montaje estiman las 4 referencias canónicas (64 canales) desde
**cualquier configuración de electrodos** (p. ej. 10-20, 19 canales) y muestran
la actividad como **mapas de calor del cuero cabelludo** (electrodos → manchas):
observación del montaje fuente → proyección analítica → modelo → verdad canónica.

`multi_montage` entrena **un solo modelo** que acepta grabaciones en cualquiera
de las configuraciones `10-20` (19 ch), el canónico (64 ch) y los densos
simulados `dense-128`/`dense-256` (mismo campo escalar REST interpolado) y
predice, **en esa misma configuración**, las medidas con otra referencia. Las
configuraciones están **balanceadas por construcción** (mismas muestras por
lote), la predicción es intra-configuración (`C_s → C_s`) y la evaluación se
compara contra la línea base analítica `T_d @ pinv(T_s)` del propio montaje
(columnas `*_ana`).

`multi_heatmap` extiende lo anterior con la lectura de la actividad como **campo
de superficie** sobre una **malla compartida** del cuero cabelludo (la
interpolación `S_s` de los topomapas): el modelo también se entrena para que el
*heatmap* predicho sea fiel, no solo cada electrodo por separado. La evaluación
guarda `metrics_surface_test.csv` (`rmse_field`/`r_field`/`ve_field` por ruta y
configuración).

Para abrirlos y ejecutarlos con el entorno del proyecto:

```bash
jupyter notebook notebooks/   # kernel Python 3 del entorno/ (ipykernel instalado)
```

Los notebooks se generan desde `notebooks/generate_notebooks.py` (nbformat);
para regenerarlos tras cambios en la lógica:

```bash
PYTHONPATH=src entorno/bin/python notebooks/generate_notebooks.py
```

## Configuraciones (`config/`)

| YAML | `model.variant` | propósito |
|---|---|---|
| `default.yaml` | `free` | núcleo canónico (config por defecto del CLI) |
| `smoke.yaml` | `free` | humo ejecutable en CPU local (dataset pequeño) |
| `montage_leadfield.yaml` | `montage_leadfield` | unificación de montajes por lead field |
| `montage_heatmap.yaml` | `montage_heatmap` | unificación de montajes por spline |
| `multi_montage.yaml` | `multi_montage` | multi-configuración 19/64/128/256 |
| `multi_heatmap.yaml` | `multi_heatmap` | multi-config + campo de superficie |
| `multi_heatmap_v2.yaml` | `multi_heatmap_v2` | igual + campo aprendible y consistencia |
| `universal_refs.yaml` | `universal_refs` | igual + cabeza temporal (`temporal_cell: conv`) |
| `universal_refs_gru.yaml` | `universal_refs` | + célula recurrente GRU |
| `universal_refs_cmp_{conv,gru,lstm,rnn}.yaml` | `universal_refs` | barrido comparativo de cabezas temporales |
| `universal_refs_smoke.yaml` | `universal_refs` | humo CPU local de `universal_refs` |

## Estructura

```text
├── pyproject.toml               # dependencias (fuente de verdad, no requirements.txt)
├── uv.lock
├── AGENTS.md, README.md
├── config/*.yaml                # una configuración por variante/experimento
├── notebooks/                   # un notebook por variante (metodología + resultados)
│   ├── generate_notebooks.py    # genera/actualiza los notebooks (nbformat)
│   ├── montage_leadfield.ipynb, montage_heatmap.ipynb
│   ├── multi_montage.ipynb, multi_heatmap.ipynb, multi_heatmap_v2.ipynb
│   ├── universal_refs.ipynb
│   └── exploraciones/           # editados a mano (NO regenerar)
│       ├── topomap_refs/        #   topomapa / grilla universal (3 notebooks + README)
│       └── universal_refs_colab/#   origen Colab de universal_refs (2 notebooks + README)
├── docs/
│   ├── guia_conceptual.md       # física, métricas y arquitectura (fuente conceptual)
│   ├── results_comparison.md    # tabla comparativa y análisis (sujetos 1–12)
│   ├── experimento_extrapolacion_hemisferio.md   # experimento sintético de extra-/interpolación
│   ├── model_variants.md        # [histórico] variantes canónicas retiradas
│   ├── mapping_wip.md           # [histórico] unificación de montajes (WIP)
│   └── experiments.md           # [histórico] experimentos multi-configuración
├── src/eeg_transform/
│   ├── config.py                # dataclasses + carga/validación YAML
│   ├── references.py            # matrices de referencia + REST
│   ├── leadfield.py             # lead field analítico (esfera 4 capas, MNE)
│   ├── mapping.py               # proyección entre montajes (spline/leadfield)
│   ├── data/loader.py           # descarga eegbci y preprocesado
│   ├── data/dataset.py          # dataset multi-referencia + splits + caché
│   ├── models/universal_transformer.py   # autoencoder lineal All-to-All (+ modo montaje)
│   ├── models/multi_montage.py  # autoencoder multi-configuración (P_s/Q_s fijos)
│   ├── models/multi_heatmap.py  # multi-config + campo de superficie (S_s → malla)
│   ├── training/trainer.py      # bucles/callbacks TF + build_multiconfig_model
│   ├── evaluation/{metrics,plots}.py
│   ├── experiments/{montage,multi,hemisphere}.py  # observaciones + experimento sintético
│   ├── nb.py                    # helpers compartidos para los notebooks
│   └── cli.py                   # build/train/eval/compare/pipeline/montage/experiment-extrapolacion
└── tests/
    ├── test_physics.py          # propiedades de las referencias y REST
    ├── test_config_ds.py        # config y dataset
    ├── test_montage.py          # proyección/modo montaje
    ├── test_multi.py            # generador y modelo multi-configuración
    ├── test_temporal.py         # cabeza temporal (conv/gru/lstm/rnn) + evaluador ventaneado
    └── test_integration.py      # el modelo aprende los mapas sobre datos
```

## Notas

* El dataset por defecto son los sujetos 1–4 de `eegbci` (PhysioNet); el primer
  `build` descarga los EDFs (~decenas de MB por sujeto). Los checkpoints viven en
  `runs/`.
* Para `universal_refs` con `temporal_window > 0`, la evaluación estándar
  alimenta tensores 2-D y **desactiva** la cabeza temporal: el beneficio se mide
  con la evaluación ventaneada (`metrics_windowed_test.csv` en el CLI).