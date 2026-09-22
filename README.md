# Universal EEG Transformer

Autoencoder **lineal** multicabezal *All-to-All* que unifica la conversión entre cuatro referencias de EEG — `unipolar`, `bipolar`, `CAR` y `REST` — a través de un espacio latente central. Al ser puramente lineal, el mapeo efectivo de cada ruta es la matriz `A_{s→d} = W^enc_s W^dec_d` (C×C), lo que respeta la estructura algebraica de los campos electrostáticos del cuero cabelludo.

## Física y convenciones

* **Convención de matrices:** `X_ref = X @ M`, con `X` de forma `(T, C)` y `M` la matriz de referencia que actúa por filas.
* **`unipolar`:** resta del canal de referencia (`Cz`). **`bipolar`:** resta del canal siguiente (cadena, rango C−1). **`CAR`:** resta de la media de todos los canales. **`REST`:** paso a una referencia al infinito virtual, calculada con el *lead field* analítico.
* **Referencia del dato original:** `eegbci` graba contra la mastoides izquierda (`data.original_reference`). Se revisa y audita en el pipeline, pero los cuatro montajes son invariantes a ella: todos eliminan el offset constante instantáneo (diferencias en uni/bip, `W_avg` en CAR/REST), así que el entrenamiento no depende de la referencia de adquisición.
* **Lead field:** esfera concéntrica de 4 capas (cerebro/CSF/cráneo/piel) vía MNE (`make_sphere_model`), con radios relativos `[0.87, 0.9, 0.97, 1.0]` y conductividades `[0.33, 1.0, 0.0042, 0.33]` S/m, y grilla de dipolos de 10 mm.
* **Matriz REST (row-application):** `M_rest = W_avg (G (W_avg G)⁺)ᵀ`, con `G` el lead field y `W_avg = I − 11ᵀ/C`. Es exacta para fuentes en el *rowspace* de `W_avg G`; unipolar y CAR se recuperan de forma exacta, mientras que bipolar pierde la componente constante.
* **Pérdida:** MSE estandarizado con Z-score del objetivo por lote para equilibrar las 16 rutas independientemente de su escala.
* **Inicialización lineal empírica:** antes de entrenar, `W^enc_s = ridge(X_s→z)` y `W^dec_d = ridge(z→X_d)` con `z = X_unipolar` (latente de dimensión `C`), calculadas sobre una submuestra de entrenamiento. Como las referencias provienen de la misma señal por operadores lineales, esta factorización deja cada ruta en `var_expl ≈ 1` desde la época 0; sin ella el gradiente queda atrapado en cuencas degeneradas del autoencoder lineal (producto Glorot) y el modelo colapsa a predecir la media (`r ≈ 0`).

## Resultados (sujetos 1–12 de eegbci, split `block`)

Métricas de test tras el ajuste fino (latente = C = 64, 16 rutas, pérdida
Z-score). Detalle completo y comparativa: `docs/results_comparison.md`.

| Variante | RMSE diag. (µV) | RMSE cross (µV) | r cross | error comp. medio | error comp. máx |
|---|---|---|---|---|---|
| **projected** (recomendado) | **0.26** | **0.28** | **1.0000** | 0.034 | 0.190 |
| free (por defecto) | 0.69 | 0.66 | 0.9997 | 0.067 | 0.363 |
| soft_group | 0.35 | 0.92 | 0.9990 | **0.005** | **0.040** |
| group (estructura exacta) | 133.4 | 133.5 | 0.709 | 0.000 | 0.000 |
| analítico `T_d pinv(T_s)` | 133.4 | 134.2 | 0.735 | 0.593 | 8.563 |

Conclusiones:

* **`projected`** da la mejor precisión (RMSE cruzado ~0.28 µV, `r_cross ≈ 1`)
  y respeta la física (anula por construcción el modo constante).
* **`soft_group`** impone la transitividad `s→d→u` como penalización suave y
  logra un error de composición ~120× menor que el encadenado analítico a
  cambio de un RMSE cruzado de ~0.9 µV.
* **`group`** (composición exacta) y el encadenado analítico fallan igual:
  ~133 µV RMSE. La pseudo-inversa rígida asume que todas las referencias
  comparten el mismo subespacio observable, y el de REST no coincide con el
  de uni/bip/CAR.
* El latente óptimo es `C`: `wide` (128) y `bottleneck` (32) empeoran.
* `error_fro_rel` ~1 en rutas REST en todos los modelos no implica mala
  predicción: la matriz REST analítica no es identificable únicamente desde
  el dato, y el modelo aprende un mapeo datos-equivalente.
* Más datos (12 vs 4 sujetos) refinaron los mapas: `free` sube `r_cross` de
  0.9987 a 0.9997 y `rest→bipolar` de `r ≈ 0.99` a 0.9977.

> Nota sobre `error_fro_rel`: se compara en el **subespacio observable** (`P = I − 11ᵀ/C` por ambos lados) porque el modo constante instantáneo no es recuperable desde ninguna referencia; sin ese centrado la métrica no refleja la calidad real del mapeo.

## Requisitos e instalación

Python 3.12+, gestión con [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                          # crea entorno e instala dependencias
uv run --with ... python -m pytest tests/   # o usar un venv existente
```

Con el entorno ya creado (`entorno/`), para ejecutar código y tests:

```bash
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli --help
```

La fuente de verdad de dependencias es `pyproject.toml` (ya no se usa `requirements.txt`).

## Uso

La CLI (`eeg-transform`) tiene cuatro comandos; `--config` alude a un YAML (ver `config/default.yaml`):

```bash
eeg-transform -c config/smoke.yaml build        # descarga PhysioNet eegbci y cachea dataset + lead field
eeg-transform -c config/smoke.yaml train        # entrena el transformador lineal
eeg-transform -c config/smoke.yaml eval         # métricas de test y figuras
eeg-transform -c config/*.yaml pipeline         # build + train + eval
eeg-transform compare runs/multi_montage runs/multi_heatmap runs/universal_refs   # tabla comparativa de ejecuciones
```

## Notebooks por variante

Cada variante tiene un **notebook autocontenido** en `notebooks/` que explica la
metodología y la arquitectura (markdown), carga el dataset cacheado, entrena o
**reutiliza el checkpoint** existente, y muestra las mismas visualizaciones
(curvas de aprendizaje, heatmap de RMSE por ruta, trazas reales vs predichas)
directamente en las celdas. Reutilizan el checkpoint por defecto (segundos);
pon `FORCE = True` solo para reentrenar desde cero.

| Notebook | Configuración | Variante |
|---|---|---|
| `notebooks/montage_leadfield.ipynb` | `config/montage_leadfield.yaml` | unificación por solución inversa (lead field) |
| `notebooks/montage_heatmap.ipynb` | `config/montage_heatmap.yaml` | unificación por mapas de calor (spline) |
| `notebooks/multi_montage.ipynb` | `config/multi_montage.yaml` | entrenamiento conjunto y balanceado sobre 19/64/128/256 electrodos |
| `notebooks/multi_heatmap.ipynb` | `config/multi_heatmap.yaml` | igual + campo de superficie (el topomapa es una salida entrenada) |
| `notebooks/multi_heatmap_v2.ipynb` | `config/multi_heatmap_v2.yaml` | igual + campo aprendible y consistencia física cruzada |
| `notebooks/universal_refs.ipynb` | `config/universal_refs.yaml` | igual + cabeza temporal de residuo (conv/gru/lstm/rnn) |

Los notebooks de montaje estiman las 4 referencias canónicas (64 canales) desde
**cualquier configuración de electrodos** (p. ej. 10-20, 19 canales) y muestran
la actividad como **mapas de calor del cuero cabelludo** (electrodos → manchas):
observación del montaje fuente → proyección analítica → modelo → verdad canónica.

El de `multi_montage` entrena **un solo modelo** que acepta grabaciones en
cualquiera de las configuraciones `10-20` (19 ch), el canónico (64 ch) y los
densos simulados `dense-128`/`dense-256` (mismo campo escalar REST interpolado)
y predice, **en esa misma configuración**, las medidas con otra referencia.
Las configuraciones están **balanceadas por construcción** (mismas muestras por
lote), la predicción es intra-configuración (`C_s → C_s`) y la evaluación se
compara contra la línea base analítica `T_d @ pinv(T_s)` del propio montaje
(columnas `*_ana`). Física y métricas en `docs/guia_conceptual.md`.

La variante `multi_heatmap` extiende lo anterior con la lectura de la actividad
como **campo de superficie** sobre una **malla compartida** del cuero cabelludo
(la interpolación `S_s` de los topomapas): el modelo también se entrena para
que el *heatmap* predicho sea fiel, no solo cada electrodo por separado. La
evaluación guarda `metrics_surface_test.csv` (`rmse_field`/`r_field`/`ve_field`
por ruta y configuración) y `runs/multi_heatmap/multiconfig_surface.png`.

Para abrirlos y ejecutarlos con el entorno del proyecto:

```bash
jupyter notebook notebooks/   # kernel Python 3 del entorno/ (ipykernel instalado)
```

Los notebooks se generan desde `notebooks/generate_notebooks.py` (nbformat);
para regenerarlos tras cambios en la lógica:

```bash
PYTHONPATH=src entorno/bin/python notebooks/generate_notebooks.py
```

Variantes disponibles en `config/` (vía `model.variant`): `default.yaml`
(`free`), `group.yaml`, `projected.yaml`, `soft_group.yaml`, y exploraciones
de latente `wide.yaml` (128) / `bottleneck.yaml` (32), las variantes de
**unificación de montajes** `montage_leadfield.yaml` / `montage_heatmap.yaml`,
y `multi_montage.yaml` (entrenamiento conjunto y balanceado sobre varias
configuraciones de electrodos a la vez) y `multi_heatmap.yaml` (igual + campo
de superficie/heatmap entrenado sobre la malla compartida). También
`multi_heatmap_v2.yaml` (como `multi_heatmap` + matriz de campo aprendible,
consistencia electrodo↔campo y entre configuraciones, adaptadores por
configuración, suavizado temporal e incertidumbre aprendida; ver
`docs/guia_conceptual.md`).
Resultados: `docs/results_comparison.md`, `docs/mapping_wip.md` y
`docs/guia_conceptual.md`.

* `build`: descarga los sujetos/corridas indicados, filtra (bandpass y notch), marca artefactos y canales malos por MAD z-score, revisa la referencia del dato original (`original_reference`) y construye las 4 referencias alineadas (unipolar, bipolar, CAR, REST), guardando `data/processed/dataset_{mode}_{subjects}.npz` y `*_leadfield.npz`. Splits por bloques temporales (`block`) o por sujeto (`subject`).
* `train`: escribe checkpoints en `runs/<nombre>/` (`best.weights.h5`, `model.keras`, `history.csv`).
* `eval`: imprime tabla de métricas por ruta (MSE/RMSE/MAE/r), compara las matrices efectivas con las analíticas y guarda figuras en `runs/<nombre>/figs/`.

## Estructura

```text
├── pyproject.toml               # dependencias (uv/pip instal ")
├── uv.lock
├── config/default.yaml          # configuración por defecto (variante free)
├── notebooks/                   # un notebook por variante (metodología + resultados)
│   ├── generate_notebooks.py    # genera/actualiza los notebooks (nbformat)
│   ├── free.ipynb               # autoencoder libre
│   ├── group.ipynb              # estructura de grupo exacta
│   ├── projected.ipynb          # anulación del modo constante
│   ├── soft_group.ipynb         # grupo suave por regularización
│   ├── wide.ipynb               # latente 128
│   ├── bottleneck.ipynb         # latente 32
│   ├── montage_leadfield.ipynb  # unificación por solución inversa (lead field)
│   ├── montage_heatmap.ipynb    # unificación por mapas de calor (spline)
│   ├── multi_montage.ipynb      # 19/64/128/256 electrodos balanceados
│   └── multi_heatmap.ipynb      # igual + campo de superficie (heatmap)
├── docs/
│   ├── model_variants.md        # descripción de las variantes de arquitectura
│   ├── mapping_wip.md           # unificación de montajes (WIP → resultados)
│   ├── guia_conceptual.md       # física, métricas y arquitectura (conceptos)
│   └── results_comparison.md    # tabla comparativa y análisis (sujetos 1–12)
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
│   ├── training/trainer.py      # bucles/callbacks TF
│   ├── evaluation/{metrics,plots}.py
│   ├── experiments/montage.py   # experimento de reconstrucción de montajes
│   ├── experiments/multi.py     # datos balanceados de multi-configuración
│   ├── nb.py                    # helpers compartidos para los notebooks
│   └── cli.py                   # comandos build/train/eval/pipeline/montage
└── tests/
    ├── test_physics.py          # propiedades de las referencias y REST
    ├── test_config_ds.py        # config y dataset
    ├── test_montage.py          # proyección/modo montaje
    ├── test_multi.py            # generador y modelo multi-configuración
    └── test_integration.py      # el modelo aprende los mapas sobre datos
```

## Notas

* El dataset por defecto son los sujetos 1–4 de `eegbci` (PhysioNet); el primer `build` descarga los EDFs (~decenas de MB por sujeto).
* El modelo se entrena con el inicializador `GlorotUniform`; inicializadores ortogonales degradan la convergencia en la factorización lineal de bajo rango.