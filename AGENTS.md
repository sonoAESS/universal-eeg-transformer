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
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q     # tests (rápido, sin GPU)
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli --help
eeg-transform -c config/smoke.yaml pipeline        # build + train + eval
PYTHONPATH=src entorno/bin/python notebooks/generate_notebooks.py   # regenerar notebooks
```

* Los tests usan datos sintéticos pequeños; deben pasar antes de cualquier
  commit. No requieren descargar eegbci.
* La primera ejecución de `build` descarga PhysioNet eegbci y cachea en
  `data/processed/`; los checkpoints viven en `runs/<variante>/`.

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
  experiments/{montage,multi}.py  # observaciones por montaje y multi-config balanceada
  models/{universal_transformer,multi_montage,multi_heatmap}.py
  training/trainer.py          # bucles TF + build_multiconfig_model
  evaluation/{metrics,plots}.py
  nb.py                        # helpers compartidos de los notebooks (load_experiment,
                               #   train_variant, evaluate_*, plot_*)
tests/test_*.py                # physics, config_ds, montage, multi, integration
notebooks/generate_notebooks.py + *.ipynb   # notebooks autogenerados (no editar a mano)
runs/<variante>/               # checkpoints, history.csv, métricas, figuras
```

## Notebooks

Se generan desde `notebooks/generate_notebooks.py` (nbformat): editar ahí la
lógica/regenerar, nunca el `.ipynb` directamente. Cada notebook reutiliza el
checkpoint de `runs/` salvo `FORCE = True`.

## Rama actual: `explore/topomap-refs` (prueba conceptual, solo notebooks)

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
  Estructura: `topomap_refs/` (`01_topomapas_y_montajes.ipynb`,
  `02_referencia_average_all.ipynb`, `03_modelo_montaje_referencia.ipynb`,
  `README.md`) — se editan a mano, NO se regeneran desde
  `generate_notebooks.py` (igual que `multi_heatmap/`).
* Las referencias/montajes nuevos se definen DENTRO de los notebooks usando
  funciones existentes del paquete (`scalp_grid_matrix`, `linked_matrix`,
  `build_reference_matrix`, `inter_reference_matrix`, `spherical_spline_matrix`)
  respetando la convención matricial `X_ref = X @ M`.
* Caché/checkpoints: `data/processed/` (ya cacheado, incluye
  `mne_asa_montages.npz` con los grids ASA de MNE `standard_1005`/`standard_1020`)
  y `runs/topomap_refs/`. Para la grilla universal, los grids ASA se alinean al
  marco PhysioNet del canonical vía Procrustes por nombres compartidos (RMSE~0;
  las convenciones MNE y PhysioNet difieren solo por rotación/reflexión).
* Prefijo de commit de la rama: `feat(topomap_refs): ...`.

## Trabajo con ramas y commits

* Mensajes de commit en español, imperativos, con el prefijo de la variante
  cuando aplique (p. ej. `feat(multi_heatmap_v2): ...`, `docs: ...`).
* Exploración de una variante = rama propia; los artefactos grandes
  (`runs/`, `data/`) no se versionan (ver `.gitignore`).
