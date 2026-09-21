# `topomap_refs` — enfoque topomapa / grilla universal

Exploración de la rama `explore/topomap-refs`: **solo notebooks** (manejados a
mano, NO se regeneran desde `generate_notebooks.py`). Usa datos reales cacheados
(eegbci 64ch canónico, `standard_1005`, sujetos 1-2, 7 referencias) sin
descargas y sin Drive.

Idea central (del concepto `/home/aess/Proyectos/eeg_to_eccog_dl`): toda señal
se representa como **campo de superficie interpolado**; desde ahí se deriva
cualquier distribución y se cambia la referencia de forma conjunta con el
montaje.

## Notebooks

* `01_topomapas_y_montajes.ipynb` — validación de las 7 referencias con
  métricas correctas (VE por familia de unidades; el laplaciano está en V/m²)
  y **grilla universal de 4 montajes**: `10-10` (42), `10-20` (19), `asa10-05`
  (343), `asa10-20` (94). Los grids ASA de MNE se **alinean al marco PhysioNet
  por Procrustes** (RMSE ~ 0; solo rotación/reflexión). Nodos coincidentes =
  medidos (self-fit), resto = interpolados (coherencia entre raíces).
* `02_referencia_average_all.ipynb` — nueva referencia `M = I − w·1ᵀ` con
  `w =` media sobre píxeles válidos del spline esférico (Σw=1). Orden físico
  medido: `car < avg_all < rest`.
* `03_modelo_montaje_referencia.ipynb` — un **único peso lineal** aprende
  `10-10 bipolar → 10-20 monopolar (average_all)`. LS cerrada recupera la ruta
  exacta (VE=1.0); Adam + whitening PCA alcanza VE≈0.99 y supera la línea base
  analítica (0.97, techo del spline). Del bipolar el nivel *monopolar Cz* es
  inobservable (gauge): `average_all` resuelve la indeterminación.

## Resultados clave

| ruta | test VE | RMSE (µV) |
|---|---|---|
| baseline analítica (bipolar→pinv→spline→avg) | 0.9704 | 3.56 |
| LS cerrada (SVD) | **1.0000** | 0.00 |
| modelo Adam (whitening PCA, k=39) | **0.9915** | 1.91 |

Grilla universal 01: self-fit (nodos medidos) mediana 0.977; round-trips vía
`asa10-05`/`asa10-20` ≥ 0.957; coherencia de raíces en interpolados 0.71
(`asa10-05`) y 0.27 (`asa10-20`).

## Notas de ejecución

* Ejecutar con el kernel **`python3`** (entorno del repo, mne disponible) y
  `PYTHONPATH=src`. El kernel `entorno` apunta al venv de `eeg_to_eccog_dl`
  (no tiene `mne`).
* Caché: `data/processed/mne_asa_montages.npz` (grids ASA de MNE); métricas y
  figuras en `runs/topomap_refs/`.
* Hallazgo en `src/` (fuera de esta rama): `inter_reference_matrix` compone
  `T_d @ pinv(T_s)`; con la convención `X @ M` lo correcto es
  `pinv(T_s) @ T_d`.