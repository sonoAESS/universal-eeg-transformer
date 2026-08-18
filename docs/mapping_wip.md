# Estado del proyecto — WIP unificación de montajes

**Fecha:** 2026-08-18 · **Rama:** `main`. Basado en `docs/results_comparison.md`.

## Lo completado (commiteado)

* Dataset real de 12 sujetos eegbci (228 269 muestras × 64 canales) y las
  variantes canónicas `free`/`group`/`projected`/`soft_group`/`wide`/`bottleneck`.
* **Proyección entre montajes** (`src/eeg_transform/mapping.py`), experimento
  de reconstrucción (`experiments/montage.py`) y subcomando `montage` del CLI.
* **Variantes entrenables de unificación de montajes**:
  `montage_leadfield` y `montage_heatmap` (ver más abajo).

## Métodos de proyección en `mapping.py`

* **spline** (interpolación esférica Perrin, estilo heatmap/topomapa):
  `spherical_spline_matrix(src, dst)` → `P (C_s, C_dst)`. Suavizado sensible
  a la **densidad** del montaje (`density_smoothness`): a más dispersos los
  electrodos, mayor regularización.
* **leadfield** (solución inversa vía lead field analítico):
  `leadfield_projection_matrix(G_src, G_dst, n_components=None)` → `P`.
  Truncado SVD a `C_s//3` (default) porque `W_s G_s` está mal condicionado
  (cond ≈ 4e15): sin regularizar los datos reales divergen (ve ≈ −0.6 y −67).
* **nearest**: línea base geométrica mínima. `select_subset`, `build_projection`,
  `scalp_grid_matrix` (malla 2D del cuero cabelludo para los mapas de calor).
* Montajes conocidos: `10-20` (19 ch, subconjunto exacto de los 64) y `10-10`
  (39 ch). `experiments/montage.resolve_montage` acepta también listas de
  canales → **cualquier configuración de electrodos**.

## Variantes entrenables de unificación de montajes

El transformador universal consume las observaciones de un montaje de `C_s`
electrodos y estima las **4 referencias canónicas** (64 ch):

```
x ∈ R^{C_s}  →(P fija C_s×C)→  x·P ∈ R^C  →  autoencoder  →  ŷ ∈ R^C
ruta efectiva s→d: A_{s→d} = P · W_enc_s · W_dec_d   (C_s×C)
```

* `montage_leadfield` (`config/montage_leadfield.yaml`): proyección por
  **solución inversa** con el lead field (SVD truncado automático `C_s//3`).
* `montage_heatmap` (`config/montage_heatmap.yaml`): proyección **spline**
  (heatmap), suavizado adaptativo por densidad (`smoothness·C_c/C_s`); la
  actividad se representa como manchas sobre el cuero cabelludo
  (`mapping.scalp_grid_matrix` + `plots.scalp_heatmap_fig`).

Entrenamiento: mismo bucle que las canónicas (Z-score, early stopping, Adam)
pero sobre observaciones de montaje; `init_from_data` ancla el latente al
unipolar canónico de las observaciones **proyectadas**.

### Resultados (test, 12 sujetos, montaje 10-20 → canónico 64)

Métricas de `metrics.evaluate_montage_routes` (subespacio observable). Las
columnas `*_proy` son la proyección analítica pura `obs @ P` (sin aprendizaje).

| Variante | rmse diag (µV) | r diag | ve diag | rmse cross (µV) | r cross | ve cross | ve_proy (cross) |
|----------|---------------:|-------:|--------:|----------------:|--------:|---------:|----------------:|
| montage_heatmap | 16.27 | 0.746 | 0.576 | 16.26 | **0.746** | **0.577** | 0.051 (spline) |
| montage_leadfield | 18.55 | 0.629 | 0.450 | 18.70 | 0.623 | 0.442 | 0.181 (leadfield) |

* El **modelo supera siempre a la proyección pura** (ve +0.26 a +0.53).
* `montage_heatmap` es la mejor ruta a pesar de partir de la peor proyección
  pura (ve 0.05): las características **suaves** del spline son más fáciles de
  refinar que los modos de la solución inversa mal condicionada.
* La ruta **bipolar** sigue siendo la más débil en ambos (esperado: es un
  operador local; el montaje fuente la observa con su propia cadena).
* Nótese que el RMSE no es comparable 1:1 con las variantes canónicas: aquí la
  entrada son 19 canales proyectados, no las referencias completas de 64.

Limitación física esperada: estimar 64 canales desde 19 electrodos tiene un
techo (`ve_proy` ≈ 0.05-0.27 según método); el modelo lo empuja hasta ~0.45-0.6.

## Notas del experimento de reconstrucción (round-trip, sin aprendizaje)

| método   | montaje | ve     | mejor ref | peor ref |
|----------|---------|--------|-----------|----------|
| leadfield| 10-20   | **0.273** | unipolar/CAR/rest (0.43) | bipolar (−0.20) |
| spline   | 10-20   | 0.104  | unipolar (0.43) | bipolar (−0.86) |
| nearest  | 10-10   | **0.526** | unipolar (0.62) | bipolar (0.23) |
| leadfield| 10-10   | 0.259  | unipolar (0.42) | bipolar |

Lectura: unipolar/CAR/rest se recuperan bien por cualquier método; bipolar
arrastra el promedio. En montajes densos el `nearest` basta; en 10-20 el
`leadfield` regularizado supera al `spline`. Ver `data/mapping_results.csv`.

## CLI y reproducción

```bash
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q          # 28 tests
# Comparación de proyecciones (round-trip):
eeg-transform -c config/default.yaml montage --montages 10-20,10-10
# Entrenar una variante de montaje (entorno/bin/python + PYTHONPATH=src):
entorno/bin/python /tmp/train_montage.py config/montage_leadfield.yaml
entorno/bin/python /tmp/train_montage.py config/montage_heatmap.yaml
# Evaluar (escribe metrics_test.csv + figs/ incluido scalp_heatmap.png):
eeg-transform -c config/montage_leadfield.yaml eval
# Notebooks autocontenidos (0 errores, 3 figuras):
notebooks/montage_leadfield.ipynb · notebooks/montage_heatmap.ipynb
```

## UX: notebooks de montaje

Los notebooks `montage_leadfield`/`montage_heatmap` muestran la cadena
completa (metodología, arquitectura, entrenamiento con checkpoint reutilizado,
evaluación vs proyección analítica y **mapas de calor del cuero cabelludo**):
de la observación del montaje fuente (electrodos → manchas) a la proyección
analítica, el modelo y la verdad canónica.