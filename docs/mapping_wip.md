# Estado del proyecto — WIP unificación de montajes

**Fecha:** 2026-08-18 · **Rama:** `main` · Basado en `docs/results_comparison.md` (ya commiteado).

## Lo completado (commiteado)

* Dataset real de 12 sujetos eegbci (`data/processed/dataset_block_1_2_3_4_5_6_7_8_9_10_11_12.npz`, 228 269 muestras × 64 canales).
* Variantes `free`/`projected`/`soft_group` entrenadas y comparadas (ver `docs/results_comparison.md`).
* `group` y el encadenado analítico `T_d pinv(T_s)` fallan igual (~133 µV) — documentado.

## En progreso (NO commiteado): unificación de montajes (`mapping`)

Archivos nuevos **sin commit**:
* `src/eeg_transform/mapping.py` — proyección entre montajes.
* `src/eeg_transform/experiments/montage.py` — experimento de reconstrucción.
* `src/eeg_transform/experiments/__init__.py` — docstring vacío.
* `docs/mapping_wip.md` — este documento.
* Tests nuevos en `tests/test_physics.py` (mapping: +5, total 28 → **23 originales + 5**).

### Métodos implementados en `mapping.py`
* **spline** (interpolación esférica Perrin, estilo heatmap/topomapa): `spherical_spline_matrix(src, dst)` → `P (C_s, C_dst)`.
* **leadfield** (solución inversa vía lead field analítico): `leadfield_projection_matrix(G_src, G_dst, n_components=None)` → `P (C_s, C_dst)`.
* **nearest** (vecino más cercano): línea base geométrica mínima.
* Utilidades: `select_subset` (canales 10-20), `build_projection(method, ...)`, `round_trip_error`, `STANDARD_10_20_19`.

### Metodología del experimento (fiel)
* Ancla: REST del montaje completo (64 ch); se observan sus columnas en los electrodos fuente → grabación real de `Cs` canales.
* Sobre esa observación se computan **las 4 referencias del propio montaje fuente** (unipolar CAR bipolar REST, con sus operadores de `Cs` canales y lead field propio).
* Se proyecta al canónico con `obs_src @ P_back` y se compara contra la referencia canónica real (ground truth), centrando cada instante (modo constante irrecuperable).
* Montajes: 10-20 (19 ch, subconjunto exacto de los 64) y 10-10 (42 ch, `MONTAGE_10_10_39`).

### Hallazgo clave: regularización del leadfield
El problema inverso `W_s G_s` está extremadamente mal condicionado (cond ≈ 4e15):
* **En datos sintéticos sin ruido** (pocas fuentes observables) recupera ve ≈ 1.0 → la fórmula `P = W_s (W_s G_s)^{+T} G_dst^T` es correcta.
* **En datos reales** el SVD sin truncar amplifica ruido y `ve` cae a −0.6 (10-20) o −67 (10-10).
* **Truncar el SVD a `n_components ≈ C_s//3`** lo estabiliza: `ve` unipolar/CAR/rest sube a ~0.43 (10-20) y ~0.42 (10-10) — ver resultados.
* Implementado como predeterminado en `build_projection` → `leadfield_projection_matrix(..., n_components=None → C_s//3)`.

### Resultados (test, 12 sujetos; ve promedio del round-trip por referencia)
| método   | montaje | n_ch | rmse µV | mae µV | r      | ve     | peor ref  |
|----------|---------|------|---------|--------|--------|--------|-----------|
| nearest  | 10-20   | 19   | 23.11   | 12.42  | 0.586  | 0.099  | bipolar   |
| spline   | 10-20   | 19   | 23.00   | 12.02  | 0.606  | 0.104  | bipolar   |
| leadfield| 10-20   | 19   | 21.16   | 12.00  | 0.555  | **0.273** | bipolar |
| nearest  | 10-10   | 42   | 17.11   | 6.54   | 0.784  | **0.526** | bipolar |
| spline   | 10-10   | 42   | 23.11   | 9.57   | 0.745  | 0.118  | bipolar   |
| leadfield| 10-10   | 42   | 21.36   | 10.53  | 0.671  | 0.259  | bipolar   |

* Desglose `ve` por referencia (test):
  * leadfield 10-20: unipolar 0.431 · CAR 0.431 · rest 0.431 · **bipolar −0.200**
  * nearest   10-10: unipolar 0.624 · CAR 0.624 · rest 0.624 · **bipolar 0.231**
  * spline    10-20: unipolar 0.426 · CAR 0.426 · rest 0.426 · **bipolar −0.862**

* Lectura: unipolar/CAR/rest se recuperan bien por cualquier método (~0.4-0.6 ve); la referencia **bipolar** (~Laplaciano local) es lo que dispara el RMSE y arrastra el promedio. En montajes densos (10-10) el `nearest` domina sin necesidad de interpolación física; en 10-20 `leadfield` regularizado supera a `spline`.
* La ve promedio que reporta `reconstruct` mezcla 4 referencias (incluye bipolar) → números medios tímidos; mirar siempre el desglose.

### Pendiente (por hacer)
1. **CLI**: ✅ subcomando `montage` registrado en `cli.py` (`run_cmd` recibe el cfg). Comando: `eeg-transform -c config/default.yaml montage --montages 10-20,10-10`. Escribe `data/mapping_results.csv`.
2. **Config**: falta `mapping:` en `config.py` (n_components, método por defecto, montajes) y un `config/montage.yaml`.
3. **Integrar** la proyección en el pipeline de datos: `build_dataset` → proyectar montaje fuente → espacio canónico `C` antes de referencias (para 12 sujetos de entrenamiento).
4. **Entrenar end-to-end** para cada método (spline/leadfield/nearest) sobre datos proyectados y comparar con el modelo canónico (12 sujetos).
5. **Montajes a probar**: 10-20 (19), 10-10 (42) y un montaje distinto de 64 (p. ej. rotado/diferentes 64 → evaluar robustez al subconjunto).
6. **Resultados** en `docs/results_comparison.md` + interpretación (leadfield físico vs spline geométrico, coste de composición, qué referencia elegir).
7. **Runtime**: la evaluación en CPU tarda ~1-2 min por variante × método.

## Para reanudar
```bash
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli -c config/default.yaml montage --montages 10-20,10-10
```