# Resultados comparativos: variantes del transformador

Fecha: 2026-08-17 · Dataset `eegbci` sujetos **1–12**, corridas 1–2 (228 269
muestras × 64 canales), split temporal por bloques (test 20 %, val 20 %),
pérdida Z-score, 60 épocas, batch 1024, `latent_dim = C = 64` (salvo
`wide`/`bottleneck`).

Reproducción: `eeg-transform -c config/<variante>.yaml pipeline` y luego
`eeg-transform compare runs/*`.

## Tabla comparativa (test)

Métricas promedio de test. `rmse` en microvoltios; `r` = correlación de
Pearson media entre canales; `comp_medio`/`comp_max` = error relativo de
composición `||P(A_{s→d}A_{d→u} − A_{s→u})P||F / ||P A_{s→u} P||F`
(diagonal 0 = grupo exacto); `error_fro` = desviación Frobenius relativa de
las matrices efectivas frente a las analíticas.

| Variante      | latent | RMSE diag (µV) | RMSE cross (µV) | r cross | r peor ruta | error_fro med | comp_medio | comp_max |
|---------------|-------:|---------------:|----------------:|--------:|------------:|--------------:|-----------:|----------:|
| **projected** | 64     | **0.26**       | **0.28**        | **1.0000** | 0.9999 | 0.127 | 0.034 | 0.190 |
| free (default)| 64     | 0.69           | 0.66            | 0.9997 | 0.9977 | 0.143 | 0.067 | 0.363 |
| soft_group    | 64     | 0.35           | 0.92            | 0.9990 | 0.9926 | 0.154 | **0.005** | **0.040** |
| wide          | 128    | 2.14           | 1.06            | 0.9960 | 0.9765 | 0.191 | 0.143 | 0.552 |
| group         | 64     | 133.4          | 133.5           | 0.7091 | 0.0414 | 0.114 | 0.000 | 0.000 |
| bottleneck    | 32     | 13.6           | 8.10            | 0.9031 | 0.7387 | 0.757 | 0.127 | 0.534 |
| analítico `T_d pinv(T_s)` | — | 133.4 | 134.2 | 0.7348 | 0.0414 | — | 0.593 | 8.563 |

Ruta más débil en todos los modelos aprendidos: `rest → bipolar`
(`r ≈ 0.99` salvo en `group`/`bottleneck`). REST es el montaje dominante por
amplitud (referencia al infinito, ~1–2 µV RMSE) y concentra el error residual.

## Interpretación

### 1. `projected` gana en precisión, `soft_group` en física

- **`projected`** es el mejor compromiso global: RMSE cruzado ~0.28 µV con
  `r_cross = 1.0000` y además anula por construcción el modo constante
  (todas las rutas son referencias válidas). Su error de composición
  (0.034/0.190) es la mitad que el de `free`.
- **`soft_group`** (libre + penalización de composición `w=0.5`) logra
  `comp_medio = 0.005`, ~14× menor que `free` y **≈120× menor que el
  encadenado analítico**, con un coste en RMSE cruzado de solo ~0.9 µV.
  Si la transitividad `s→d→u` es un requisito (p. ej. cascadas de
  re-referencia), es la opción físicamente más fiel sin sacrificar la
  reconstrucción.
- **`free`** queda en medio: buena precisión, composición aceptable.

### 2. `group` y el encadenado analítico fallan por el mismo motivo

La variante `group` (decodificador = pseudo-inversa del encoder, grupo
exacto) y la línea base `T_d pinv(T_s)` tienen RMSE idéntico (~133 µV) y la
misma ruta más débil (`r = 0.04`). La estructura rígida asume que todas las
referencias comparten **el mismo subespacio observable** y que el mapa entre
ellas es invertible en él. No lo es para REST, cuyo rowspace (lead field a
infinito) no coincide con el de uni/bip/CAR; forzar `A_{s→d} = W_s (W_d)⁺`
impone una equivalencia espectral que el dato real no satisface. La
consistencia de composición exacta (0.000) es real, pero es un grupo sobre un
espacio equivocado.

### 3. `error_fro` alto en rutas REST no implica mala predicción

En **todas** las variantes el `error_fro_rel` es ~1 en rutas con REST
(media 0.13–0.19, máximo ~0.99), aunque `projected` predice esas rutas con
`r ≈ 1.0000`. El modelo aprende una matriz **datos-equivalente** a la REST
analítica pero distinta en norma Frobenius: la REST del lead field no es
identificable únicamente desde el dato y hay muchas matrices que implementan
el mismo mapeo. La predicción manda; la comparación matricial es solo
informativa.

### 4. El latente óptimo es `C`

- `wide` (latente 128 > C): empeora (1.06 µV) — parámetros extra sin
  ganancia física y peor composición.
- `bottleneck` (latente 32 < C): colapsa (8.1 µV, `r = 0.90`) — cada
  referencia necesita su propio subespacio de señal; comprimirlo destruye
  información.
- Ambos parten además sin la inicialización empírica (`init_from_data` exige
  `latent_dim == C`), por lo que parte de la caída se debe a la ausencia del
  pre-entrenamiento lineal. Con `latent_dim = C` el modelo reproduce los
  mapas casi exactos desde la época 0.

### 5. Más datos (sujetos 1–12 vs 1–4) mejoraron la generalización

Frente a la tabla del README previo (sujetos 1–4): `free` baja RMSE cruzado
de ~0.8 µV a 0.66 µV y sube `r_cross` de 0.9987 a 0.9997; `rest→bipolar`
pasa de `r ≈ 0.99` a 0.9977. Más sujetos refinan las matrices efectivas sin
cambiar la conclusión física.

## Recomendación

| Si priorizas…                              | elige         |
|--------------------------------------------|---------------|
| Precisión pura (menor RMSE, mayor r)       | `projected`   |
| Transitividad exacta + buena precisión     | `soft_group`  |
| Fisicalidad estructural (grupo exacto)     | `group` (solo conceptual) |
| Referencia rápida sin entrenar             | analítico `T_d pinv(T_s)` (no transitivo) |

## Unificación de montajes (variantes `montage_*`)

**Fecha:** 2026-08-18 · Mismo dataset (12 sujetos) con el montaje **10-20
(19 canales)** como configuración de electrodos de origen → canónico de 64.
El transformador universal estima las 4 referencias canónicas desde
observaciones del montaje fuente proyectadas con una matriz fija `P`
(`mapping.build_projection`). Las métricas comparan la estimación (modelo)
contra la proyección analítica pura `obs @ P` (`*_proy`).

| Variante | método P | rmse cross (µV) | r cross | ve cross | ve_proy | mejor que P |
|----------|----------|----------------:|--------:|---------:|--------:|:-----------:|
| **montage_heatmap** | spline (Perrin) | **16.26** | **0.746** | **0.577** | 0.051 | +0.53 |
| montage_leadfield | solución inversa | 18.70 | 0.623 | 0.442 | 0.181 | +0.26 |

* **El modelo supera siempre a la proyección pura.** La ganancia es la
  contribución del aprendizaje: la proyección analítica fija solo fija la
  geometría; el autoencoder refina el resto del campo y la conversión entre
  referencias.
* `montage_heatmap` es mejor a pesar de que su proyección pura es la peor
  (spline ve 0.05 vs leadfield 0.18): las **características suaves** del
  spline son más fáciles de refinar que los modos de la solución inversa mal
  condicionada (truncada a `C_s//3`).
* La referencia **bipolar** sigue siendo la ruta más débil en ambas variantes
  (es un operador local cuyo montaje fuente observa con otra cadena).
* Interpretación: el RMSE de montaje **no es comparable 1:1** con el de las
  variantes canónicas (entrada de 19 canales proyectados vs 64 canales). El
  techo físico de estimar 64 desde 19 electrodos es la `ve_proy`; el modelo lo
  sube a ~0.45-0.6.

Detalle completo, metodología fiel y desglose por referencia en
`docs/mapping_wip.md`; visualización con mapas de calor del cuero cabelludo en
`notebooks/montage_leadfield.ipynb` y `notebooks/montage_heatmap.ipynb`.

## Múltiples configuraciones de electrodos (multi_montage / multi_heatmap)

**Fecha:** 2026-08-20 · Un solo modelo comparte un core canónico (64 ch) y
proyecciones fijas `P_s`/`Q_s` por configuración: **19 (10-20), 64
(canonical), 128 y 256 (densos mapeados por spline sobre 10-20)**, con las 4
referencias intra-configuración. `multi_heatmap` añade el término
`surface_loss` sobre la malla compartida. Mismo dataset (12 sujetos).
Detalle en `docs/experiments.md`.

| Configuración | rmse diag (µV) | rmse cross (µV) | r diag | r cross | peor ruta (cross) |
|---------------|---------------:|----------------:|-------:|--------:|------------------:|
| **10-20 (19ch)** | 4.6 | 7.8 | 0.977 | 0.938 | rest→bipolar 0.85 |
| canonical (64ch) | 11.7 | 14.5 | 0.853 | 0.807 | bipolar→rest 0.56 |
| dense-128 | 6.9 | 8.5 | 0.786 | 0.587 | bipolar→rest 0.07 |
| dense-256 | 6.9 | 7.6 | 0.727 | 0.593 | bipolar→rest 0.18 |

Campo de superficie (`multi_heatmap`): 10-20 `r 0.97/0.93` (diag/cross),
canonical `0.86/0.81`, dense `0.51–0.64/0.26–0.34`.

### Interpretación (multi vs variantes canónicas)

* **No es comparable 1:1 con las variantes de referencia** (mismo techo
  físico que en montaje: las rutas parten de observaciones 19→256 ch, no de 64
  nativas). La mejor config (10-20) logra `r_cross 0.94`; el canonical
  compartido sufre multitarea (uni→uni 0.99 pero REST baja a 0.56).
* **El `surface_loss` ancla el patrón espacial**: campo y rutas correlacionan
  (10-20: 0.93 vs 0.94); no es decorativo.
* **En densas el REST es degenerado por el spline**: la línea base analítica
  `T_d·pinv(T_s)` también falla (r 0.17–0.35, ve −48) porque los canales
  extra son interpolaciones de 19 y el promedio REST del montaje denso no
  representa el de 64 ch. Exploración de alternativas (RESTRIDGE/lead field)
  en curso.

## Variantes temporales / recurrentes (universal_refs)

Las variantes con cabeza temporal de residuo (`universal_refs`,
`temporal_cell: conv` o recurrente `gru`/`lstm`/`rnn`) **no comparten la
tabla anterior**: la evaluación estándar por rutas alimenta tensores 2-D
instantáneos que desactivan la cabeza. El beneficio temporal se mide con la
**evaluación ventaneada** (`evaluate_multiconfig_windowed`): ventanas causales
deslizantes y salida del último paso, reportada aparte en
`metrics_windowed_test.csv`. Añadir aquí los resultados (RMSE/r/VE por ruta y
config, con su línea base analítica) al disponer de una ejecución de
`config/universal_refs_gru.yaml`.
