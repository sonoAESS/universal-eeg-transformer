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
