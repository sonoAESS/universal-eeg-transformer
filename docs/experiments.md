# Experimentos: más datos y variantes con propiedades físicas

Métricas de **test** sobre sujetos 1–12 de `eegbci` (baseline `runs [1,2]`,
64 canales @160 Hz, split temporal por bloques). API `eeg_transform` v4
(montaje + variantes), pérdida Z-score por lote.

```bash
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli compare \
  runs/default runs/group runs/projected runs/soft_group
```

escribe en `runs/compare/figs/`: `multi_heatmap_rmse.png`,
`multi_heatmap_r.png`, `multi_consistency.png`, `multi_learning_val.png`.

## Datos

* 12 sujetos × 2 corridas (ojos abiertos/cerrados) = **228 269 muestras**
  (146 088 train / 36 524 val / 45 657 test).
* Preprocesado: `bandpass 1–45 Hz`, rechazo de artefactos z/MAD, canales
  malos marcados; referencia de adquisición: mastoides izquierda (los cuatro
  montajes son invariantes a ella).

## Variantes

| variante      | parámetros | física impuesta |
|---------------|------------|-----------------|
| `free`        | 8×C²       | ninguna         |
| `group`       | 4×C² + pinv fijo | grupo exacto (transitividad + identidad observable) |
| `projected`   | 8×C² centrados (`W = P W_raw`) | anulación del modo constante en toda ruta |
| `soft_group`  | 8×C² + penalización | composición `A_sd A_du ≈ A_su` suave (w=0.5) |

Detalle en `docs/model_variants.md`.

## Resultados (test, sujetos 1–12)

| variante | val_loss | r diag | r cross | rmse diag µV | rmse cross µV | comp medio | comp max |
|----------|----------|--------|---------|--------------|---------------|------------|----------|
| free       | 1.9e-4 | 1.0000 | 0.9997 | 0.686 | 0.659 | 0.0667 | 0.3629 |
| group      | 0.3125 | 0.7092 | 0.7091 | 133.4 | 133.5 | ~0      | ~0   |
| projected  | 1.5e-5 | 1.0000 | 0.99998| 0.262 | 0.282 | 0.0339  | 0.1901 |
| soft_group | 1.2e-3 | 1.0000 | 0.9990 | 0.348 | 0.917 | 0.0048  | 0.0400 |

`comp` = error de composición `||P(A_sd A_du − A_su)P||_F / ||P A_su P||_F`
(0 = grupo exacto; el baseline analítico `T_d pinv(T_s)` ≈ 1).

## Interpretación

* **Más datos ayudan.** Con 12 sujetos, `free` pasa de `r ≥ 0.98` (sujetos
  1–4) a `r cross = 0.9997` y `val_loss` 4× menor. El modelo es lineal y el
  exceso de señal mejora el condicionamiento de la regresión empírica.

* **`projected` domina a `free` en TODO.** La restricción "cada matriz anula
  el modo constante" no penaliza (misma familia de rutas) pero elimina el
  grado de libertad DC (no observable) que `free` desperdicia: `val_loss`
  12× menor, `rmse cross` 0.28 vs 0.66 µV y composición 2× más consistente
  (0.034 vs 0.067). La hipótesis: el subespacio restringido condiciona mejor
  la optimización y la generalización.

* **`group` es exacto pero su representación está "girada".** La factorización
  rígida `A_sd = W_s (W_d)^+` garantiza composición ~0 exacta, pero la
  familia representable es la **inversa** de la familia analítica
  `T_d pinv(T_s)`; con la inicialización ridge las rutas hacia/desde REST
  quedan mal invertidas (`r cross 0.71`, 133 µV). `restore_best_weights`
  mantiene el mejor estado (≈ init), que sigue siendo la solución cerrada
  para la estructura, no para la precisión por ruta. Es la *garantía* de
  invariante de composición — el coste es precisión.

* **`soft_group` es el compromiso práctico.** Arquitectura libre + penalización
  de composición (w=0.5): reduce el error de composición **14× vs free**
  (0.0048 vs 0.067) y 3× respecto a `projected`, con precisión cercana a
  `free` (r cross 0.9990, 0.92 µV). El peso w barre el frente
  precisión↔consistencia: `w→0` es `free`, `w→∞` tiende a `group`.

### Recomendación

* Uso general: **`projected`** (mejor precisión y consistencia que `free`;
  físicamente admisible por construcción).
* Aplicaciones que exigen invariante de composición exacta o razonamiento
  transitivo sobre referencias: **`soft_group`** con `w` afinado (0.2–1), o
  `group` si prima la garantía estructural sobre la precisión.

## Lectura (equivalencias en el modelo)

`free`/`projected`/`soft_group` representan la familia analítica `T_d pinv(T_s)`
y convergen a ella con la data; `group` representa su inversa y por eso la
comparación `error_fro_rel` con las matrices analíticas es menormente
significativa para `group` (0.11) — la métrica que lo distingue es `comp`:
exacta por construcción.