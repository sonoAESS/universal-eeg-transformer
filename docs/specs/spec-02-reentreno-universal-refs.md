# spec-02 — Reentrenar `universal_refs_{conv,gru}` con el código actual

**Estado:** en especificación (payoff rojo) · **Prefijo de commit:** `spec(02): ...`

## Contexto

La variante temporal `universal_refs` (cabeza de residuo sobre el núcleo lineal
instantáneo) aparece en el vídeo-topomapa con **r = 0.95**, peor que
`multi_montage` (r = 0.98). No es comparable: el checkpoint usado
(`runs/universal_refs_cmp_conv/best.weights.h5`) se entrenó con una versión
anterior de variables (`_v2_params`) y en el vídeo se carga con
`skip_mismatch=True`. Los YAMLs dedicados (`config/topomap_video_universal_refs_{conv,gru}.yaml`)
no tienen pesos válidos (`runs/topomap_video/universal_refs_conv` solo tiene un
`history.csv` de 0 bytes). El beneficio temporal honesto se mide con la
evaluación **ventaneada** (`evaluate_multiconfig_windowed` /
`predict_multiconfig_windowed` → `metrics_windowed_test.csv`).

## Requisitos funcionales

1. Dado el entrenamiento de `config/topomap_video_universal_refs_{conv,gru}.yaml`
   (sujetos 1-2, canonical, 7 referencias, `temporal_cell` conv/gru), cada run
   deposita en `runs/topomap_video/universal_refs_{cell}/`:
   `best.weights.h5` (> 0 bytes), `history.csv` completo y
   `metrics_windowed_test.csv` (columnas `(origen, destino, rmse_uV, r, ve,
   cell)` por ventana causal).
2. Dado el checkpoint nuevo, se carga **sin** `skip_mismatch` (sin warnings de
   variables incompatibles) desde el código actual.
3. Dado `metrics_windowed_test.csv`, el `r` medio (sobre rutas y ventanas) es
   **≥ 0.97** y el `VE` medio **≥ 0.92** — mejora sobre el estado actual 0.95.
4. Dado el mismo `csv`, conv supera o iguala al analítico y a `multi_montage`
   en `VE` medio (la cabeza de residuo temporal debe justificar su existencia).
5. Dado `temporal_cell: gru`, el payoff usa los mismos umbrales que `conv`
   (misma escala de presupuesto: mismas épocas, lr, ventana 32).

## Restricciones de diseño

- El núcleo lineal instantáneo no cambia; solo se (re)entrena la cabeza con el
  código corriente. Sin `skip_mismatch` como vía de carga en producción.
- La evaluación estándar (tensores 2-D) sigue desactivando la cabeza; el payoff
  es **ventaneado** por construcción.
- Las cabezas se inicializan a cero: el arranque debe ser idéntico al modelo
  lineal (ablation trivial).

## Criterios de aceptación (payoff)

| # | Criterio | Umbral | Test | Payoff |
|---|----------|--------|------|--------|
| 2.1 | El checkpoint de `universal_refs_conv` carga estrictamente (sin `skip_mismatch`) desde `runs/topomap_video/universal_refs_conv/best.weights.h5`. | — | `tests/test_spec_universal_refs_reentreno.py` | sí |
| 2.2 | `metrics_windowed_test.csv` existe y `r` medio ≥ 0.97, `VE` medio ≥ 0.92. | r/VE | igual | sí |
| 2.3 | `VE` medio ≥ `VE(multi_montage)` − 0.01 (justifica la cabeza). | rel. | igual | sí |
| 2.4 | Idem 2.2 para `universal_refs_gru`. | r/VE | igual | sí |
| 2.5 | La suite `test_temporal.py` y `test_topomap_video.py` queda verde. | — | suite rápida | no |

## Notas de implementación

- Reutilizar la infraestructura `evaluate_multiconfig_windowed` del CLI; generar
  `metrics_windowed_test.csv` con el mismo código que el barrido `cmp_*` pero
  con los configs dedicados del vídeo.
- No reentrenar manualmente alterando configs: el payoff exige exactamente los
  YAMLs `topomap_video_universal_refs_*` con sus `run_dir` canónicos.

## Histórico

- **2026-09-23**: spec abierta; payoff rojo (no hay `best.weights.h5` ni
  `metrics_windowed_test.csv` en los `run_dir` del vídeo).