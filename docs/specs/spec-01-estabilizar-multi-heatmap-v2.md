# spec-01 — Estabilizar `multi_heatmap_v2` (campo de superficie)

**Estado:** en especificación (payoff rojo) · **Prefijo de commit:** `spec(01): ...`

## Contexto

`multi_heatmap_v2` (config `config/topomap_video_multi_heatmap_v2.yaml`, 60
épocas, sujetos 1-2, 7 referencias) no converge: el historial
`runs/topomap_video/multi_heatmap_v2/history.csv` muestra `loss_estandarizada`
explotando hasta **1.2e9** y `surface_loss` hasta **1e10**, mientras
`mse_real_V2` permanece pequeño. La rama de campo de superficie diverge y
corrompe el entrenamiento; la evaluación ventaneada lo confirma con
`VE = -2.89` (canonical) y `VE = -3.22` (10-20), peor que la predicción trivial.

La superficie es además la vía central del objetivo general (grilla universal:
derivar cualquier distribución/referencia desde un campo interpolado), de modo
que estabilizarla es prerrequisito de `spec-06-grilla-densa`.

## Requisitos funcionales

1. Dado el entrenamiento de `topomap_video_multi_heatmap_v2.yaml` completo
   (sin `--force-train` sin necesidad, con el mismo presupuesto de épocas),
   el checklist de calidad por ruta del config **canonical** y del **10-20**
   supera los umbrales de la tabla del payoff.
2. Dado el historial `history.csv` de ese entrenamiento:
   - `val_loss_estandarizada` es decreciente en el tramo final y acotada
     (`max ≤ 5 × primer valor`; hoy crece hasta ~1.2e9).
   - `val_surface_loss` no diverge: `max ≤ 5000` (hoy ~1e10).
   - `val_xconfig_loss` y `val_field_consist_loss` permanecen acotadas
     (orden de magnitud ≤ 10× el valor inicial).
3. Dado el módulo `models/multi_heatmap.py`, la red tiene *gradient clipping*
   configurable y las contribuciones de campo (superficie/consistencia)
   estandarizables: si la divergencia reaparece, la pérdida no puede explotar a
   escala `surface_loss`, sino quedar acotada por diseño.
4. Dados los tests de regresión de los módulos tocados (`tests/test_topomap_video.py`,
   `test_multi.py`, `test_physics.py`), permanecen en verde.

## Restricciones de diseño

- No tocar la convención matricial ni la linealidad del núcleo; el cambio se
  restringe a la **función de pérdida** de la rama de superficie y a la
  estabilidad numérica del entrenamiento (clipping, normalización, pesos).
- Config extensible vía YAML (nuevos campos con valores por defecto que no
  rompan configs existentes).
- Sin cambios en el esquema de datos/datasets cacheados.

## Criterios de aceptación (payoff)

| # | Criterio | Umbral | Test | Payoff |
|---|----------|--------|------|--------|
| 1.1 | `val_loss_estandarizada` final < inicial y `max ≤ 5× inicial`. | estricto | `tests/test_spec_multi_heatmap_estabilidad.py` | sí |
| 1.2 | `val_surface_loss` acotada: `max ≤ 5000` (hoy 1e10). | estricto | igual | sí |
| 1.3 | `VE` medio por ruta (csv de vídeo, método `multi_heatmap_v2`) ≥ 0.90 para todas las referencias salvo `laplacian`. | 0.90 | igual | sí |
| 1.4 | `VE` por ruta de `laplacian` ≥ 0.70 (derivada espacial, escala ~200 mV). | 0.70 | igual | sí |
| 1.5 | Los umbrales 1.3–1.4 se cumplen en **canonical** y **10-20**. | doble | en el mismo test | sí |
| 1.6 | Regresión: suite `test_topomap_video.py` y `test_multi.py` en verde. | — | suite rápida | no |

## Notas de implementación

- Hipotetizar primero la causa raíz con un run de diagnóstico corto (p. ej.
  `surface_loss_weight` a 0 isotáblemente) antes de cambiar la función de
  pérdida; documentar el diagnóstico en la spec (sección histórica, con fecha).
- `learnable_interp: true`, `field_consistency_weight: 0.1` y
  `xconfig_consistency_weight: 0.1` son los sospechosos inmediatos; verificar
  también normalización de los campos objetivo (magnitudes descontroladas suelen
  venir de interpolaciones aprendidas sin estandarizar).

## Histórico

- **2026-09-23**: spec abierta; payoff rojo con los artefactos del run actual
  (`VE = -2.89/−3.22`, `surface_loss` ~1e10).