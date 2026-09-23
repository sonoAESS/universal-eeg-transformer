# spec-03 — Gate de calidad en `topomap-video`

**Estado:** en especificación (payoff rojo) · **Prefijo de commit:** `spec(03): ...`

## Contexto

`run_topomap_video` mezcla en el mismo vídeo/Png métodos con calidades
incompatibles (hoy `multi_heatmap_v2` con `VE < 0`). Un gate de calidad evita
publicar filas que falsean la comparación (p. ej. "método con VE −2.9" como si
fuera una predicción válida) y hace el tool honesto para creación de reportes.

## Requisitos funcionales

1. Dado `topomap-video --models ...`, se computa un reporte de calidad por
   método a partir de las métricas de ventana causal (`VE` medio y `r` medio
   sobre rutas), antes de serializar la figura.
2. Dado un umbral configurable `quality_gate` (VE mínimo por defecto **0.85**),
   un método que no lo alcanza queda **marcado en la figura** (sufijo
   ``" (no convergido)"`` en su etiqueta de fila) y se **excluye de las
   métricas agregadas** del reporte final; nunca aborta el run completo.
3. Dado `--quality-gate 0` (o `--no-gate`), se desactiva el filtrado.
4. Dado el reporte, se guarda `*_resumen_calidad.csv` con columnas
   `(method, ve_medio, r_medio, convergido)` junto a `*_metrics.csv`.

## Restricciones de diseño

- El gate es **visualización/reporting**: no altera predicciones ni métricas por
  ventana; solo decide la visibilidad de filas y la agregación.
- Nueva CLI flag `--quality-gate` (float, por defecto 0.85) y la salida
  `{"figuras": {...}, "metrics": ..., "calidad": Path}` de `run_topomap_video`.

## Criterios de aceptación (payoff)

| # | Criterio | Umbral | Test | Payoff |
|---|----------|--------|------|--------|
| 3.1 | `inspect.signature(run_topomap_video)` incluye el parámetro `quality_gate` con defecto 0.85. | — | `tests/test_spec_gate_calidad.py` | sí |
| 3.2 | Con un `MethodResult` sintético de `VE = 0.5`, el reporte lo declara `convergido=False` y su fila queda marcada. | 0.85 | igual | sí |
| 3.3 | Con `quality_gate=0`, ningún método se marca (comportamiento por defecto actual preservable). | 0 | igual | sí |
| 3.4 | El run real produce `*_resumen_calidad.csv` con los métodos etiquetados. | — | igual | sí |

## Notas de implementación

- Afixar el sufijo en `topomap_video_fig` vía `results[i].name` (ya mostrado en
  el `ylabel`): el gate solo reescribe el nombre de la fila y filtra en el
  `pd.concat` de métricas agregadas.
- API prevista: `apply_quality_gate(results, threshold) -> list[bool]`
  (``convergido`` por método, en orden) sobre objetos `MethodResult` con su
  `ve_medio`; testable con resultados falsos sin modelos ni entrenamiento.

## Histórico

- **2026-09-23**: spec abierta; payoff rojo (parámetro y reporte inexistentes).