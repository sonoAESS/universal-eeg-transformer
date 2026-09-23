# spec-04 — Reporte de rutas normalizado (la `laplacian` no domina)

**Estado:** en especificación (payoff rojo) · **Prefijo de commit:** `spec(04): ...`

## Contexto

Las métricas de ventana `*_metrics.csv` mezclan referencias con escalas muy
distintas: `laplacian` (~200 mV, derivada espacial) descompensa RMSE/VE medios
y enterra las diferencias reales de las referencias usuales (unipolar, bipolar,
CAR, REST). La pérdida ya estandariza por referencia (`_zscore_loss`); el
reporte debe hacer lo mismo para que las decisiones sean per-ruta y no por
la referencia más amplia. Es prerrequisito de reportería de specs-01/02/03.

## Requisitos funcionales

1. Dado un run de `topomap-video`, junto a `*_metrics.csv` se generan:
   - `*_resumen_rutas.csv` con columnas
     `(method, ruta, rmse_uV, rmse_norm, r, ve, n_ventanas)`.
   - `*_resumen_metodos.csv` con agregados por método
     `(method, rmse_norm_mediana, ve_mediana)`.
2. `rmse_norm` normaliza cada ruta por su desviación típica en el segmento
   (análogo al z-score), de modo que las 7 referencias sean comparables.
3. Para cada método, el agregado de resumen usa la **mediana de `rmse_norm`**
   por ruta y la **mediana de `ve`** (VE puede ser negativo, así que no se usa
   media geométrica) — nunca la media aritmética del rmse absoluto.
4. Dado un método modelo, el reporte permite comparar per-ruta contra el
   analítico y contra `multi_montage` (referencia de calidad del proyecto).

## Restricciones de diseño

- No cambia las columnas ya publicadas de `*_metrics.csv` (compatibilidad con
  vídeos/reportes previos); solo añade el resumen normalizado.
- `rmse_norm` se calcula sobre el subespacio observable centrado (igual que
  `_route_stats`), no sobre los valores absolutos crudos.

## Criterios de aceptación (payoff)

| # | Criterio | Umbral | Test | Payoff |
|---|----------|--------|------|--------|
| 4.1 | `*_resumen_rutas.csv` y `*_resumen_metodos.csv` existen tras un run con las columnas de los requisitos 1–3. | — | `tests/test_spec_reporte_por_ruta.py` | sí |
| 4.2 | `rmse_norm` de `laplacian` es del mismo orden que el de `unipolar` (≤ 3×) en el analítico. | ≤ 3× | igual | sí |
| 4.3 | `*_resumen_metodos.csv` agrega con mediana de `rmse_norm` y mediana de `ve` (no la media de `rmse_uV`). | — | igual | sí |

## Notas de implementación

- Reutilizar `_route_stats` (ya centra temporalmente); añadir un campo
  `std_ruta` al agrupado y dividir `rmse` por él.
- Mantener el `*_metrics.csv` intacto para no romper los PNG/vídeos anteriores.

## Histórico

- **2026-09-23**: spec abierta; payoff rojo (el resumen no existe hoy).