# spec-00 — Proceso de desarrollo guiado por especificación (SDD)

**Estado:** en vigor · **Prefijo de commit:** `spec(00): ...` o según el slug de la spec.

## Contexto

El proyecto (objetivo general en `README.md` y `docs/guia_conceptual.md`) persigue
un autoencoder **lineal** all-to-all que unifica conversiones entre referencias y
montajes. El estado actual revela métodos con payoffs medibles opuestos (p. ej.
`multi_heatmap_v2` sin converger con `VE < 0`, mientras `free`/`multi_montage`
alcanzan `r ≈ 1.0`). Los criterios de éxito deben definirse **antes** de
implementar: la especificación es la fuente de verdad y el historial de sus
cambios vive en git (nunca el código primero).

## Requisitos funcionales

1. Dado un trabajo de modelo/entrenamiento/evaluación que arranca, existe una
   spec previa en `docs/specs/spec-XX-<slug>.md` con las secciones obligatorias.
2. Dada una spec, existe al menos un archivo `tests/test_spec_XX_*.py` que
   codifica sus **criterios de aceptación** como tests `pytest`.
3. Dado un criterio de aceptación que precisa artefactos de entrenamiento o
   de evaluación (never descargas/GPU), se marca con `@pytest.mark.payoff` y se
   excluye por defecto de la suite rápida (`addopts = -m 'not payoff'`).
4. Dada una implementación que contradice la spec (payoff no alcanzable), se
   reescribe la spec primero; el commit de la reescritura documenta el porqué.
5. Dado un trabajo terminado, la verificación es: `pytest tests -q` (rápido) y
   `pytest tests/test_spec_XX_* -m payoff` (aceptación) + evidencia numérica de
   CSV/history anexada o referenciada en la spec.

## Restricciones de diseño

- Convención matricial sagrada `X_ref = X @ M`; las matrices actúan **por filas**
  (`P_s`, `Q_s`, `S_s`). Nunca transponer sin verificar esta regla.
- Linealidad del núcleo: cada ruta efectiva es `A_{s→d} = W^enc_s W^dec_d`;
  las survaciones de esta restricción deben especificarse explícitamente.
- Idioma español en docstrings/specs; identificadores en inglés fijados por el
  dominio. Logging vía `logging_conf.get_logger`.
- Los tests de spec usan datos sintéticos pequeños o caché de `data/processed/`;
  nunca descargan eegbci desde los tests de la suite rápida.

## Criterios de aceptación

| # | Criterio | Test | Payoff |
|---|----------|------|--------|
| 0.1 | Toda spec en `docs/specs/` tiene `## Contexto`, `## Requisitos`, `## Criterios de aceptación` y un test asociado. | `tests/test_spec_sdd_proceso.py` | no |
| 0.2 | Todo spec que defina payoffs numéricos tiene al menos un test `@pytest.mark.payoff` con el mismo umbral. | `tests/test_spec_sdd_proceso.py` | no |
| 0.3 | La suite rápida ignora los tests `payoff`; la de aceptación los ejecuta explícitamente. | config de pytest + `-m payoff` | no |

## Notas de implementación

- Registrar el marcador `payoff` en `[tool.pytest.ini_options]` y excluirlo por
  defecto: `addopts = "-ra -q -m 'not payoff'"`.
- Las specs semilla: `spec-01` (estabilizar `multi_heatmap_v2`), `spec-02`
  (reentrenar `universal_refs_{conv,gru}`), `spec-03` (gate de calidad en
  `topomap-video`), `spec-04` (reporte de rutas normalizado). Fase 2 de
  investigación: bandas de frecuencia, grilla densa 128/256, capacidad latente.