#!/usr/bin/env python3
"""Genera los notebooks de exploración de variantes del Universal EEG Transformer.

Dos familias de notebooks autocontenidos en ``notebooks/``:

* **Montaje** (montage_leadfield, montage_heatmap): unificación de montajes.
  El transformador universal estima las 4 referencias canónicas desde las
  observaciones de un montaje de electrodos (p. ej. 10-20, 19 canales)
  proyectadas al espacio canónico con una matriz fija ``P``:

  - ``montage_leadfield``: solución inversa con el lead field analítico
    (SVD truncado a ``C_s//3`` porque el problema está mal condicionado).
  - ``montage_heatmap``: interpolación esférica (Perrin) — la actividad se
    ve como manchas sobre el cuero cabelludo, suavizadas según la densidad
    del montaje.

* **Multi-configuración** (multi_montage, multi_heatmap, multi_heatmap_v2,
  universal_refs): un solo modelo que acepta 19/64/128/256 electrodos y
  predice las referencias intra-configuración. ``universal_refs`` añade la
  cabeza temporal de residuo (``temporal_cell``: conv/gru/lstm/rnn) sobre el
  núcleo lineal instantáneo; su evaluación es **ventaneada** (causal).

Estructura común: introducción (markdown) → configuración → arquitectura →
entrenamiento (reutiliza el checkpoint) → evaluación → figuras inline.

Uso:
    PYTHONPATH=src entorno/bin/python notebooks/generate_notebooks.py
    # requiere: jupyter nbformat (ya en pyproject)
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Contenido por variante (metodología + arquitectura)
# ---------------------------------------------------------------------------
# El texto proviene de docs/model_variants.md y docs/results_comparison.md
# (misma terminología y física).

_MONTAGE_BASE_DESCRIPTION = r"""
El **Universal EEG Transformer** en modo **montaje** unifica grabaciones con
**cualquier configuración de electrodos** hacia un espacio canónico y,
desde ahí, hacia las cuatro referencias de EEG:
`unipolar`, `bipolar`, `CAR` y `REST` (todas de `C = 64` canales).

### Idea: proyectar y refinar

1. Un montaje de `C_s` electrodos (p. ej. 10-20, `C_s = 19`) observa la
   actividad del cuero cabelludo solo en sus posiciones. La grabación real de
   esos electrodos se **simula** de forma fiel: se parte de la referencia al
   infinito (REST) del montaje completo y se observan sus columnas en los
   electrodos del montaje fuente; sobre esa observación se computan las 4
   referencias **del propio montaje** (sus operadores, con su lead field).

2. Una matriz de proyección **fija** `P (C_s × C)` lleva la observación al
   espacio canónico: `x ∈ R^{C_s} ↦ x·P ∈ R^C`.

3. El **autoencoder lineal universal** aprende a refinar la proyección y a
   estimar las 4 referencias canónicas desde `x·P`: la conversión
   `s→d` queda como la matriz `A_{s→d} = P·W^{enc}_s·W^{dec}_d` (C_s×C).

### Los dos mecanismos de proyección (`mapping.method`)

| Método | Variante | Mecanismo | Suavizado |
|---|---|---|---|
| `leadfield` | `montage_leadfield` | **Solución inversa**: estima los potenciales canónicos resolviendo `P = W_s (W_s G_s)^{+T} G_c^T` con el lead field analítico multicapa | SVD truncado a `C_s//3` (el problema está mal condicionado, cond ~1e15) |
| `spline` | `montage_heatmap` | **Heatmap/topomapa**: interpolación esférica (Perrin) de la actividad entre los electrodos, que se percibe como **manchas** sobre el cuero cabelludo | Ridge que crece con la `densidad`: a montajes más dispersos, más suavizado |

### Calidad de la proyección pura (sin aprendizaje)

Reconstrucción *round-trip* al canónico con **solo** la proyección
(`data/mapping_results.csv`, test, 12 sujetos):

| Montaje | leadfield (ve) | spline (ve) | nearest (ve) |
|---|---|---|---|
| 10-20 (19) | **0.273** | 0.104 | 0.099 |
| 10-10 (39) | **0.259** | 0.118 | 0.526 |

El modelo universal parte de esta línea base y aprende a superarla.

### Identidad de la variante

**`@VARIANT@`** — @VARIANT_LINE@ (config `@CONFIG_REL@`).

@VARIANT_DETAIL@

### Metodología del experimento

1. **Dataset real** `eegbci` (PhysioNet), sujetos 1–12, corridas 1–2:
   228 269 muestras × 64 canales, filtrado bandpass 1–45 Hz, split por
   bloques (train/val/test). El montaje 10-20 es un subconjunto exacto de los
   64 canales.
2. **Proyección fija** `P` construida con `mapping.build_projection`
   (`leadfield` con truncado SVD automático `C_s//3`, `spline` con
   `smoothness = 1e-5 · C_c/C_s`).
3. **Inicialización lineal empírica**: `init_from_data` ajusta las matrices
   sobre las observaciones **proyectadas** (`x·P`), de modo que el latente
   (`C=64`) ancla al unipolar canónico y cada ruta arranca bien condicionada.
4. **Pérdida**: MSE estandarizado (Z-score por lote) sobre las 16 rutas
   (origen del montaje fuente → destino canónico).
5. **Evaluación (test)**: RMSE/correlación por ruta de las estimaciones del
   modelo contra las referencias canónicas verdaderas, comparadas con la
   **proyección analítica pura** `obs @ P` (columnas `*_proy`).
6. **Figuras**: curvas de aprendizaje, heatmap de RMSE por ruta y **mapas de
   calor del cuero cabelludo** (electrodos → manchas) en 4 etapas:
   observación → proyección analítica → modelo → verdad canónica.
"""

MONTAGE_VARIANT_DETAIL = {
    "montage_leadfield": (
        "estimación por SOLUCIÓN INVERSA con el lead field analítico",
        r"""
La variante **`montage_leadfield`** proyecta las observaciones del montaje
fuente con el operador de **solución inversa**

```
P = W_s (W_s G_s)^{+T} G_c^T ,      W_s = I − 11ᵀ/C_s
```

que estima los potenciales al infinito en las posiciones canónicas a partir de
los electrodos fuente (equivalente a REST pero entre montajes).

Como el problema inverso `W_s G_s` está fuertemente mal condicionado
(cond ≈ 1e15), los mínimos cuadrados sin regularizar amplifican el ruido real
(`ve` negativa). Se **trunca el SVD** a `n_components ≈ C_s//3` (modos
espaciales de baja frecuencia), lo que estabiliza la proyección (ve ≈ 0.27 en
10-20) y deja al autoencoder refinar el resto.

**Resultado esperado**: el modelo aprende a corregir la estimación inversa y
a componer las 4 referencias canónicas mejor que la proyección pura (la
columna `*_proy` de la tabla de evaluación es ese punto de partida).

**Resultado obtenido** (12 sujetos, test): el modelo supera la inversa pura —
RMSE cross ≈ **18.7 µV**, `r ≈ 0.623`, `ve ≈ 0.44` frente a ve ≈ 0.18 de la
proyección analítica. El desglose por ruta se muestra en la sección 4.
""",
    ),
    "montage_heatmap": (
        "estimación por interpólation esférica (Perrin) — actividad en manchas",
        r"""
La variante **`montage_heatmap`** proyecta con **splines esféricos** (Perrin,
1989), la misma interpolación de los topomapas clásicos: el campo de
potenciales se extiende de forma suave entre los electrodos y se representa
como un **heatmap** sobre el cuero cabelludo donde cada foco de actividad
aparece como una **mancha**.

El suavizado es **sensible a la densidad** del montaje (`adaptive_smoothness`):
cuanto más dispersos los electrodos (menos canales fuente por zona), mayor la
regularización ridge del spline (`smoothness = base · C_c/C_s`). Así la misma
pipeline sirve para configuraciones densas (64) o muy sparse (8).

**Resultado esperado**: la proyección es más suave que la inversa (vuelve peor
en *round-trip* solo con `P`, ve ≈ 0.10 en 10-20) pero muy estable; el
autoencoder universal la refina hasta aproximar las referencias canónicas.

**Resultado obtenido** (12 sujetos, test): a pesar de partir de la peor
proyección pura (ve ≈ 0.05), el modelo la supera con holgura y es **la mejor
de las dos variantes de montaje** — RMSE cross ≈ **16.3 µV**, `r ≈ 0.746`,
`ve ≈ 0.58`. Las características suaves del spline son más fáciles de refinar
que los modos de la solución inversa mal condicionada.
""",
    ),
}


# ---------------------------------------------------------------------------
# Celdas de código (compartidas entre todos los notebooks)
# ---------------------------------------------------------------------------

CELL_PLOT_TRAINING = r"""# Curvas de aprendizaje (pérdida estandarizada y MSE real)
plot_training(run_dir / "history.csv")
plt.show()"""

CELL_PLOT_ROUTES = r"""# Heatmap de RMSE real por ruta origen→destino (µV, escala log10)
plot_routes(metrics_df, value_col="rmse",
            title=f"RMSE real por ruta (µV) — {cfg.model.variant}")
plt.show()"""

CELL_MONTAGE_SETUP = r"""# ---- Configuración del entorno ----------------------------------------
# Renderizado inline de figuras (debe activarse antes de importar matplotlib)
%matplotlib inline

import os, sys
from pathlib import Path

# Ruta raíz del repo y paquetes propios (src/)
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

# Las rutas relativas (data/, runs/, config/) se resuelven contra la raíz,
# no contra el cwd del kernel (correcto incluso desde otro directorio).
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_transform.nb import (
    config_table, evaluate_montage, load_experiment, montage_inputs,
    plot_routes, plot_scalp, plot_training, train_variant,
)
from eeg_transform.training.trainer import build_model

# Modo notebook: semillas fijas
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
plt.rcParams["figure.dpi"] = 110

CONFIG = ROOT / "{config_rel}"
FORCE  = False          # True = reentrenar desde cero ignorando el checkpoint
"""

CELL_MONTAGE_CONFIG = r"""cfg, ds = load_experiment(CONFIG)
mi = montage_inputs(cfg, ds)
print(ds.summary())
print(f"Montaje fuente: {mi.montage} ({len(mi.src_names)} electrodos) "
      f"via '{mi.method}' → canónico {ds.n_channels} canales")
config_table(cfg).set_index(["sección", "parámetro"])"""

CELL_MONTAGE_ARCH = r"""# Arquitectura efectiva: autoencoder universal + proyección fija P
model = build_model(cfg, ds.n_channels, projection=mi.projection)
model.ensure_built()

n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Variante: {cfg.model.variant}  |  latente: {cfg.model.latent_dim}  |  "
      f"montaje origen: {len(mi.src_names)} → canónico: {ds.n_channels}  |  "
      f"parámetros entrenables: {n_params:,}")
print(f"Proyección P: {mi.projection.shape} "
      "(fija; el autoencoder refina y convierte de referencias)")
print(f"Ruta efectiva s→d: A = P · W_enc_s · W_dec_d")
_ = model  # se reutiliza en train/eval"""

CELL_MONTAGE_TRAIN = r"""# Entrenamiento (reutiliza best.weights.h5 si existe, salvo FORCE=True)
model, history = train_variant(cfg, ds, force=FORCE)
run_dir = Path(cfg.training.run_dir)
print(f"run_dir: {run_dir}")"""

CELL_MONTAGE_EVAL = r"""# Evaluación en test: RMSE/ve por ruta vs proyección analítica pura (P)
metrics_df, summary = evaluate_montage(cfg, ds, model, montage=mi)

print("=== RESUMEN MONTAJE (RMSE en µV): modelo vs proyección analítica ===")
print(summary.round(3).to_string(index=False))
print("\n=== DETALLE POR RUTA (test) — columnas *_proy = solo proyección P ===")
print(metrics_df.round(9).to_string(index=False))"""

CELL_MONTAGE_SCALP = r"""# Mapas de calor del cuero cabelludo: electrodos como manchas de actividad
# Filas = referencia; columnas: observación del montaje fuente → proyección
# analítica → modelo → verdad canónica (el instante de máxima amplitud).
plot_scalp(cfg, ds, model, montage=mi)
plt.show()"""

# ---------------------------------------------------------------------------
# Variante multi-configuración (``multi_montage``)
# ---------------------------------------------------------------------------

CELL_MULTICONFIG_SETUP = r"""# ---- Configuración del entorno ----------------------------------------
# Renderizado inline de figuras (debe activarse antes de importar matplotlib)
%matplotlib inline

import os, sys
from pathlib import Path

# Ruta raíz del repo y paquetes propios (src/)
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_transform.experiments.multi import multiconfig_summary
from eeg_transform.nb import (
    config_table, evaluate_multiconfig, evaluate_multiconfig_surface,
    evaluate_multiconfig_windowed, load_experiment, multiconfig_data,
    plot_multiconfig_bars, plot_multiconfig_heatmap, plot_multiconfig_scalps,
    plot_multiconfig_surface, plot_training, train_variant,
)
from eeg_transform.training.trainer import (
    build_multiconfig_model, is_multiconfig_variant,
)

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
plt.rcParams["figure.dpi"] = 110

CONFIG = ROOT / "{config_rel}"
FORCE  = False          # True = reentrenar desde cero ignorando el checkpoint
"""

CELL_MULTICONFIG_CONFIG = r"""cfg, ds = load_experiment(CONFIG)
data = multiconfig_data(cfg, ds)
assert is_multiconfig_variant(cfg), f"Variant {cfg.model.variant} no es multi_montage/multi_heatmap"
print(ds.summary())
print(multiconfig_summary(data), "\n")
config_table(cfg).set_index(["sección", "parámetro"])"""

CELL_MULTICONFIG_ARCH = r"""# Un latente canónico + proyecciones fijas por configuración P_s/Q_s
model = build_multiconfig_model(cfg, data)

n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Variante: {cfg.model.variant}  |  latente canónico: {model.n_canonical}  |  "
      f"configs: {len(model.configs)}  |  parámetros: {n_params:,}")
for lbl in model.configs:
    A = model.transfer_matrices(lbl)
    m = next(iter(A.values()))
    print(f"  {lbl:10s} P {model.projections[lbl].shape}  "
          f"Q {model.out_maps[lbl].shape}  A_s→d {m.shape}")
print("La predicción es INTRA-configuración: C_s → C_s (misma disposición).")
_ = model  # se reutiliza en train/eval"""

CELL_MULTICONFIG_TRAIN = r"""# Entrenamiento balanceado (mismas muestras por config/época); reutiliza el
# checkpoint de runs/{VARIANT}/best.weights.h5 si existe (salvo FORCE=True).
model, history = train_variant(cfg, ds, force=FORCE)
run_dir = Path(cfg.training.run_dir)
print(f"run_dir: {run_dir}")"""

CELL_MULTICONFIG_EVAL = r"""# Evaluación en test: RMSE/ve por configuración, diagonal y cruzada; las
# columnas *_ana son la línea base analítica T_d @ pinv(T_s) sobre el ancla.
metrics_df, summary = evaluate_multiconfig(cfg, ds, model, data)

print("=== RESUMEN MULTI-CONFIG (RMSE en µV): modelo vs línea base analítica ===")
print(summary.round(3).to_string(index=False))
print("\n=== DETALLE POR CONFIGURACIÓN (test) ===")
print(metrics_df.round(9).to_string(index=False))"""

CELL_MULTICONFIG_HEATMAP = r"""# Heatmap de RMSE real por config origen→destino (µV, escala log10)
plot_multiconfig_heatmap(metrics_df,
                         title=f"RMSE real multi-config (µV, log10) — {cfg.model.variant}")
plt.show()"""

CELL_MULTICONFIG_BARS = r"""# Barras: RMSE y ve por configuración, diagonal y cruzada, frente a la
# línea base analítica (*_ana) del propio montaje.
plot_multiconfig_bars(metrics_df,
                      title=f"RMSE/ve multi-config vs análisis — {cfg.model.variant}")
plt.show()"""

CELL_MULTICONFIG_SCALPS = r"""# Mapas de calor del cuero cabelludo por configuración. Filas = referencia
# (unipolar local, bipolar local, CAR, REST); columnas: observación →
# modelo (predicción intra-config) → verdad. El ancla REST es la referencia
# infinita simulada y las demás se derivan de sus operadores.
for label, fig in plot_multiconfig_scalps(cfg, model, data=data).items():
    print(f"--- {label} ---")
    plt.show()"""

CELL_MULTICONFIG_SURFACE_EVAL = r"""# Campo de superficie (solo ``multi_heatmap``): RMSE/r/VE del HEATMAP.
# La lectura se interpola con la matriz fija S_s (electrodos -> malla
# compartida); el término de pérdida del entrenamiento actúa sobre estas
# "manchas", no solo sobre cada electrodo por separado.
surface_df, surf_summary = evaluate_multiconfig_surface(model, data)

print("=== CAMPO DE SUPERFICIE (TEST) — RMSE en µV sobre la malla ===")
print(surf_summary.round(3).to_string(index=False))
print("\n=== DETALLE POR RUTA (campo, malla compartida) ===")
print(surface_df.round(9).to_string(index=False))"""

CELL_MULTICONFIG_SURFACE_FIG = r"""# Barras del campo de superficie por configuración: RMSE en la malla y VE
# del heatmap predicho (diagonal y cruzada entre referencias).
plot_multiconfig_surface(surface_df,
                         title=f"Campo de superficie (heatmap) — {cfg.model.variant}")
plt.show()"""

CELL_MULTICONFIG_FIELD_AGREE = r"""# Acuerdo de campo entre configuraciones (v2): los campos predichos de
# 10-20 / canónico / densos (misma malla) deben coincidir (consistencia B4).
from eeg_transform.evaluation import metrics as _metrics
field_agree_df = _metrics.evaluate_multiconfig_field_agreement(model, data, split="test")
if not field_agree_df.empty:
    print(f"rmse_field pareado medio: {field_agree_df['rmse_field'].mean()*1e6:.3f} uV")
    print(f"ve_field medio:            {field_agree_df['ve_field'].mean():.3f}")
print(field_agree_df.round(9).to_string(index=False))"""

CELL_MULTICONFIG_EVAL_WINDOWED = r"""# Evaluación VENTANEADA (universal_refs): ventanas causales deslizantes,
# salida del último paso; captura el beneficio de la recurrencia frente a la
# evaluación instantánea 2-D (que desactiva la cabeza temporal).
metrics_win, summary_win = evaluate_multiconfig_windowed(cfg, ds, model, data)
print("=== RESUMEN VENTANEADO (RMSE en µV) — cabeza temporal activa ===")
print(summary_win.round(3).to_string(index=False))
print("\n=== DETALLE POR CONFIGURACIÓN (test, ventaneado) ===")
print(metrics_win.round(9).to_string(index=False))
metrics_df, summary = evaluate_multiconfig(cfg, ds, model, data)
print("\n=== REFERENCIA INSTANTÁNEA (2-D, cabeza desactivada) ===")
print(summary.round(3).to_string(index=False))"""


def _code_cell(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


def _md_cell(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def _build_montage_markdown(variant: str) -> str:
    config_rel = f"config/{variant}.yaml"
    line, detail = MONTAGE_VARIANT_DETAIL[variant]
    text = _MONTAGE_BASE_DESCRIPTION.replace("@VARIANT@", variant)
    text = text.replace("@CONFIG_REL@", config_rel)
    text = text.replace("@VARIANT_LINE@", line)
    text = text.replace("@VARIANT_DETAIL@", detail)
    return text


def _montage_warning_markdown(variant: str) -> str:
    return (
        f"> **Nota de reproducción:** el modelo ya entrenado (12 sujetos) está en "
        f"`runs/{variant}`. Con `FORCE = False` el notebook **reutiliza el "
        f"checkpoint** sin reentrenar (segundos); póngalo en `True` solo para "
        f"reentrenar desde cero."
    )


def build_montage_notebook(variant: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (entorno)", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        _md_cell(f"# Variante `{variant}` — Universal EEG Transformer\n\n"
                 f"{_build_montage_markdown(variant)}\n\n"
                 f"{_montage_warning_markdown(variant)}"),
        _md_cell("## 1. Carga del experimento\n\n"
                 "Configuración YAML, dataset real cacheado y los **insumos de "
                 "montaje** (observaciones del montaje fuente + proyección `P`)."),
        _code_cell(CELL_MONTAGE_SETUP.format(config_rel=f"config/{variant}.yaml")),
        _code_cell(CELL_MONTAGE_CONFIG),
        _md_cell("## 2. Arquitectura\n\n"
                 "Autoencoder universal + **proyección fija** `P (C_s→C)`: la "
                 "ruta efectiva `s→d` es `A = P·W_enc·W_dec` y opera sobre las "
                 "observaciones del montaje fuente."),
        _code_cell(CELL_MONTAGE_ARCH),
        _md_cell("## 3. Entrenamiento\n\n"
                 "Igual que las variantes canónicas (Adam, Z-score por lote, "
                 "early stopping) pero sobre observaciones de montaje. "
                 "Reutiliza el checkpoint si existe."),
        _code_cell(CELL_MONTAGE_TRAIN),
        _md_cell("## 4. Evaluación en test\n\n"
                 "Métricas del **modelo** vs la **proyección analítica pura** "
                 "(`obs @ P`, columnas `*_proy`): la ganancia cuantifica el "
                 "refinamiento que aprende el transformador."),
        _code_cell(CELL_MONTAGE_EVAL),
        _md_cell("## 5. Figuras inline\n\n"
                 "Curvas de aprendizaje, heatmap de RMSE por ruta (modelo) y "
                 "los **mapas de calor del cuero cabelludo** que muestran la "
                 "actividad como manchas desde el montaje fuente hasta la "
                 "verdad canónica."),
        _code_cell(CELL_PLOT_TRAINING),
        _code_cell(CELL_PLOT_ROUTES),
        _code_cell(CELL_MONTAGE_SCALP),
        _md_cell("## Conclusiones\n\n"
                 "Consulte `docs/mapping_wip.md` y `docs/results_comparison.md` "
                 "para la interpretación comparativa de la unificación de "
                 "montajes frente a las variantes canónicas."),
    ]
    nb["cells"] = cells
    return nb


_MULTICONFIG_MARKDOWN = r"""
El **Universal EEG Transformer** en modo **multi-configuración** entrena **un
solo modelo** que acepta grabaciones en *cualquier* configuración de
electrodos y predice, **en esa misma configuración**, las medidas con una
referencia distinta (conversión de referencia intra-configuración).

### Las configuraciones (`mapping.configs`)

| Config | Canales | Origen |
|---|---|---|
| `10-20` | 19 | Subconjunto exacto de los 64 canales (selección de columnas) |
| `10-10` | 39 | Subconjunto exacto de los 64 canales |
| `canonical` | 64 | El montaje completo del dataset |
| `dense-128` | 128 | **Fundido denso**: mismo campo escalar REST interpolado |
| `dense-256` | 256 | **Fundido denso**: mismo campo escalar REST interpolado |

### Cómo se construyen las observaciones (fiel a la física)

1. **Ancla única**: la referencia al infinito (REST) del montaje canónico,
   obtenida con el lead field multicapa analítico (`T_canon pg`).
2. Cada configuración observa ese campo en sus propias posiciones:
   los subconjuntos reales son **selección de columnas** del ancla; los densos
   son **interpolación esférica (Perrin)** del ancla a `128/256` posiciones
   en el casquete (fibonacci, radio = mediana de la norma canónica).
3. Sobre esa observación se computan las 4 referencias **del propio montaje**
   (con su lead field): unipolar/bipolar/CAR locales y REST de configuración.
4. Proyecciones fijas `P_s`/`Q_s`: para subconjuntos, selección exacta
   (`Q` selecciona las columnas del core canonico); para densos, splines. La
   ruta efectiva `s→s` es `A_s = Q_s·core·P_s`.

### Fairness del experimento

- **Balanceo por configuración**: todas aportan las mismas muestras por
  lote/época (`multi_max_samples_per_split` por split), sin sesgar la
  minimización hacia las configs más densas.
- **Inicialización lineal empírica** sobre el canónico (`latent_dim = C`):
  el latente ancla al unipolar canónico y cada ruta arranca bien condicionada.
- **Pérdida**: MSE estandarizado por ruta (Z-score por lote) + MSE real.
- **Evaluación (test)**: RMSE/correlación/variación explicada **por
  configuración** (diagonal y cruzada entre referencias) comparadas con la
  **línea base analítica** `T_d @ pinv(T_s)` del propio montaje (columnas
  `*_ana`): la ganancia del modelo se lee restando ambas.

### Notas

- Las configs `dense-N` son simuladas (interpolación, no registros reales);
  las 10-20/10-10 son subconjuntos reales de los 64 canales (PhysioNet eegbci).
- Conceptos completos (física lead-field/REST, métricas, lectura de
  resultados): `docs/guia_conceptual.md`.
"""

_MULTI_HEATMAP_EXTRA = r"""
### Multi_heatmap: el "heatmap" es una salida entrenada

La variante **`multi_heatmap`** añade a `multi_montage` la lectura de la
actividad como **campo de superficie sobre una malla compartida** del cuero
cabelludo. Para cada configuración la observación se interpola a los
`grid_px²` nodos con la matriz fija `S_s (C_s → n_grid)` (la misma
`scalp_grid_matrix` de los topomapas):

$$\hat F^{(d)}_s = \hat x^{(d)}_s\, S_s^\top$$

El término de pérdida es el MSE estandarizado **sobre el patrón espacial**
(peso `model.surface_loss_weight = 0.1`), de modo que el modelo no solo acierta
electrodo a electrodo sino también la forma de las "manchas". Al ser `S_s`
fijo, todos los campos viven en la **misma malla**: los heatmaps de 19/64/128/
256 canales son comparables entre sí.
"""

_MULTI_HEATMAP_V2_EXTRA = r"""
### Multi_heatmap_v2: campo aprendible y consistencia física cruzada

La variante **`multi_heatmap_v2`** corrige la redundancia del campo de la v1
(donde `F̂ = x̂·S_sᵀ` con `S_s` fijo, un mero espejo suavizado) y añade
regularización física:

* **Interpolación aprendible (A2):** la matriz de campo `R_s` es entrenable,
  inicializada en la spline `S_s` y regularizada hacia ella.
* **Bucle electrodo↔campo (A1):** se exige `x̂ ≈ R_sᵀ·F̂ = (R_sᵀR_s)·x̂`, de
  modo que el campo sea una representación fiel e invertible.
* **Consistencia entre configuraciones (B4):** los campos de 10-20, canónico y
  densos (misma malla) deben coincidir, alineándolos sin señal externa.
* **Adaptadores por configuración (B5):** residual low-rank en el espacio
  latente canónico para dar capacidad local sin tocar el core compartido.
* **Suavizado temporal ligero (C7):** variación total entre muestras
  consecutivas (requiere dataset ordenado en tiempo).
* **Ponderación por incertidumbre (A3):** los pesos de los términos se
  aprenden como `log σ²` (multi-task) cuando `learn_uncertainty` está activo.
* **REST por configuración (C6):** `rest_rcond` se elige por configuración
  vía validación cruzada de la VE analítica cuando `leadfield.rest_rcond_cv`
  está activo.

La evaluación añade `metrics_field_agreement_test.csv` (acuerdo de campo
pareado entre configuraciones) como sustituto del topomapa real (D9).
"""

_UNIVERSAL_REFS_EXTRA = r"""
### Universal_refs: cabeza temporal de residuo sobre el núcleo lineal

La variante **`universal_refs`** añade a `multi_heatmap_v2` una **cabeza
temporal de residuo** (`model.temporal_cell`: `conv` por defecto, o
`gru`/`lstm`/`rnn`) que predice la corrección dinámica desde una ventana de
lo medido, manteniendo un núcleo instantáneo puramente lineal por debajo:

* Las cabezas se **inicializan a cero**: el arranque es idéntico al modelo
  lineal (ablation trivial) para cualquier `temporal_cell`.
* `conv`: bloques depthwise+pointwise sobre ventana **centrada**
  (`padding="same"`, acausal); `gru`/`lstm`/`rnn`: una célula recurrente
  **causal** (units = `temporal_channels`) generalizable a secuencias más
  largas que la ventana de entrenamiento.
* Con tensores 2-D la cabeza se desactiva y el forward coincide con
  `multi_heatmap_v2`.

La evaluación estándar alimenta tensores 2-D y **desactiva** la cabeza; el
beneficio temporal se mide con la **evaluación ventaneada** (ventanas causales
deslizantes, salida del último paso). En el CLI se genera
`metrics_windowed_test.csv`; aquí se compara esa métrica frente a la
instantánea.
"""


def _build_multiconfig_markdown(variant: str) -> str:
    text = _MULTICONFIG_MARKDOWN
    if variant in ("multi_heatmap", "multi_heatmap_v2"):
        extra = _MULTI_HEATMAP_V2_EXTRA if variant == "multi_heatmap_v2" \
            else _MULTI_HEATMAP_EXTRA
        text = text.rstrip() + "\n" + extra
    elif variant == "universal_refs":
        text = text.rstrip() + "\n" + _UNIVERSAL_REFS_EXTRA
    return text


def _multiconfig_warning_markdown(variant: str) -> str:
    return (
        f"> **Nota de reproducción:** el modelo ya entrenado (12 sujetos) está en "
        f"`runs/{variant}`. Con `FORCE = False` el notebook **reutiliza el "
        f"checkpoint** sin reentrenar (segundos); póngalo en `True` solo para "
        f"reentrenar desde cero."
    )


def build_multiconfig_notebook(variant: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (entorno)", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        _md_cell(f"# Variante `{variant}` — Universal EEG Transformer\n\n"
                 f"{_build_multiconfig_markdown(variant)}\n\n"
                 f"{_multiconfig_warning_markdown(variant)}"),
        _md_cell("## 1. Carga del experimento\n\n"
                 "Configuración YAML, dataset real cacheado y los **insumos "
                 "multi-configuración** (observaciones y referencias por "
                 "configuración)."),
        _code_cell(CELL_MULTICONFIG_SETUP.format(config_rel=f"config/{variant}.yaml")),
        _code_cell(CELL_MULTICONFIG_CONFIG),
        _md_cell("## 2. Arquitectura\n\n"
                 "Un autoencoder universal en el canónico + **proyecciones fijas** "
                 "`P_s`/`Q_s` por configuración. Las matrices efectivas "
                 "`A_s→s = Q_s·core·P_s` convierten las 4 referencias dentro de "
                 "cada configuración."),
        _code_cell(CELL_MULTICONFIG_ARCH),
        _md_cell("## 3. Entrenamiento\n\n"
                 "Adam (lr 1e-3), Z-score por lote, early stopping y reduce-LR. "
                 "Balanceado: mismas muestras por config/lote. Reutiliza el "
                 "checkpoint si existe."),
        _code_cell(CELL_MULTICONFIG_TRAIN),
        _md_cell("## 4. Evaluación en test\n\n"
                 "Métricas por configuración (diagonal/cruzada entre referencias) "
                 "frente a la **línea base analítica** `T_d @ pinv(T_s)` del propio "
                 "montaje (columnas `*_ana`). Para `universal_refs` la evaluación "
                 "es **ventaneada** (cabeza temporal activa) y se contrasta con la "
                 "instantánea 2-D."),
        _code_cell(CELL_MULTICONFIG_EVAL_WINDOWED
                   if variant == "universal_refs" else CELL_MULTICONFIG_EVAL),
        _md_cell("## 5. Figuras inline\n\n"
                 "Curvas de aprendizaje, heatmap de RMSE por config origen→destino, "
                 "barras modelo vs análisis, y los **mapas de calor del cuero "
                 "cabelludo** por configuración (observación → modelo → verdad)."),
        _code_cell(CELL_PLOT_TRAINING),
        _code_cell(CELL_MULTICONFIG_HEATMAP),
        _code_cell(CELL_MULTICONFIG_BARS),
        _code_cell(CELL_MULTICONFIG_SCALPS),
    ]
    if variant in ("multi_heatmap", "multi_heatmap_v2"):
        cells += [
            _md_cell("## 6. Campo de superficie (heatmap\n\n"
                     "La lectura como campo en la malla compartida es una salida "
                     "**entrenada** (peso `surface_loss_weight`): aquí se mide "
                     "RMSE/r/VE del patrón espacial y se compara por "
                     "configuración."),
            _code_cell(CELL_MULTICONFIG_SURFACE_EVAL),
            _code_cell(CELL_MULTICONFIG_SURFACE_FIG),
        ]
    if variant == "multi_heatmap_v2":
        cells += [
            _md_cell("## 7. Acuerdo de campo entre configuraciones\n\n"
                     "Métrica de la consistencia B4: los heatmaps predichos de "
                     "todas las configuraciones viven en la misma malla y deben "
                     "describir el mismo potencial de superficie."),
            _code_cell(CELL_MULTICONFIG_FIELD_AGREE),
        ]
    elif variant == "universal_refs":
        cells += [
            _md_cell("## 6. Metadatos de la evaluación ventaneada\n\n"
                     "La métrica temporal depende de la ventana de contexto: "
                     "configs sensorialmente densas (> canales que la cabeza "
                     "puede memorizar) deberían mostrar mayor beneficio de la "
                     "recurrencia. Los resultados quedan en "
                     "`runs/universal_refs/metrics_windowed_test.csv`."),
            _code_cell("""import os
from pathlib import Path
p = Path(cfg.training.run_dir) / "metrics_windowed_test.csv"
if p.exists():
    print(pd.read_csv(p).round(9).to_string(index=False))
else:
    print("Sin metrics_windowed_test.csv (la evalua el CLI con temporal_window > 0)")"""),
        ]
    cells += [
        _md_cell("## Conclusiones\n\n"
                 "Consulte `docs/guia_conceptual.md` (conceptos y métricas), "
                 "`docs/results_comparison.md` (comparativa de variantes) y los "
                 "resultados guardados en "
                 f"`runs/{variant}/metrics_test.csv` (y "
                 f"`runs/{variant}/metrics_surface_test.csv` para "
                 f"`multi_heatmap`)."),
    ]
    nb["cells"] = cells
    return nb


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ("montage_leadfield", "montage_heatmap"):
        nb = build_montage_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    for variant in ("multi_montage", "multi_heatmap", "multi_heatmap_v2",
                    "universal_refs"):
        nb = build_multiconfig_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    print("Listo. Notebooks en", OUT_DIR)


if __name__ == "__main__":
    main()