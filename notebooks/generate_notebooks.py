#!/usr/bin/env python3
"""Genera los notebooks de exploración de variantes del Universal EEG Transformer.

Dos familias de notebooks autocontenidos en ``notebooks/``:

* **Canónicas** (free, group, projected, soft_group, wide, bottleneck):
  conversión entre las 4 referencias en el montaje fijo de 64 canales.

* **Montaje** (montage_leadfield, montage_heatmap): unificación de montajes.
  El transformador universal estima las 4 referencias canónicas desde las
  observaciones de un montaje de electrodos (p. ej. 10-20, 19 canales)
  proyectadas al espacio canónico con una matriz fija ``P``:

  - ``montage_leadfield``: solución inversa con el lead field analítico
    (SVD truncado a ``C_s//3`` porque el problema está mal condicionado).
  - ``montage_heatmap``: interpolación esférica (Perrin) — la actividad se
    ve como manchas sobre el cuero cabelludo, suavizadas según la densidad
    del montaje.

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

_BASE_DESCRIPTION = r"""
El **Universal EEG Transformer** es un autoencoder **lineal** multientrada/
multisalida que unifica la conversión entre cuatro referencias de EEG —
`unipolar`, `bipolar`, `CAR` y `REST` — a través de un espacio latente central
`z`:

$$\quad x_s \xrightarrow{W^{enc}_s} z \xrightarrow{W^{dec}_d} \hat{x}_d$$

Toda la red es lineal (sin activaciones), de modo que el mapeo efectivo por
ruta es la matriz `A_{s→d} = W^{enc}_s\, W^{dec}_d` (C×C), actuando
`X_ref = X @ A` sobre señales `(tiempo, canales)`. Esto respeta la estructura
algebraica de los campos electrostáticos sobre el cuero cabelludo: suma y
reescalado se preservan exactamente.

### Las cuatro referencias (física)

| Referencia | Operador | Notas |
|---|---|---|
| `unipolar` | resta del canal `Cz` | rango C−1 |
| `bipolar` | resta del canal siguiente (cadena) | rango C−1 |
| `CAR` | resta de la media de todos los canales | rango C−1 |
| `REST` | referencia al infinito virtual (lead field analítico multicapa) | rango C−1 |

Todas anulan la componente constante instantánea, por lo que su valor **no
depende** de la referencia física de adquisición (en `eegbci`, mastoides
izquierda). La señal vive en el **subespacio observable** de dimensión `C−1`
(`P = I − 11ᵀ/C` es el proyector de centrado usado por las métricas).

### Identidad de la variante

**`@VARIANT@`** — @VARIANT_LINE@ (config `@CONFIG_REL@`).

@VARIANT_DETAIL@

### Metodología del experimento

1. **Dataset real** `eegbci` (PhysioNet), sujetos 1–12, corridas 1–2:
   228 269 muestras × 64 canales, filtrado bandpass 1–45 Hz, split temporal
   por bloques (train 60 % / val 20 % / test 20 %).
2. **Lead field analítico**: esfera concéntrica de 4 capas (cerebro/CSF/
   cráneo/piel) con MNE `make_sphere_model`, radios relativos
   `[0.87, 0.9, 0.97, 1.0]`, conductividades `[0.33, 1.0, 0.0042, 0.33]` S/m,
   grilla de dipolos de 10 mm. Matriz REST (aplicación por filas):
   `M_rest = W_avg (G (W_avg G)⁺)ᵀ`.
3. **Inicialización lineal empírica**: `init_from_data` ajusta
   `W^enc_s = ridge(X_s→z)` y `W^dec_d = ridge(z→X_d)` con
   `z = X_unipolar` (latente `C = 64`). Deja cada ruta con `var_expl ≈ 1`
   desde la época 0; sin ella el gradiente del autoencoder lineal colapsa a
   predecir la media (`r ≈ 0`). Se aplica **solo si `latent_dim == C`**.
4. **Pérdida**: MSE estandarizado con Z-score del objetivo por lote, para
   equilibrar las 16 rutas independientemente de su escala.
5. **Entrenamiento**: Adam lr 1e-3, 60 épocas, batch 1024, reduce-LR en
   meseta, early stopping con restauración de los mejores pesos, semillas fijas.
6. **Evaluación (test)**:
   * `metrics_eval`: MSE/RMSE/MAE y correlación de Pearson por ruta.
   * `transfer_error_matrix`: error Frobenius relativo en el subespacio
     observable respecto a las matrices analíticas `T_d pinv(T_s)`.
   * `composition_error_table`: error de composición `s→d→u` (física de grupo).
   * Figuras: curvas de aprendizaje, heatmap de RMSE por ruta, y trazas reales
     vs predichas en el canal `Cz`.
"""

VARIANT_DETAIL = {
    "free": (
        "arquitectura libre del autoencoder (8 matrices C×C sin restricciones)",
        r"""
La variante **`free`** es el punto de partida: 8 matrices C×C aprendidas
(encoder + decoder por montaje). No impone ninguna restricción física a las
matrices efectivas `A_{s→d}`, que formalmente podrían introducir un modo
constante espurio o componer de forma no transitiva. Sirve de línea base:
mide cuánta física aprende el modelo solo con los datos.

**Resultados esperados** (tabla comparativa de `docs/results_comparison.md`):
RMSE diag ≈ **0.69 µV**, RMSE cross ≈ **0.66 µV**, `r_cross` ≈ **0.9997**,
`comp_medio` ≈ **0.067**. Buena precisión y composición aceptable; `projected`
y `soft_group` la superan en precisión y física respectivamente.
""",
    ),
    "group": (
        "estructura de grupo exacta (decodificador = pseudo-inversa del encoder)",
        r"""
La variante **`group`** impone de forma **exacta** la física de grupo
fijando el decodificador de cada montaje como la pseudo-inversa de su encoder:
`W^{dec}_k = (W^{enc}_k)⁺`. Solo hay 4 matrices aprendidas (los encoders), con
centrado doble `W = P W_raw P` para que todos compartan el subespacio
observable. Con esto:

```
A_{s→d}      = W_s (W_d)⁺
A_{s→s}      = W_s (W_s)⁺  =  proyector (≈ identidad observable)
A_{s→d}A_{d→u} = A_{s→u}    (transitividad EXACTA)
```

El encadenado analítico habitual `T_d pinv(T_s)` **no** es un grupo
(comp_medio ≈ 0.59, verificado numéricamente); esta variante lo garantiza por
construcción.

> **Resultado esperado** (`docs/results_comparison.md`): comp_medio = **0.000**
> exacto, pero RMSE ≈ **133 µV** y `r ≈ 0.71`: la rigidez de la pseudo-inversa
> asume que **todas** las referencias comparten el mismo subespacio observable,
> y el rowspace de REST (lead field a infinito) no coincide con el de
> uni/bip/CAR. La consistencia es exacta sobre un subespacio *equivocado* — el
> mismo fracaso que el baseline `T_d pinv(T_s)`.
""",
    ),
    "projected": (
        "anulación del modo constante por construcción (W = P W_raw)",
        r"""
La variante **`projected`** parametriza toda matriz aprendida como
`W_eff = P W_raw` con `P = I − 11ᵀ/C`. Como `1ᵀP = 0`, toda salida anula la
entrada constante instantánea: cada ruta es una **referencia válida** (no puede
introducir un DC espurio ni componer "pseudo-referencias"). Es la restricción
mínima que respeta la física de adquisición, dejando el resto del aprendizaje
libre (8 matrices).

**Resultados esperados** (`docs/results_comparison.md`): es la mejor precisión
global — RMSE cross ≈ **0.28 µV**, `r_cross` ≈ **1.0000** — y además
`comp_medio` ≈ **0.034** (mitad que `free`).
""",
    ),
    "soft_group": (
        "grupo suave por regularización de composición en la pérdida",
        r"""
La variante **`soft_group`** usa la arquitectura libre (8 matrices) pero
añade a la pérdida una **penalización suave** de la consistencia de
composición, por lote:

```
loss = loss_rutas + w · ⟨ ||P(A_{s→d}A_{d→u} − A_{s→u})P||_F / ||P A_{s→u} P||_F ⟩_{s,d,u}
```

con `w = 0.5` (config `soft_group.yaml`). Interpola entre `free` (sin
restricción) y `group` (estructura exacta): obliga a componer bien
`A_{s→d}A_{d→u} ≈ A_{s→u}` **sin** la pseudo-inversa rígida que degrada la
convergencia de `group`.

**Resultados esperados** (`docs/results_comparison.md`):
`comp_medio` ≈ **0.005** (~120× menor que el encadenado analítico) con RMSE
cross ≈ **0.92 µV**. Si la transitividad `s→d→u` es requisito (cascadas de
re-referencia), es la opción físicamente más fiel.
""",
    ),
    "wide": (
        "free con latente sobredimensionado (128 > C = 64)",
        r"""
La variante **`wide`** es `free` con `latent_dim = 128` (> C = 64): explora
si un espacio latente más amplio mejora la reconstrucción o solo añade
parámetros sin ganancia física. Por tener `latent_dim != C`, **no** recibe la
inicialización lineal empírica (`init_from_data` la exige igual a C), por lo
que parte de la caída se atribuye también a esa ausencia.

**Resultados esperados** (`docs/results_comparison.md`): empeora —
RMSE cross ≈ **1.06 µV**, `r_cross` ≈ **0.996**, `comp_medio` ≈ **0.143**.
El latente **óptimo es C**: parámetros extra no aportan física.
""",
    ),
    "bottleneck": (
        "free con latente comprimido (32 < C = 64)",
        r"""
La variante **`bottleneck`** es `free` con `latent_dim = 32` (< C = 64):
explora si la información de referencia se comprime en un subespacio de menor
dimensión o si cada referencia necesita su propio subespacio de la señal.
Como `wide`, no recibe `init_from_data` (latente ≠ C).

**Resultados esperados** (`docs/results_comparison.md`): colapsa —
RMSE cross ≈ **8.1 µV**, `r_cross` ≈ **0.903**, `comp_medio` ≈ **0.127**.
Comprimir el latente destruye información; cada referencia ocupa su propio
subespacio dentro de la señal.
""",
    ),
}

INTROS: dict[str, str] = {}


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


def _build_markdown(variant: str) -> str:
    config_rel = "config/default.yaml" if variant == "free" else f"config/{variant}.yaml"
    line, detail = VARIANT_DETAIL[variant]
    # placeholder @VARIANT@ para no colisionar con las claves LaTeX {…}
    text = _BASE_DESCRIPTION.replace("@VARIANT@", variant)
    text = text.replace("@CONFIG_REL@", config_rel)
    text = text.replace("@VARIANT_LINE@", line)
    text = text.replace("@VARIANT_DETAIL@", detail)
    return text


# ---------------------------------------------------------------------------
# Celdas de código (compartidas entre todos los notebooks)
# ---------------------------------------------------------------------------

CELL_SETUP = r"""# ---- Configuración del entorno ----------------------------------------
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
    config_table, evaluate, load_experiment,
    plot_routes, plot_training, plot_traces, summarize, train_variant,
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

CELL_CONFIG = r"""cfg, ds = load_experiment(CONFIG)
print(ds.summary())
config_table(cfg).set_index(["sección", "parámetro"])"""

CELL_ARCH_SUMMARY = r"""# Arquitectura e hiperparámetros efectivos del modelo
model = build_model(cfg, ds.n_channels)
model.ensure_built()

n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Variante: {cfg.model.variant}  |  latente: {cfg.model.latent_dim}  |  "
      f"canales: {ds.n_channels}  |  parámetros entrenables: {n_params:,}")
print(f"Configuración de modelo:\n{config_table(cfg).query('sección == \"model\"').to_string(index=False)}")
_ = model  # se reutiliza en train/eval"""

CELL_TRAIN = r"""# Entrenamiento (reutiliza best.weights.h5 si existe, salvo FORCE=True)
model, history = train_variant(cfg, ds, force=FORCE)
run_dir = Path(cfg.training.run_dir)
print(f"run_dir: {run_dir}")"""

CELL_EVAL = r"""# Evaluación en test: métricas por ruta + física (Frobenius y composición)
metrics_df, err_matrix, comp = evaluate(cfg, ds, model)

print("=== RESUMEN (RMSE en µV) ===")
print(summarize(metrics_df).to_string(index=False))
print("\n=== ERROR FROBENIUS RELATIVO vs ANALÍTICO (subespacio observable) ===")
print(err_matrix.round(4).to_string(index=False))
print("\n=== CONSISTENCIA DE COMPOSICIÓN s→d→u ===")
print(comp.round(4).to_string(index=False))"""

CELL_PLOT_TRAINING = r"""# Curvas de aprendizaje (pérdida estandarizada y MSE real)
plot_training(run_dir / "history.csv")
plt.show()"""

CELL_PLOT_ROUTES = r"""# Heatmap de RMSE real por ruta origen→destino (µV, escala log10)
plot_routes(metrics_df, value_col="rmse",
            title=f"RMSE real por ruta (µV) — {cfg.model.variant}")
plt.show()"""

CELL_PLOT_TRACES = r"""# Trazas reales vs predichas en el canal Cz (test, rutas representativas)
plot_traces(model, ds, cfg)
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
    load_experiment, multiconfig_data,
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
assert is_multiconfig_variant(cfg), "Variant {cfg.model.variant} no es multi_montage/multi_heatmap"
print(ds.summary())
print(multiconfig_summary(data), "\n")
config_table(cfg).set_index(["sección", "parámetro"])"""

CELL_MULTICONFIG_ARCH = r"""# Un latente canónico + proyecciones fijas por configuración P_s/Q_s
model = build_multiconfig_model(cfg, ds, data)
model.ensure_built()

n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"Variante: {cfg.model.variant}  |  latente: {cfg.model.n_canonical}  |  "
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


def _code_cell(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


def _md_cell(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def _warning_markdown(variant: str) -> str:
    return (
        f"> **Nota de reproducción:** el modelo ya entrenado (12 sujetos) está en "
        f"`runs/{variant}`. Con `FORCE = False` el notebook **reutiliza el "
        f"checkpoint** sin reentrenar (segundos); póngalo en `True` solo para "
        f"reentrenar desde cero."
    )


def build_notebook(variant: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (entorno)", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = [
        _md_cell(f"# Variante `{variant}` — Universal EEG Transformer\n\n"
                 f"{_build_markdown(variant)}\n\n{_warning_markdown(variant)}"),
        _md_cell("## 1. Carga del experimento\n\n"
                 "Se cargan la configuración YAML y el dataset real (cacheado). "
                 "El resumen muestra las muestras por split y el lead field."),
        _code_cell(CELL_SETUP.format(config_rel=(
            "config/default.yaml" if variant == "free" else f"config/{variant}.yaml"
        ))),
        _code_cell(CELL_CONFIG),
        _md_cell("## 2. Arquitectura\n\n"
                 "La red es un autoencoder lineal; las matrices efectivas por "
                 "ruta son `A_{s→d} = W_enc·W_dec`. Se muestran los "
                 "hiperparámetros efectivos."),
        _code_cell(CELL_ARCH_SUMMARY),
        _md_cell("## 3. Entrenamiento\n\n"
                 "Entrena con Adam (lr 1e-3), Z-score por lote, early stopping "
                 "y reduce-LR. Reutiliza el checkpoint si existe."),
        _code_cell(CELL_TRAIN),
        _md_cell("## 4. Evaluación en test\n\n"
                 "Tres tablas: resumen diagonal/cruzada (RMSE µV), error "
                 "Frobenius relativo frente a las matrices analíticas, y "
                 "consistencia de composición (física de grupo)."),
        _code_cell(CELL_EVAL),
        _md_cell("## 5. Figuras inline\n\n"
                 "Las mismas visualizaciones que `eval` guarda en "
                 "`runs/<variante>/figs/`, mostradas aquí directamente."),
        _code_cell(CELL_PLOT_TRAINING),
        _code_cell(CELL_PLOT_ROUTES),
        _code_cell(CELL_PLOT_TRACES),
        _md_cell("## Conclusiones\n\n"
                 "Consulte `docs/results_comparison.md` para la interpretación "
                 "comparativa completa de todas las variantes. Para esta variante, "
                 "los resultados esperados se detallan en la introducción."),
    ]
    nb["cells"] = cells
    return nb


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


def _build_multiconfig_markdown(variant: str) -> str:
    text = _MULTICONFIG_MARKDOWN
    if variant == "multi_heatmap":
        text = text.rstrip() + "\n" + _MULTI_HEATMAP_EXTRA
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
                 "montaje (columnas `*_ana`)."),
        _code_cell(CELL_MULTICONFIG_EVAL),
        _md_cell("## 5. Figuras inline\n\n"
                 "Curvas de aprendizaje, heatmap de RMSE por config origen→destino, "
                 "barras modelo vs análisis, y los **mapas de calor del cuero "
                 "cabelludo** por configuración (observación → modelo → verdad)."),
        _code_cell(CELL_PLOT_TRAINING),
        _code_cell(CELL_MULTICONFIG_HEATMAP),
        _code_cell(CELL_MULTICONFIG_BARS),
        _code_cell(CELL_MULTICONFIG_SCALPS),
    ]
    if variant == "multi_heatmap":
        cells += [
            _md_cell("## 6. Campo de superficie (heatmap\n\n"
                     "La lectura como campo en la malla compartida es una salida "
                     "**entrenada** (peso `surface_loss_weight`): aquí se mide "
                     "RMSE/r/VE del patrón espacial y se compara por "
                     "configuración."),
            _code_cell(CELL_MULTICONFIG_SURFACE_EVAL),
            _code_cell(CELL_MULTICONFIG_SURFACE_FIG),
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
    for variant in ("free", "group", "projected", "soft_group", "wide", "bottleneck"):
        nb = build_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    for variant in ("montage_leadfield", "montage_heatmap"):
        nb = build_montage_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    for variant in ("multi_montage", "multi_heatmap"):
        nb = build_multiconfig_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    print("Listo. Notebooks en", OUT_DIR)


if __name__ == "__main__":
    main()