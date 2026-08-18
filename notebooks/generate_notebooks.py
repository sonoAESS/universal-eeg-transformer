#!/usr/bin/env python3
"""Genera los notebooks de exploración de variantes del Universal EEG Transformer.

Cada variante (free, group, projected, soft_group, wide, bottleneck) tiene un
notebook autocontenido en ``notebooks/`` con la misma estructura:

1. Introducción (markdown): metodología, arquitectura y física de la variante.
2. Configuración (código): carga del YAML, dataset cacheado y resumen.
3. Entrenamiento (código): reutiliza el checkpoint si existe (sin reentrenar).
4. Evaluación (código): métricas por ruta, error Frobenius y composición.
5. Figuras inline: curvas de aprendizaje, heatmap de RMSE y trazas reales vs
   predichas (mismas visualizaciones que produce ``eval`` en ``runs/<nombre>/figs``).

Se mantienen buenas prácticas: la lógica pesada vive en ``src/eeg_transform``
(función ``eeg_transform.nb``) y los notebooks son solo presentación; semillas
fijas; un flag ``FORCE`` para reentrenar desde cero de forma explícita.

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat_version = 4
    for variant in ("free", "group", "projected", "soft_group", "wide", "bottleneck"):
        nb = build_notebook(variant)
        path = OUT_DIR / f"{variant}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Generado: {path}")
    print("Listo. Notebooks en", OUT_DIR)


if __name__ == "__main__":
    main()