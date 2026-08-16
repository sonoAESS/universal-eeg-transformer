# Universal EEG Transformer

Autoencoder **lineal** multicabezal *All-to-All* que unifica la conversión entre cuatro referencias de EEG — `unipolar`, `bipolar`, `CAR` y `REST` — a través de un espacio latente central. Al ser puramente lineal, el mapeo efectivo de cada ruta es la matriz `A_{s→d} = W^enc_s W^dec_d` (C×C), lo que respeta la estructura algebraica de los campos electrostáticos del cuero cabelludo.

## Física y convenciones

* **Convención de matrices:** `X_ref = X @ M`, con `X` de forma `(T, C)` y `M` la matriz de referencia que actúa por filas.
* **`unipolar`:** resta del canal de referencia (`Cz`). **`bipolar`:** resta del canal siguiente (cadena, rango C−1). **`CAR`:** resta de la media de todos los canales. **`REST`:** paso a una referencia al infinito virtual, calculada con el *lead field* analítico.
* **Referencia del dato original:** `eegbci` graba contra la mastoides izquierda (`data.original_reference`). Se revisa y audita en el pipeline, pero los cuatro montajes son invariantes a ella: todos eliminan el offset constante instantáneo (diferencias en uni/bip, `W_avg` en CAR/REST), así que el entrenamiento no depende de la referencia de adquisición.
* **Lead field:** esfera concéntrica de 4 capas (cerebro/CSF/cráneo/piel) vía MNE (`make_sphere_model`), con radios relativos `[0.87, 0.9, 0.97, 1.0]` y conductividades `[0.33, 1.0, 0.0042, 0.33]` S/m, y grilla de dipolos de 10 mm.
* **Matriz REST (row-application):** `M_rest = W_avg (G (W_avg G)⁺)ᵀ`, con `G` el lead field y `W_avg = I − 11ᵀ/C`. Es exacta para fuentes en el *rowspace* de `W_avg G`; unipolar y CAR se recuperan de forma exacta, mientras que bipolar pierde la componente constante.
* **Pérdida:** MSE estandarizado con Z-score del objetivo por lote para equilibrar las 16 rutas independientemente de su escala.
* **Inicialización lineal empírica:** antes de entrenar, `W^enc_s = ridge(X_s→z)` y `W^dec_d = ridge(z→X_d)` con `z = X_unipolar` (latente de dimensión `C`), calculadas sobre una submuestra de entrenamiento. Como las referencias provienen de la misma señal por operadores lineales, esta factorización deja cada ruta en `var_expl ≈ 1` desde la época 0; sin ella el gradiente queda atrapado en cuencas degeneradas del autoencoder lineal (producto Glorot) y el modelo colapsa a predecir la media (`r ≈ 0`).

## Resultados (sujetos 1–4 de eegbci, split `block`)

Métricas de test tras el ajuste fino (latente = 64, 16 rutas, pérdida Z-score):

| Bloque | MSE (V²) | RMSE (V) | r |
|---|---|---|---|
| Auto-reconstrucción (diagonal) | 1.1e-12 | 6.7e-7 | 0.9999 |
| Transformación cruzada | 9.9e-13 | 8.0e-7 | 0.9987 |

Todas las rutas superan `r ≥ 0.99` (la más baja es `rest→bipolar`, `r ≈ 0.99`). REST (~1–2 µV RMSE) era el montaje dominante por amplitud (referencia al infinito); el resto ronda los 0.1–0.3 µV RMSE.

> Nota sobre `error_fro_rel`: se compara en el **subespacio observable** (`P = I − 11ᵀ/C` por ambos lados) porque el modo constante instantáneo no es recuperable desde ninguna referencia; sin ese centrado la métrica no refleja la calidad real del mapeo.

## Requisitos e instalación

Python 3.12+, gestión con [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                          # crea entorno e instala dependencias
uv run --with ... python -m pytest tests/   # o usar un venv existente
```

Con el entorno ya creado (`entorno/`), para ejecutar código y tests:

```bash
PYTHONPATH=src entorno/bin/python -m pytest tests/ -q
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli --help
```

La fuente de verdad de dependencias es `pyproject.toml` (ya no se usa `requirements.txt`).

## Uso

La CLI (`eeg-transform`) tiene tres comandos; `--config` alude a un YAML (ver `config/default.yaml`):

```bash
eeg-transform -c config/smoke.yaml build      # descarga PhysioNet eegbci y cachea dataset + lead field
eeg-transform -c config/smoke.yaml train      # entrena el transformador lineal
eeg-transform -c config/smoke.yaml eval       # métricas de test y figuras
eeg-transform -c config/smoke.yaml pipeline   # build + train + eval
```

* `build`: descarga los sujetos/corridas indicados, filtra (bandpass y notch), marca artefactos y canales malos por MAD z-score, revisa la referencia del dato original (`original_reference`) y construye las 4 referencias alineadas (unipolar, bipolar, CAR, REST), guardando `data/processed/dataset_{mode}_{subjects}.npz` y `*_leadfield.npz`. Splits por bloques temporales (`block`) o por sujeto (`subject`).
* `train`: escribe checkpoints en `runs/<nombre>/` (`best.weights.h5`, `model.keras`, `history.csv`).
* `eval`: imprime tabla de métricas por ruta (MSE/RMSE/MAE/r), compara las matrices efectivas con las analíticas y guarda figuras en `runs/<nombre>/figs/`.

## Estructura

```text
├── pyproject.toml               # dependencias (uv/pip instal ")
├── uv.lock
├── config/default.yaml          # configuración por defecto
├── src/eeg_transform/
│   ├── config.py                # dataclasses + carga/validación YAML
│   ├── references.py            # matrices de referencia + REST
│   ├── leadfield.py             # lead field analítico (esfera 4 capas, MNE)
│   ├── data/loader.py           # descarga eegbci y preprocesado
│   ├── data/dataset.py          # dataset multi-referencia + splits + caché
│   ├── models/universal_transformer.py   # autoencoder lineal All-to-All
│   ├── training/trainer.py      # bucles/callbacks TF
│   ├── evaluation/{metrics,plots}.py
│   └── cli.py                   # comandos build/train/eval/pipeline
└── tests/
    ├── test_physics.py          # propiedades de las referencias y REST
    ├── test_config_ds.py        # config y dataset
    └── test_integration.py      # el modelo aprende los mapas sobre datos
```

## Notas

* El dataset por defecto son los sujetos 1–4 de `eegbci` (PhysioNet); el primer `build` descarga los EDFs (~decenas de MB por sujeto).
* El modelo se entrena con el inicializador `GlorotUniform`; inicializadores ortogonales degradan la convergencia en la factorización lineal de bajo rango.