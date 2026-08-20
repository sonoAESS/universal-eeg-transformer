# Guía conceptual: EEG, referencias y el Universal EEG Transformer

Este documento explica, desde cero, los conceptos que sustentan el proyecto:
qué es una referencia en EEG, los cuatro montajes que se unifican, la física y
la matemática detrás de cada herramienta (lead field, REST, splines), cómo leer
las métricas (qué es un valor bueno o malo) y cómo funcionan las arquitecturas
del transformador, además de las características del conjunto de datos usado.

---

## 1. ¿Qué es un EEG y qué es una "referencia"?

Un **electroencefalograma (EEG)** mide la diferencia de potencial eléctrico
(de orden de microvoltios, µV) entre un par de puntos del cuero cabelludo. En
ningún punto se puede medir un potencial "absoluto": **siempre se mide una
diferencia con respecto a un electrodo de referencia** o frente a una
referencia computada.

Formalmente, la señal adquirida con la referencia *r* es

```
X_r = X_inf - X[:, r]        (se resta el potencial del electrodo r)
```

donde `X_inf` es el potencial "al infinito" (ideal). Toda conversión entre
referencias es una **transformación lineal** sobre el vector de canales, y por
eso un **modelo lineal** puede —en principio— aprenderla de forma exacta.

### La referencia física del dato
El dato original (p. ej. *EEG Motor Movement/Imagery Dataset* de PhysioNet)
se graba contra la **mastoides izquierda**. Ese offset constante por instante no
interesa: los cuatro montajes que usamos eliminan cualquier componente
constante instantánea y por ello son **invariantes** a la referencia física de
adquisición.

### Los cuatro montajes o referencias canónicas
Todas se representan como matrices `M (C×C)` que se aplican por filas,
`X_ref = X @ M`:

| Montaje | Operador | Qué hace |
|---|---|---|
| **Unipolar** | `M = I - e_u 1ᵀ` | Resta un canal de referencia fijo (por defecto `Cz`) a todos los canales. |
| **Bipolar** | `M[i,j] = δij − δ(i+1)modC, j` | Cadena diferencial: cada canal se referencia al siguiente; anula componentes lentas y el modo común. |
| **CAR** (promedio común) | `M = I − (1/C)11ᵀ` | Resta el promedio instantáneo de todos los canales. |
| **REST** (referencia al infinito) | `M = W (G_L(WG_L)⁺)ᵀ` | Estima el potencial al infinito con el modelo de conducción de volumen (Yao, 2001). |

Las cuatro devuelven señales que satisfacen una propiedad clave: **anulan el
modo constante instantáneo** (cada fila suma 0), por lo que ninguna depende de
la referencia física original.

---

## 2. La física: potencial al infinito, lead field y REST

### Modelo de conducción de volumen
El cráneo y los tejidos conducen las corrientes neuronales hasta el cuero
cabelludo. El proyecto usa el modelo analítico de **esfera concéntrica
multicapa** (de MNE), con 4 capas —cerebro, LCR, cráneo y piel— de
conductividades típicas:

```
head_radius = 0.09 m
rel_radii   = [0.87, 0.90, 0.97, 1.00]   (fracciones del radio exterior)
sigmas      = [0.33, 1.00, 0.0042, 0.33] (S/m)
```

Las fuentes se colocan en una rejilla volumétrica del cerebro (espaciado
10 mm) y se calcula el **lead field** `G`: la matriz de ganancia
`(electrodos × 3·fuentes)` que relaciona los momentos dipolares `q` de las
fuentes con los potenciales en los electrodos,

```
V = G q
```

### REST (referencia al infinito, Yao 2001)
Dado el lead field, la estimación del potencial al infinito desde una señal
referenciada al promedio `V_avg` es

```
V_inf ≈ L V_avg,   L = G_L (W G_L)⁺ ,   W = I − (1/C)11ᵀ
```

Como la señal original puede referenciarse a CAR con `W`, la matriz de la
referencia REST es

```
M_rest = W Lᵀ,   X_rest = X M_rest
```

### Problema inverso entre montajes
Si conocemos el lead field de un montaje fuente `G_s (C_s×N)` y del canónico
`G_c (C_c×N)`, estimar los potenciales canónicos desde los observados es
resolver un **problema inverso**:

```
P = W_s (W_s G_s)⁺ᵀ G_cᵀ ,   X_c ≈ X_s P
```

La matriz `W_s G_s` está **muy mal condicionada** (condicionamiento del orden
de 10¹⁵). Para estabilizar se truncá el SVD a los modos de mayor valor
singular (en la práctica `n_components ≈ C_s // 3`). Sin regularizar, la
solución amplifica ruido y la calidad empeora (VE negativa) aunque sea exacta
en datos sin ruido.

### Interpolación esférica (spline de Perrin)
Para montajes densos o mapas de calor se interpola el campo de potenciales
sobre la **esfera unitaria** con el núcleo de Legendre

```
G(x) = Σ_{n=1}^{N} (2n+1)/(n(n+1))⁴ P_n(x)
```

resolviendo un sistema lineal con *gauge* de suma nula y regularización ridge
(`smoothness`). Con suavizado **adaptativo por densidad**
(`smoothness × C_dst/C_src`) los montajes más dispersos se difuminan más.

---

## 3. La arquitectura: autoencoder lineal *All-to-All*

### Idea central
Cada referencia de origen se codifica con un encoder lineal `W_enc^s` hacia un
**espacio latente central** `z`, y cada referencia destino se decodifica con un
cabezal lineal `W_dec^d`:

```
x_s →(W_enc^s)→ z →(W_dec^d)→ x̂_d      A_{s→d} = W_enc^s W_dec^d
```

La red es **lineal** (sin activaciones) porque la física subyacente es lineal:
preserva la suma y el escalado de señales.

### Pérdida
Se entrena con una pérdida **adimensional** (*Z-score por lote*): cada ruta se
normaliza por la desviación de su objetivo para que las 16 rutas (4×4) pesen
igual en la retropropagación. También se registran métricas en unidades reales
(V, V²).

### Variantes
| Variante | Qué hace | Cuándo usarla |
|---|---|---|
| `free` | 8 matrices aprendidas sin restricciones. | Línea base de máxima flexibilidad. |
| `projected` | Cada matriz se parametriza como `P W` con `P = I − 11ᵀ/C`: por construcción toda salida anula el modo constante. | Precisión en rutas individuales. |
| `group` | `W_dec = (W_enc)⁺` (pseudo-inversa): produce composición **exacta** (`A_{s→d}A_{d→u}=A_{s→u}`) y auto-reconstrucción. | Cuando importa la física de grupo. |
| `soft_group` | Igual que `free` pero con penalización suave de composición en la pérdida. | Compromiso precisión/física. |
| `montage_*` | Un montaje fuente de `C_s` electrodos entra por una **proyección fija** `P` y el modelo predice las referencias canónicas (64). | Unificar un montaje concreto al espacio canónico. |
| `multi_montage` | Entrena **varias configuraciones a la vez** (19/64/128/256) con el mismo autoencoder; cada una se embebe con `P_s` y se lee con `Q_s`, y el modelo predice referencias **en la propia configuración**. | Un modelo universal para cualquier configuración de electrodos. |
| `multi_heatmap` | Igual que `multi_montage` + **campo de superficie**: además del MSE por electrodo, la actividad se lee y se entrena como *heatmap* sobre una **malla compartida** del cuero cabelludo (la de los topomapas). | Un modelo universal cuyo topomapa también es fiel, no solo los electrodos. |

### Variante `multi_montage` (nueva)
Ecuación por configuración `s`:

```
x̂(a→b) = Q_s · W_dec^b · W_enc^a · P_s · x_a
```

con `P_s` (C_s→64) la proyección fija al espacio canónico y `Q_s` (64→C_s) el
retorno a las posiciones del montaje. Las matrices `W_enc`/`W_dec` se comparten
entre todas las configuraciones y los datos se **balancean por construcción**:
todas las configuraciones usan los mismos índices y el mismo presupuesto de
muestras por split.

Configuraciones disponibles:
- `10-20`: 19 canales reales (subconjunto exacto del montaje 10-10 nativo).
- `10-10`: 39 canales reales (subconjunto mayor).
- `canonical`: los 64 electrodos nativos (10-10).
- `dense-N`: N posiciones simuladas cuasi-uniformes sobre el casquete (p. ej.
  `dense-128`, `dense-256`), coherentes en cobertura con el montaje canónico.

### Variante `multi_heatmap`: el topomapa es una salida entrenada

`multi_heatmap` = `multi_montage` + lectura de la actividad como **campo de
superficie** sobre una **malla compartida** del cuero cabelludo. Para cada
configuración `s`, la matriz fija `S_s` (C_s → n_grid) lleva un vector de
electrodos a los `grid_px²` nodos de la malla (la misma interpolación
esférica de `scalp_grid_matrix` y de los topomapas):

```
F̂(d)_s = x̂(d)_s · S_sᵀ          (campo en la malla, n_grid por instante)
```

La pérdida combina el MSE estandarizado por electrodo (idéntico a `multi_montage`)
con el MSE estandarizado **sobre el campo** valuado en la malla, con peso
`model.surface_loss_weight` (0.1):

```
loss = <Z-loss por ruta (electrodos)> + 0.1 · <Z-loss por ruta (campo)>
```

Como `S_s` es fijo e independiente de la referencia, **todos los campos viven
en la misma malla**: los heatmaps de 19/64/128/256 canales son comparables en
una sola figura. Requiere `mapping.method: spline` (el "heatmap" presupone la
interpolación suave sobre el casquete).

Los resultados se guardan en `runs/<variante>/metrics_surface_test.csv` con
columnas `rmse_field`/`r_field`/`ve_field` por ruta y configuración.

---

## 4. Conjunto de datos

**Datasets**

| Dataset | Enlace | Canales | Referencia original | Cómo se usa |
|---|---|---|---|---|
| EEG Motor Movement/Imagery (eegbci) | https://physionet.org/content/eegmmidb/ | 64 (10-10) | Mastoides izquierda | Dato base; sujetos 1–12, 2 corridas (abrir/cerrar ojos) |
| Montaje 10-20 | — (subconjunto real) | 19 | Id. | Configuración de electrodos fuente |
| Montajes densos simulados | — (generados con spline esférico) | 128 / 256 | Id. | Mismo campo escalar muestreado más denso |

Preprocesado: filtro banda 1–45 Hz, anulación de artefactos por *z-score* MAD
sobre amplitud/canal, y cuatro referencias alineadas en el tiempo. Splits por
**bloques temporales** (test = último 20 %, val = el 20 % previo, resto train)
o por **sujeto**.

**Balanceo multi-configuración:** todas las configuraciones comparten los
mismos índices de split y el mismo presupuesto por split
(`multi_max_samples_per_split`, por defecto 40 000 por configuración → las
mismas proporciones, sin desbalance).

---

## 5. Métricas: qué miden y qué es "bueno o malo"

### RMSE (raíz del error cuadrático medio)
```
RMSE = sqrt( (1/N) Σ (ŷ − y)² )
```
- Misma unidad que la señal (V; se reporta en µV).
- **Referencia (en este dataset):** las amplitudes típicas por canal son del
  orden de 10–100 µV. Un RMSE < 1 µV en transformación cruzada es **excelente**;
  entre 1 y 10 µV es **muy bueno**; > 30 µV en rutas cruzadas es **deficiente**
  (comparable al tamaño de la señal).
- **Ojo:** RMSE de montajes de distinto número de canales NO se compara 1:1;
  compáralo siempre con la línea base analítica de la misma configuración.

### MAE (error absoluto medio)
RMSE sin cuadrar/potenciar. Menos sensible a valores atípicos; suele ser ~2–3
veces menor que el RMSE cuando los errores son gaussianos.

### r (correlación de Pearson, promediada por canal)
```
r = correlación(ŷ_c, y_c) por canal c, luego media
```
- Mide **forma/sincronía**, no magnitud.
- Referencia: r > 0.99 en `projected` (canónico) ≈ **excelente**; 0.95–0.99
  **muy bueno**; 0.90–0.95 **aceptable**; < 0.90 en rutas cruzadas indica que
  la tarea no está bien resuelta. En problemas de referencia, un r > 0.99 es
  típico y esperable porque la tarea es lineal casi exacta.

### VE (varianza explicada)
```
VE = 1 − Σ(ŷ−y)² / Σ(y²)     sobre el subespacio observable (centrado por instante)
```
- 1.0 = reconstrucción perfecta; 0.0 = tan malo como predecir la media; **negativo**
  = peor que predecir el valor constante (el modelo proyecta en la dirección
  equivocada o hay amplificación de ruido).
- Referencia: VE > 0.9 **muy bueno**; 0.5–0.9 **aceptable/útil**; < 0 (y sobre
  todo < −1) indica fallo (p. ej. problema inverso sin regularizar).

### Error relativo de Frobenius (`error_fro_rel`)
Compara la matriz efectiva aprendida con la analítica en el subespacio
observable:
```
||P(A_modelo − A_analitica)P||_F / ||P A_analitica P||_F
```
- < 0.05: el modelo reproduce la física; ~1: memorización o mala convergencia.

### Error de composición (física de grupo)
```
||P(A_{s→d}A_{d→u} − A_{s→u})P||_F / ||P A_{s→u} P||_F
```
- **0 exacto** en `group`; ~1 en el encadenado analítico `T_d pinv(T_s)`
  (compone mal aunque cada ruta sea algebraica). Cuanto más cerca de 0, mejor
  satisface la transitividad de grupo.

### Línea base analítica
Cada configuración se compara contra la conversión **analítica** dentro de ella
(`T_d @ pinv(T_s)`): es el límite superior de lo que una transformación lineal
exacta puede hacer sin aprender. El modelo debe igualarlo (o mejorarlo, pues
`pinv` no es estable en datos ruidosos).

---

## 6. Cómo leer los resultados y cuándo confiar

| Señal | Interpretación |
|---|---|
| Modelo ≈ línea base analítica | El aprendizaje **no aporta**; la conversión es matemática. |
| Modelo > analítico (VE mayor, RMSE menor) | El modelo regulariza mejor que `pinv` (típico en bipolar, mal condicionado). |
| `group` con error de composición ~0 | La física de grupo se satisface exactamente. |
| `multi_montage` con VE alta en *todas* las configuraciones | El autoencoder compartido generaliza entre 19 y 256 electrodos. |
| `multi_heatmap` con `ve_field_cross` alto **y** `rmse_field_cross_uV` bajo | El topomapa predicho es fiel en la malla compartida, no solo electrodo a electrodo. |

Regla práctica: **nunca mires el RMSE sin su línea base analítica ni sin la
amplitud de la señal**; usa VE para calibrar la calidad relativa y `r` para la
forma. En comparativas entre variantes usa siempre el mismo montaje y el mismo
split.

---

## 7. Reproducción y lectura de código

- `docs/mapping_wip.md` — unificación de montajes (métodos y CLI).
- `docs/model_variants.md` — detalle de arquitecturas y física de grupo.
- `docs/results_comparison.md` — resultados comparativos por variante.
- `state_of_art.md` — revisión de literatura (REST, EEG-GAN, etc.).
- `src/eeg_transform/references.py` — matrices de referencia (unipolar,
  bipolar, CAR, REST).
- `src/eeg_transform/leadfield.py` — lead field multicapa y REST.
- `src/eeg_transform/mapping.py` — splines, problema inverso, proyección.
- `src/eeg_transform/experiments/multi.py` — generador multi-configuración.
- `src/eeg_transform/models/` — arquitecturas (incl. `multi_montage.py`,
  `multi_heatmap.py`).
- `config/multi_montage.yaml` — configuración de entrenamiento multi-config.
- `config/multi_heatmap.yaml` — igual + campo de superficie (`surface_loss_weight`).

Para entrenar la variante multi-configuración:

```bash
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli pipeline \
    -c config/multi_montage.yaml --force
PYTHONPATH=src entorno/bin/python -m eeg_transform.cli eval \
    -c config/multi_heatmap.yaml    # también genera metrics_surface_test.csv
```