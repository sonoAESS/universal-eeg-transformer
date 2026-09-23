# Variantes de modelo del Universal EEG Transformer

> **HISTÓRICO.** Este documento describe variantes/fases que ya **no forman
> parte del código** (estas ramas de exploración se cerraron y el núcleo
> canónico se simplificó a la variante `free`). Se conserva como referencia
> conceptual; el estado vigente está en `README.md`,
> `docs/guia_conceptual.md` y `docs/results_comparison.md`.

El transformador es un autoencoder **lineal** multientrada/multisalida:
cada ruta efectiva es la matriz `A_{s→d} = W_enc^s @ W_dec^d` (C×C), actuando
`X_ref = X @ A` sobre señales `(tiempo, canales)`. El espacio latente tiene
dimensión igual al número de canales (`latent_dim: 0` en los YAML).

Se ofrecen tres variantes, seleccionadas con `model.variant`. Todas parten de
la misma **inicialización lineal empírica** (`init_from_data`): regresión
ridge con `z = X_unipolar` como latente, que desde la época 0 deja la
varianza explicada cerca de 1 en las 16 rutas.

| Variante     | Parámetros aprendidos                 | Propiedades físicas impuestas |
|--------------|---------------------------------------|-------------------------------|
| `free`       | 8 matrices `C×C` (enc + dec por montaje) | ninguna adicional              |
| `group`      | 4 matrices `C×C` (enc) + pseudo-inversas fijas | grupo exacto (ver abajo) |
| `projected`  | 8 matrices `C×C` centradas (`W = P W_raw`) | anulación del modo constante |
| `soft_group` | 8 matrices `C×C` + penalización de composición en la pérdida | grupo suave (ver abajo) |

## Proyección de observables

Todas las referencias (unipolar, bipolar, CAR, REST) son singulares:
anulan el modo constante instantáneo (columnas de `T` con suma cero). La
señal observada vive por tanto en el subespacio **observable** de
dimensión `C−1`, y el modo constante no es recuperable desde ninguna
referencia. `P = I − 11ᵀ/C` es el proyector de centrado (sobre el espacio
observable, `P` es la identidad).

La métrica `transfer_error_matrix` compara por eso `P A Pu P` por ambos
lados. De lo contrario el error relativo quedaría ~1 incluso para un modelo
casi exacto.

## Variante `group`: estructura de grupo en el subespacio observable

El decodificador de cada montaje es la **pseudo-inversa** del encoder del
mismo montaje: `W_dec^k = (W_enc^k)^+`. Con esto:

```
A_{s→d}   = W_enc^s (W_enc^d)^+
A_{s→s}   = (W_enc^s)(W_enc^s)^+  =  proyector (≈ identidad observable)
A_{s→d} A_{d→u} = A_{s→u}          (transitividad EXACTA)
```

Los encoders usan además centrado doble (`W = P W_raw P`), de modo que el
*rowspace* y el conúcleo de todas las matrices efectivas viven en el
subespacio observable; así la propiedad se cumple exactamente incluso con
las matrices empíricas de la inicialización.

**Por qué es física relevante:** el encadenado analítico habitual
`T_d pinv(T_s)` **no** es un grupo: la composición
`A_{s→d} A_{d→u}` difiere de `A_{s→u}` con error relativo del orden de 1
(verificado numéricamente). Un modelo con estructura de grupo garantiza
que ir de `s` a `d` y luego a `u` produce exactamente lo mismo que ir
directo de `s` a `u`, una propiedad que toda transformación de referencia
consistentemente definida debería cumplir.

**Coste:** `A_{s→d} = W_s (W_d)^+` supone que todos los montajes comparten el
mismo subespacio observable; el modelo no puede representar un mapa con
rango completo (que no tendría sentido físico de todos modos).

> **Hallazgo empírico (12 sujetos):** esta rigidez es también su talón de
> Aquiles. El rowspace de REST (lead field a infinito) no coincide con el de
> uni/bip/CAR, y al forzar la pseudo-inversa la inicialización empírica ya
> incurre en `error_fro_rel` > 14 en rutas con REST. En test la variante
> `group` no converge (RMSE cross ~133 µV, `r ≈ 0.71`), el mismo fracaso que
> el encadenado analítico `T_d pinv(T_s)`. La consistencia de composición es
> exacta (0.000), pero sobre un subespacio equivocado.

## Variante `soft_group`: grupo suave por regularización

Arquitectura libre (8 matrices) pero la pérdida total suma una
**penalización de composición** cada lote:

```
loss_total = loss_rutas + w · ⟨||P(A_{s→d}A_{d→u} − A_{s→u})P||F / ||P A_{s→u} P||F⟩_{s,d,u}
```

Con `w = 0.5` (config `soft_group.yaml`) el modelo aprende a componer casi
de forma exacta sin la pseudo-inversa rígida: en test `comp_medio = 0.005`
(~14× mejor que `free`, ~120× mejor que el encadenado analítico) con RMSE
cruzado de solo ~0.9 µV — degrada menos de 0.3 µV frente a `free`. Es el
equilibrio recomendado si la transitividad importa.

## Variante `projected`: anulación del modo constante por construcción

Toda matriz aprendida se parametriza como `W_eff = P W_raw`. Como `1ᵀP = 0`,
`1ᵀ W_eff = 0`: una entrada constante sobre canales produce salida nula.
Ninguna ruta puede introducir un DC espurio ni componer "pseudo-referencias"
inválidas. Es la restricción mínima que respeta la física de adquisición en
cada ruta, dejando el resto del aprendizaje libre (8 matrices).

## Inicialización por variante

* `free` / `projected` / `soft_group`: `W_enc^s = ridge(X_s → U)`, `W_dec^d = ridge(U → X_d)`.
* `group`: solo `W_enc^s = ridge(X_s → U)`; los decoders se derivan.

`init_from_data` exige `latent_dim == C`.

## Evaluación de física

Además de las métricas de precisión por ruta (RMSE, correlación), `eval`
guarda `consistency.csv` con la tabla de error de composición
`||P(A_{s→d} A_{d→u} − A_{s→u})P||ₓ / ||P A_{s→u} P||ₓ`:

* `group`: 0 exacto.
* `soft_group`: ~0.005 (penalizado en la pérdida).
* `free`/`projected`: cercano a 0 si la data es suficientemente informativa
  (el modelo aprende a componer bien), sin estar garantizado.
* línea base analítica `T_d pinv(T_s)`: ~0.6 de media (no es un grupo).