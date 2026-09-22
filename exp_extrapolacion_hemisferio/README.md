# `exp_extrapolacion_hemisferio` — cómo interpolar a un hemisferio ciego

Experimento **sintético y autocontenido** para responder, con números, la
pregunta de la exploración `topomap_refs`:

> Si mido una configuración **bipolar solo en un hemisferio** y quiero
> interpolar para "ver" el otro hemisferio usando las diferencias, ¿qué tal
> sale? ¿Mejora si la interpolación se hace **con el lead field** (física del
> problema) en lugar de con la spline esférica (geometría pura)?

## Diseño

* **Verdad de campo**: lead field analítico MCB (`leadfield.py`, MNE) en la
  grilla `standard_1005` (343 nodos del casco). Potenciales sintéticos
  `V = G·q` con dipolos aleatorios restringidos por lateralidad y
  re-referenciados a la media instantánea (CAR).
* **Montajes medidos** (hemisferio izquierdo, `x<0`): `asa10-20` (43 ch),
  `asa10-10` (19 ch) y `dense` (165 ch) — volúmenes realistas de un montaje
  lateralizado.
* **Bipolares**: diferencias a lo largo de las aristas Delaunay locales
  (solo dentro del hemisferio medido).
* **Reconstrucción**, dos familias:
  * **Geométrica** (spline esférica, `mapping.py`):
    * `monopolar` (control): interpolar el potencial medido directamente.
    * `bipolar`: integrar las diferencias (`pinv` + centrado) y luego spline.
  * **Física** (inversión del lead field, `leadfield_projection_matrix`):
    * `leadfield` (monopolar): REST entre montajes `P = W(W G_s)^+G`, con
      truncado del SVD a `n_components`.
    * `leadfield_bip` (diferencias): ajuste de dipolos a los bipolares
      `G_bip = MᵀG_s` (truncado a `n_components`) y campo de casco generado
      por esos dipolos: `V_rec = V_bip @ (U_k s_k⁻¹ Vᵏ Gᵀ)`.
* **Métricas por región** (medida = izquierda+midline, ciego = derecha): VE,
  VE del patrón espacial (sin offset instantáneo), `r` mediana, ratio de
  amplitud RMS y **fisicidad**.

## Resultado central (fuentes mixtas)

| montaje | ruta | VE medida | VE ciego | r ciega | amp ciega |
|---|---|---|---|---|---|
| asa10-20 (43) | spline bipolar | 0.72 | **−1.20** | 0.68 | 1.8× |
| asa10-20 | leadfield_bip (k=21) | 0.88 | **+0.55** | 0.77 | 1.2× |
| asa10-10 (19) | spline bipolar | 0.73 | **−1.03** | 0.68 | 1.9× |
| asa10-10 | leadfield_bip (k=9) | 0.84 | **+0.48** | 0.72 | 1.4× |
| dense (165) | spline bipolar | 0.77 | **−3.14** | 0.74 | 2.3× |
| dense | leadfield_bip (k=82) | 0.91 | **+0.65** | 0.87 | 1.3× |

Incluye el resto de lateralidades y todos los `n_components` en
`metricas.csv`. Las rutas `leadfield` (monopolar) emparejan o superan a
`leadfield_bip` en los montajes densos/10-10 (VE ciega 0.49–0.66).

## Interpretación

### Por qué funciona el lead field (la "realidad de las condiciones de borde")

1. **Entra la física del conductor de volumen**. La spline es una regresión
   geométrica: suaviza en la esfera sin saber cómo se propaga el potencial. El
   lead field fuerza a que la reconstrucción provenga de dipolos corticales y
   que el campo extrapolado al hemisferio ciego sea **el campo físico que esos
   dipolos inducen** allí, no una decoración suave. Esto convierte la
   extrapolación de "continuación geométrica" en "predicción por el forward".

2. **Resultado**: VE ciega pasa de **negativa** (spline, −1.0…−3.1) a
   **positiva** (lead field, +0.48…+0.65), y `r` ciega sube a ~0.72–0.87.
   Además NO degrada el hemisferio medido: VE medida cae menos (0.84–0.91) que
   la spline bipolar (~0.72–0.77), que pierde el gauge/offset.

3. **El truncado SVD es la regularización clave**: el problema inverso
   `W G_s` está mal condicionado (κ~1e15); truncar a `C_s//2` modos estables
   da el mejor balance fidelidad/ruido (figura `07_*`). Con el montaje
   completo (dense) el hemisferio ciego se recupera con VE ~0.65 y `r`~0.87.

### Qué NO dice la fisicidad (hallazgo honesto)

La métrica `fisicidad = 1 − ‖V_rec − G(G⁺V_recᵀ)ᵀ‖²/‖V_rec‖²` mide qué
fracción de la reconstrucción vive en el espacio fuente-realizable `col(G)`.
**Satura en ≈1.0 para TODAS las rutas**: con la grilla de 2006 dipolos el
lead field tiene rango casi completo (339/343) y cualquier campo suave es (casi)
realizable. No es un discriminador — la restricción física útil no es "qué
campos son realizables" (casi todos) sino "qué campo *predice el forward* dado
lo medido", que captura la VE ciega de las rutas lead field.

### La spline queda vetada para cruzar zonas sin sensores

El experimento original ya lo demostraba (sección anterior): extrapolar un
hemisferio completo con la spline produce VE negativa y amplitudes infladas
1.3–3×. La solución **física** lo corrige: si se dispone de un modelo
anatómico del casco, la interpolación-extrapolación debe hacerse por inversión
del lead field, no por spline. Para el problema gestor (montaje + referencia),
esto ratifica que el mapa "interior" (hacia nodos cubiertos) lo hacen bien los
notebooks de `topomap_refs`; el salto a zonas no medidas requiere volver al
forward.

## Ficheros

* `run.py` — experimento (siembra, escenas, reconstrucción, métricas, figuras).
* `runs/exp_extrapolacion_hemisferio/metricas.csv` — tabla completa
  (order × montaje × lateralidad × ruta × n_components).
* `runs/exp_extrapolacion_hemisferio/metricas.json` — resumen compacto.
* `runs/exp_extrapolacion_hemisferio/resumen_leadfield_bip_mixto.csv` — mejor
  `n_components` por montaje (leadfield_bip, fuentes mixtas).
* `runs/exp_extrapolacion_hemisferio/figs/` — figuras (panorama spline,
  comparativa spline-vs-leadfield, VE medida vs ciega, efecto n_components,
  VE por orden/lateralidad/distancia).