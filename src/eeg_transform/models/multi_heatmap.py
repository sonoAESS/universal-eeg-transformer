"""Autoencoder universal multi-configuración con campo de superficie (heatmap).

La variante ``multi_heatmap`` entrena un autoencoder lineal canónico (64
canales) compartido entre varias configuraciones, y además de predecir las
referencias intra-configuración ``C_s -> C_s`` entrena explícitamente la
lectura de la actividad como **campo de superficie sobre la malla compartida**
del cuero cabelludo (los "heatmaps" de las figuras topomapa).

Esta clase es la base de ``multi_heatmap`` y de ``multi_heatmap_v2``. La v2
añade, sin romper el comportamiento de la v1 (pesos en 0 / interpolación fija):

* **Interpolación aprendible (A2):** la matriz de campo ``R_s`` (C_s -> malla)
  es entrenable, inicializada en la spline fija ``S_s`` y regularizada hacia
  ella. El campo predicho deja de ser un mero espejo suavizado de ``x̂``.
* **Bucle electrodo↔campo (A1):** se exige ``x̂ ≈ R_sᵀ·F̂ = (R_sᵀR_s)·x̂``, de
  modo que el campo sea una representación fiel e invertible del electrodo.
* **Consistencia entre configuraciones (B4):** los campos predichos de todas
  las configuraciones (misma malla) deben coincidir; esto alinea 10-20,
  canónico y densos sin señal externa.
* **Adaptadores por configuración (B5):** un residual low-rank en el espacio
  latente canónico da capacidad local sin tocar el core compartido.
* **Suavizado temporal ligero (C7):** regularización de variación total entre
  muestras consecutivas (requiere dataset ordenado en tiempo).
* **Ponderación por incertidumbre (A3):** los pesos de los términos se
  aprenden como ``log σ²`` (multi-task), auto-balanceando todas las pérdidas.

La pérdida base (MSE estandarizado Z-score por electrodo y por campo) es
idéntica en espíritu a la de ``multi_montage``; los nuevos términos se suman
con los pesos correspondientes (o con la formulación de incertidumbre).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..config import REFERENCE_KINDS
from .multi_montage import MultiMontageAutoencoder

KINDS = list(REFERENCE_KINDS)

# Referencia usada como representación de campo para la consistencia cross-config
# (todas las referencias son transformaciones lineales del mismo potencial).
DEFAULT_FIELD_KIND = "unipolar"


class _V2Params(tf.keras.layers.Layer):
    """Capa auxiliar que agrupa las Variables entrenables de v2.

    Keras 3 no registra de forma fiable las ``tf.Variable`` guardadas en
    diccionarios/``setattr``, así que se alojan como pesos de una capa hija
    (``add_weight``) que el modelo padre sí trackea.
    """

    def __init__(self, configs, screen, learnable_interp, adapter_rank,
                 latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.configs = list(configs)
        self.learnable_interp = bool(learnable_interp)
        self.adapter_rank = int(adapter_rank)
        self.latent_dim = int(latent_dim)
        self.R_s = {}
        for label, s in screen.items():
            self.R_s[label] = self.add_weight(
                name=f"R_s_{label}", shape=tuple(s.shape),
                initializer=tf.constant_initializer(s.astype(np.float32)),
                trainable=self.learnable_interp,
            )
        self.adapters = {}
        if self.adapter_rank > 0:
            for label in self.configs:
                u = self.add_weight(
                    name=f"adapter_U_{label}",
                    shape=(self.latent_dim, self.adapter_rank),
                    initializer="zeros", trainable=True)
                v = self.add_weight(
                    name=f"adapter_V_{label}",
                    shape=(self.adapter_rank, self.latent_dim),
                    initializer="zeros", trainable=True)
                self.adapters[label] = (u, v)



class MultiHeatmapAutoencoder(MultiMontageAutoencoder):
    """Autoencoder multi-configuración con pérdida de campo de superficie.

    Parameters
    ----------
    surfaces:
        ``{label: matriz (n_grid, C_s)}``: interpolación electrodos -> malla
        compartida del cuero cabelludo por configuración (spline fija ``S_s``).
    surface_loss_weight:
        Peso relativo del término de superficie frente al MSE por electrodo
        (usado cuando ``learn_uncertainty`` es falso).
    learnable_interp:
        Si es verdadero, la matriz de campo ``R_s`` es entrenable (A2);
        arranca en ``S_s`` y se regulariza hacia ella.
    field_consistency_weight:
        Peso de la consistencia campo↔electrodo (A1): ``x̂ ≈ R_sᵀ·F̂``.
    xconfig_consistency_weight:
        Peso de la consistencia entre campos de distintas configuraciones (B4).
    adapter_rank:
        Rango del adaptador low-rank por configuración (B5); 0 = desactivado.
    temporal_smoothness_weight:
        Peso de la regularización de variación total temporal (C7).
    learn_uncertainty:
        Si es verdadero, todos los pesos anteriores se reemplazan por
        variables ``log σ²`` aprendidas (A3) y la pérdida combina como
        ``Σ exp(-logσ²)·L + logσ²``.
    field_kind:
        Referencia usada como campo para la consistencia cross-config.
    """

    def __init__(
        self,
        core,
        projections: Dict[str, np.ndarray],
        out_maps: Dict[str, np.ndarray],
        surfaces: Dict[str, np.ndarray],
        surface_loss_weight: float = 0.1,
        learnable_interp: bool = False,
        field_consistency_weight: float = 0.0,
        xconfig_consistency_weight: float = 0.0,
        adapter_rank: int = 0,
        temporal_smoothness_weight: float = 0.0,
        learn_uncertainty: bool = False,
        field_kind: str = DEFAULT_FIELD_KIND,
        **kwargs,
    ):
        super().__init__(core=core, projections=projections, out_maps=out_maps,
                         **kwargs)
        self.screen = {label: np.asarray(s, dtype=np.float32)
                       for label, s in surfaces.items()}
        self.surfaces = {
            label: tf.constant(np.asarray(s, dtype=np.float32))
            for label, s in self.screen.items()
        }
        self.learnable_interp = bool(learnable_interp)
        # Capa de parámetros entrenables de v2 (R_s aprendible y adaptadores).
        # Los adaptadores actúan en el espacio de salida canónico (n_canonical),
        # idéntico a la salida del decoder y coherente con Q_s.
        self.params = _V2Params(
            self.configs, self.screen, self.learnable_interp,
            int(adapter_rank), self.n_canonical,
        )
        self.adapter_rank = int(adapter_rank)

        self.surface_loss_weight = float(surface_loss_weight)
        self.field_consistency_weight = float(field_consistency_weight)
        self.xconfig_consistency_weight = float(xconfig_consistency_weight)
        self.temporal_smoothness_weight = float(temporal_smoothness_weight)
        self.learn_uncertainty = bool(learn_uncertainty)
        self.field_kind = field_kind if field_kind in KINDS else DEFAULT_FIELD_KIND

        self.surface_tracker = tf.keras.metrics.Mean(name="surface_loss")
        self.field_consist_tracker = tf.keras.metrics.Mean(name="field_consist_loss")
        self.xconfig_tracker = tf.keras.metrics.Mean(name="xconfig_loss")
        self.temporal_tracker = tf.keras.metrics.Mean(name="temporal_loss")

        if self.learn_uncertainty:
            self.log_vars = {
                "elec": tf.Variable(0.0, trainable=True, dtype=tf.float32,
                                     name="logvar_elec"),
                "surf": tf.Variable(0.0, trainable=True, dtype=tf.float32,
                                     name="logvar_surf"),
                "field": tf.Variable(0.0, trainable=True, dtype=tf.float32,
                                     name="logvar_field"),
                "xconfig": tf.Variable(0.0, trainable=True, dtype=tf.float32,
                                       name="logvar_xconfig"),
                "temp": tf.Variable(0.0, trainable=True, dtype=tf.float32,
                                    name="logvar_temp"),
            }

    # ------------------------------------------------------------------
    # Cálculo por configuración (con adaptador opcional)
    # ------------------------------------------------------------------
    def _predict_cfg(self, x: tf.Tensor, cfg: str, source: str) -> Dict[str, tf.Tensor]:
        """Para una configuración y origen: dict {destino: (n, C_s)}."""
        u = tf.matmul(tf.cast(x, tf.float32), self.projections[cfg])  # (n, C)
        z = self.core.encode(u, source)
        preds = {d: self.core.decode(z, d) for d in self.kinds}
        if self.adapter_rank > 0:
            u_mat, v_mat = self.params.adapters[cfg]
            preds = {d: tf.matmul(preds[d], u_mat) @ v_mat for d in self.kinds}
        return {d: tf.matmul(preds[d], self.out_maps[cfg]) for d in self.kinds}

    def _predict_all(self, sources) -> Dict[str, Dict[str, Dict[str, tf.Tensor]]]:
        out = {}
        for c in self.configs:
            out[c] = {
                s: self._predict_cfg(sources[c][s], c, s) for s in self.kinds
            }
        return out

    def _field_matrix(self, cfg: str) -> tf.Tensor:
        """Matriz de campo efectiva (aprendible o fija) de la configuración."""
        return self.params.R_s[cfg]

    # ------------------------------------------------------------------
    # Términos de pérdida
    # ------------------------------------------------------------------
    def _electrode_loss_from(self, preds_all, targets) -> tf.Tensor:
        terms = []
        for c in self.configs:
            for s in self.kinds:
                for d in self.kinds:
                    terms.append(self._zscore_loss(targets[c][d], preds_all[c][s][d]))
        n = max(1, len(terms))
        return tf.add_n(terms) / n if terms else tf.zeros(())

    def _surface_loss_from(self, preds_all, targets) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Pérdida estandarizada sobre el campo; devuelve (loss, campos_pred)."""
        terms = []
        fields: Dict[str, tf.Tensor] = {}
        for c in self.configs:
            r = self._field_matrix(c)
            f_true_base = tf.matmul(targets[c]["unipolar"], self.surfaces[c], transpose_b=True)
            fields[c] = tf.matmul(preds_all[c][self.field_kind][self.field_kind], r, transpose_b=True)
            for s in self.kinds:
                for d in self.kinds:
                    f_pred = tf.matmul(preds_all[c][s][d], r, transpose_b=True)
                    terms.append(self._zscore_loss(f_true_base, f_pred))
        n = max(1, len(terms))
        return (tf.add_n(terms) / n if terms else tf.zeros(())), fields

    def _field_consistency_loss(self, preds_all) -> tf.Tensor:
        """Consistencia electrodo↔campo: x̂ ≈ R_sᵀ·F̂ = (R_sᵀ R_s)·x̂ (A1)."""
        terms = []
        for c in self.configs:
            r = self._field_matrix(c)
            rt_r = tf.matmul(r, r, transpose_a=True)  # (C_s, C_s)
            for s in self.kinds:
                for d in self.kinds:
                    x = preds_all[c][s][d]
                    recon = tf.matmul(x, rt_r)
                    terms.append(self._zscore_loss(x, recon))
        n = max(1, len(terms))
        return tf.add_n(terms) / n if terms else tf.zeros(())

    def _xconfig_field_loss(self, fields: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Consistencia entre campos de distintas configuraciones (B4)."""
        labels = list(fields.keys())
        terms = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                terms.append(self._zscore_loss(fields[labels[i]], fields[labels[j]]))
        n = max(1, len(terms))
        return tf.add_n(terms) / n if terms else tf.zeros(())

    def _temporal_loss(self, preds_all) -> tf.Tensor:
        """Variación total entre muestras consecutivas (C7)."""
        terms = []
        for c in self.configs:
            for s in self.kinds:
                for d in self.kinds:
                    p = preds_all[c][s][d]
                    if p.shape[0] is not None and p.shape[0] > 1:
                        diff = p[1:] - p[:-1]
                        terms.append(tf.reduce_mean(tf.square(diff)))
        n = max(1, len(terms))
        return tf.add_n(terms) / n if terms else tf.zeros(())

    def _combine_losses(self, base, surf, field_consist, xconfig, temporal):
        if self.learn_uncertainty:
            lv = self.log_vars
            parts = {
                "elec": (base, lv["elec"]),
                "surf": (surf, lv["surf"]),
                "field": (field_consist, lv["field"]),
                "xconfig": (xconfig, lv["xconfig"]),
                "temp": (temporal, lv["temp"]),
            }
            total = tf.zeros((), dtype=tf.float32)
            for _, (lval, logvar) in parts.items():
                total = total + tf.exp(-logvar) * lval + logvar
            return total
        total = base
        total = total + self.surface_loss_weight * surf
        total = total + self.field_consistency_weight * field_consist
        total = total + self.xconfig_consistency_weight * xconfig
        total = total + self.temporal_smoothness_weight * temporal
        return total

    # ------------------------------------------------------------------
    # Bucles de entrenamiento/validación
    # ------------------------------------------------------------------
    def _forward_losses(self, sources, targets):
        preds_all = self._predict_all(sources)
        base = self._electrode_loss_from(preds_all, targets)
        surf, fields = self._surface_loss_from(preds_all, targets)
        field_consist = self._field_consistency_loss(preds_all)
        xconfig = self._xconfig_field_loss(fields)
        temporal = self._temporal_loss(preds_all)
        return base, surf, field_consist, xconfig, temporal, preds_all

    def train_step(self, data):
        sources, targets = data
        with tf.GradientTape() as tape:
            base, surf, field_consist, xconfig, temporal, _ = self._forward_losses(sources, targets)
            loss = self._combine_losses(base, surf, field_consist, xconfig, temporal)
            if self.losses:
                loss = loss + tf.add_n(self.losses)  # regularizaciones L2

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        self.field_consist_tracker.update_state(field_consist)
        self.xconfig_tracker.update_state(xconfig)
        self.temporal_tracker.update_state(temporal)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "surface_loss": self.surface_tracker.result(),
            "field_consist_loss": self.field_consist_tracker.result(),
            "xconfig_loss": self.xconfig_tracker.result(),
            "temporal_loss": self.temporal_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        sources, targets = data
        base, surf, field_consist, xconfig, temporal, _ = self._forward_losses(sources, targets)
        loss = self._combine_losses(base, surf, field_consist, xconfig, temporal)
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        self.field_consist_tracker.update_state(field_consist)
        self.xconfig_tracker.update_state(xconfig)
        self.temporal_tracker.update_state(temporal)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "surface_loss": self.surface_tracker.result(),
            "field_consist_loss": self.field_consist_tracker.result(),
            "xconfig_loss": self.xconfig_tracker.result(),
            "temporal_loss": self.temporal_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "surfaces": {l: self.screen[l].tolist() for l in self.configs},
            "surface_loss_weight": self.surface_loss_weight,
            "learnable_interp": self.learnable_interp,
            "field_consistency_weight": self.field_consistency_weight,
            "xconfig_consistency_weight": self.xconfig_consistency_weight,
            "adapter_rank": self.adapter_rank,
            "temporal_smoothness_weight": self.temporal_smoothness_weight,
            "learn_uncertainty": self.learn_uncertainty,
            "field_kind": self.field_kind,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from .universal_transformer import UniversalEEGTransformer

        core = UniversalEEGTransformer.from_config(config["core_cfg"])
        projections = {
            l: np.asarray(m, dtype=np.float32) for l, m in config["projections"].items()
        }
        out_maps = {
            l: np.asarray(m, dtype=np.float32) for l, m in config["out_maps"].items()
        }
        surfaces = {
            l: np.asarray(m, dtype=np.float32) for l, m in config["surfaces"].items()
        }
        return cls(
            core=core,
            projections=projections,
            out_maps=out_maps,
            surfaces=surfaces,
            surface_loss_weight=config.get("surface_loss_weight", 0.1),
            learnable_interp=config.get("learnable_interp", False),
            field_consistency_weight=config.get("field_consistency_weight", 0.0),
            xconfig_consistency_weight=config.get("xconfig_consistency_weight", 0.0),
            adapter_rank=config.get("adapter_rank", 0),
            temporal_smoothness_weight=config.get("temporal_smoothness_weight", 0.0),
            learn_uncertainty=config.get("learn_uncertainty", False),
            field_kind=config.get("field_kind", DEFAULT_FIELD_KIND),
        )


class TemporalResidualHead(tf.keras.layers.Layer):
    """Cabeza ligera que predice el residuo dinámico.

    Opera sobre la ventana temporal ``(T_w, C)`` en el espacio canónico:

    * **``cell="conv"``** (por defecto): bloques depthwise+pointwise con
      conexión residual sobre la ventana centrada ``padding="same"``; captura
      patrones locales acausales.
    * **``cell`` recurrente** (``"gru"``/``"lstm"``/``"rnn"``): una sola capa
      recurrente causal tras ``in_proj``; el residuo en ``t`` depende solo de
      lo medido hasta ``t`` (ventana de medida -> predicción) y puede
      generalizar a secuencias más largas que la ventana de entrenamiento.

    Las cabezas por destino son proyecciones ``1×1`` inicializadas a **cero**,
    de modo que con pesos iniciales la salida es exactamente la del modelo
    instantáneo lineal (ablation trivial).
    """

    def __init__(self, kinds, channels=64, num_layers=2, kernel_size=7,
                 cell="conv", **kwargs):
        super().__init__(**kwargs)
        self.kinds = list(kinds)
        self.channels = int(channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.cell = cell
        self.blocks = []
        self.rnn = None
        if cell == "conv":
            for i in range(self.num_layers):
                self.blocks.append([
                    tf.keras.layers.DepthwiseConv1D(
                        self.kernel_size, padding="same", name=f"tdw{i}"),
                    tf.keras.layers.Conv1D(self.channels, 1, name=f"tpw{i}"),
                    tf.keras.layers.Activation("gelu"),
                ])
        else:
            rnn_kinds = {
                "gru": tf.keras.layers.GRU,
                "lstm": tf.keras.layers.LSTM,
                "rnn": tf.keras.layers.SimpleRNN,
            }
            self.rnn = rnn_kinds[cell](
                self.channels, return_sequences=True, name="trnn0",
            )
        # proyección de entrada a `channels` (pointwise)
        self.in_proj = tf.keras.layers.Conv1D(self.channels, 1, name="tin")
        # las cabezas por destino se crean en build() (necesitan units=C)
        self.heads = {}

    def build(self, input_shape):
        c = int(input_shape[-1])
        for k in self.kinds:
            self.heads[k] = tf.keras.layers.Conv1D(
                c, 1, name=f"thead_{k}",
                kernel_initializer="zeros", bias_initializer="zeros",
            )
        super().build(input_shape)

    def call(self, u):                      # u: (..., T, C)
        h = self.in_proj(u)
        if self.cell == "conv":
            for dw, pw, act in self.blocks:
                h = h + act(pw(dw(h)))
        else:
            h = self.rnn(h)
        return {k: head(h) for k, head in self.heads.items()}

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"kinds": self.kinds, "channels": self.channels,
                    "num_layers": self.num_layers,
                    "kernel_size": self.kernel_size,
                    "cell": self.cell})
        return cfg


def mode_projector_matrix(positions: np.ndarray) -> np.ndarray:
    """Proyector que anula los modos espaciales l<=1 del casco.

    Construye ``P = I - G(GᵀG)⁻¹Gᵀ`` con ``G = [1, x, y, z]`` sobre las
    posiciones unitarias de los electrodos. Aplicado por columnas
    (``pred @ P``) deja fuera todo componente común y gradiente lineal: la
    condición física que cumplen bipolar/linked (l=0) y el Laplaciano
    (l<=1). Sirve como penalización de realismo para las salidas del modelo.
    """
    pos = np.asarray(positions, dtype=np.float64)
    pos = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    g = np.hstack([np.ones((len(pos), 1)), pos])
    p = np.eye(len(pos)) - g @ np.linalg.solve(g.T @ g, g.T)
    return p.astype(np.float32)


class MultiHeatmapTemporal(MultiHeatmapAutoencoder):
    """universal_refs: núcleo lineal instantáneo + cabeza temporal de residuo.

    Sobre :class:`MultiHeatmapAutoencoder` añade:

    * **Cabeza dinámica** (:class:`TemporalResidualHead`) en el espacio
      canónico: consume la ventana centrada ``(T_w, C)`` de la señal
      proyectada ``u = x·P_s`` y corrige cada destino antes de ``Q_s``.
      Con capa final a cero el arranque es idéntico al modelo lineal.
    * **Penalización de modos espaciales**: las salidas diferenciales
      (bipolar, linked, laplacian) se penalizan si contienen componentes
      comunes o gradientes lineales (proyector ``l<=1`` por configuración).

    Las entradas deben ser ventanas ``(batch, T_w, C_s)`` cuando
    ``temporal_window > 0``; con tensores 2-D el residuo se desactiva y el
    forward coincide con ``multi_heatmap_v2``.
    """

    def __init__(
        self,
        core,
        projections,
        out_maps,
        surfaces,
        mode_projectors: Dict[str, np.ndarray] | None = None,
        temporal_window: int = 0,
        temporal_stride: int = 1,
        temporal_channels: int = 64,
        temporal_layers: int = 2,
        temporal_kernel: int = 7,
        temporal_cell: str = "conv",
        temporal_residual_weight: float = 1.0,
        mode_penalty_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(core=core, projections=projections,
                         out_maps=out_maps, surfaces=surfaces, **kwargs)
        self.temporal_window = int(temporal_window)
        self.temporal_stride = max(1, int(temporal_stride))
        self.temporal_cell = str(temporal_cell)
        self.temporal_residual_weight = float(temporal_residual_weight)
        self.mode_penalty_weight = float(mode_penalty_weight)

        self.head = TemporalResidualHead(
            kinds=self.kinds, channels=temporal_channels,
            num_layers=temporal_layers, kernel_size=temporal_kernel,
            cell=self.temporal_cell, name="temporal_head",
        )
        if self.n_canonical:
            self.head.build(tf.TensorShape([None, None, self.n_canonical]))

        self.mode_projectors = {
            label: tf.constant(np.asarray(mat, dtype=np.float32))
            for label, mat in (mode_projectors or {}).items()
        }

        self.residual_tracker = tf.keras.metrics.Mean(name="residual_norm")
        self.mode_tracker = tf.keras.metrics.Mean(name="mode_loss")

    # ------------------------------------------------------------------
    def _predict_cfg(self, x, cfg, source):
        u = tf.matmul(tf.cast(x, tf.float32), self.projections[cfg])
        z = self.core.encode(u, source)
        preds = {d: self.core.decode(z, d) for d in self.kinds}
        if self.adapter_rank > 0:
            u_mat, v_mat = self.params.adapters[cfg]
            preds = {d: tf.matmul(preds[d], u_mat) @ v_mat for d in self.kinds}
        base = {d: tf.matmul(preds[d], self.out_maps[cfg]) for d in self.kinds}
        if self.temporal_window > 0 and u.shape.rank == 3:
            delta = self.head(u)                       # {d: (n,T,C)}
            q = self.out_maps[cfg]
            base = {
                d: base[d] + self.temporal_residual_weight
                * tf.matmul(delta[d], q)
                for d in self.kinds
            }
            self.residual_tracker.update_state(
                tf.add_n([tf.reduce_mean(tf.square(v)) for v in delta.values()])
                / len(self.kinds)
            )
        return base

    def _mode_penalty_from(self, preds_all):
        terms = []
        for c in self.configs:
            if c not in self.mode_projectors:
                continue
            p_mat = self.mode_projectors[c]
            for s in self.kinds:
                for d in ("bipolar", "linked_mastoids", "linked_ears",
                          "laplacian"):
                    x = preds_all[c][s][d]
                    proj = tf.matmul(x, p_mat)
                    terms.append(self._zscore_loss(proj,
                                                   tf.zeros_like(proj)))
        return tf.add_n(terms) / len(terms) if terms else tf.zeros(())

    def train_step(self, data):
        sources, targets = data
        with tf.GradientTape() as tape:
            base, surf, fc, xc, temp, preds_all = self._forward_losses(sources, targets)
            mode = self._mode_penalty_from(preds_all)
            loss = self._combine_losses(base, surf, fc, xc, temp)
            loss = loss + self.mode_penalty_weight * mode
            if self.losses:
                loss = loss + tf.add_n(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        self.field_consist_tracker.update_state(fc)
        self.xconfig_tracker.update_state(xc)
        self.temporal_tracker.update_state(temp)
        self.mode_tracker.update_state(mode)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "surface_loss": self.surface_tracker.result(),
            "field_consist_loss": self.field_consist_tracker.result(),
            "xconfig_loss": self.xconfig_tracker.result(),
            "temporal_loss": self.temporal_tracker.result(),
            "mode_loss": self.mode_tracker.result(),
            "residual_norm": self.residual_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        sources, targets = data
        base, surf, fc, xc, temp, preds_all = self._forward_losses(sources, targets)
        mode = self._mode_penalty_from(preds_all)
        loss = self._combine_losses(base, surf, fc, xc, temp)
        loss = loss + self.mode_penalty_weight * mode
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        self.field_consist_tracker.update_state(fc)
        self.xconfig_tracker.update_state(xc)
        self.mode_tracker.update_state(mode)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mode_loss": self.mode_tracker.result(),
            "residual_norm": self.residual_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    # ------------------------------------------------------------------
    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "mode_projectors": {
                l: m.numpy().tolist() for l, m in self.mode_projectors.items()
            },
            "temporal_window": self.temporal_window,
            "temporal_stride": self.temporal_stride,
            "temporal_cell": self.temporal_cell,
            "temporal_residual_weight": self.temporal_residual_weight,
            "mode_penalty_weight": self.mode_penalty_weight,
            "head_cfg": self.head.get_config(),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from .universal_transformer import UniversalEEGTransformer

        core = UniversalEEGTransformer.from_config(config["core_cfg"])
        model = cls(
            core=core,
            projections={l: np.asarray(m, np.float32)
                         for l, m in config["projections"].items()},
            out_maps={l: np.asarray(m, np.float32)
                      for l, m in config["out_maps"].items()},
            surfaces={l: np.asarray(m, np.float32)
                      for l, m in config["surfaces"].items()},
            mode_projectors={l: np.asarray(m, np.float32)
                             for l, m in config.get("mode_projectors", {}).items()},
            surface_loss_weight=config.get("surface_loss_weight", 0.1),
            learnable_interp=config.get("learnable_interp", False),
            field_consistency_weight=config.get("field_consistency_weight", 0.0),
            xconfig_consistency_weight=config.get("xconfig_consistency_weight", 0.0),
            adapter_rank=config.get("adapter_rank", 0),
            temporal_smoothness_weight=config.get("temporal_smoothness_weight", 0.0),
            learn_uncertainty=config.get("learn_uncertainty", False),
            field_kind=config.get("field_kind", DEFAULT_FIELD_KIND),
            temporal_window=config.get("temporal_window", 0),
            temporal_stride=config.get("temporal_stride", 1),
            temporal_cell=config.get("temporal_cell", "conv"),
            temporal_residual_weight=config.get("temporal_residual_weight", 1.0),
            mode_penalty_weight=config.get("mode_penalty_weight", 0.0),
        )
        head_cfg = config.get("head_cfg") or {}
        model.head.channels = int(head_cfg.get("channels", 64))
        model.head.num_layers = int(head_cfg.get("num_layers", 2))
        model.head.kernel_size = int(head_cfg.get("kernel_size", 7))
        return model
