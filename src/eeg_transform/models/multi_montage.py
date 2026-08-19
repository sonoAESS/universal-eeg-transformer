"""Autoencoder universal sobre múltiples configuraciones de electrodos.

La variante ``multi_montage`` entrena un autoencoder lineal canónico (64
canales) compartido entre distintas configuraciones de electrodos. Cada
configuración se embebe con una matriz fija ``P_s`` (C_s -> 64) y su salida se
lee con ``Q_s`` (64 -> C_s), de modo que:

.. math::

    \\hat x^{(b)}_s = Q_s\\, W^{dec}_b\\, W^{enc}_a\\, P_s\\, x^{(a)}_s

es decir, el modelo aprende a **convertir referencias dentro de la propia
configuración**: dado ``x`` de ``C_s`` electrodos en una referencia cualquiera,
predice las restantes referencias sobre los mismos ``C_s`` electrodos. Los
coeficientes ``W_enc``/``W_dec`` son las mismas variables para todas las
configuraciones, por lo que la capacidad es compartida y el dato queda
balanceado por construcción (mismas muestras por configuración).

Las matrices ``P_s``/``Q_s`` son fijas (interpolación esférica / selección y
la proyección inversa opcional), de la misma familia que ``mapping``/``montage``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses

from ..config import REFERENCE_KINDS
from .universal_transformer import UniversalEEGTransformer

KINDS = list(REFERENCE_KINDS)


class MultiMontageAutoencoder(tf.keras.Model):
    """Envoltorio del autoencoder canónico con mapas fijos por configuración.

    Parameters
    ----------
    core:
        Instancia de :class:`UniversalEEGTransformer` en modo canónico (64
        canales, sin ``projection``); sus matrices ``W_enc``/``W_dec`` se
        comparten entre todas las configuraciones.
    projections:
        ``{label: matriz (C_s, C)}`` de entrada de cada configuración.
    out_maps:
        ``{label: matriz (C, C_s)}`` de salida de cada configuración.
    """

    def __init__(
        self,
        core: UniversalEEGTransformer,
        projections: Dict[str, np.ndarray],
        out_maps: Dict[str, np.ndarray],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core = core
        self.kinds = list(REFERENCE_KINDS)
        self.configs = list(projections.keys())
        self.n_canonical = core.n_channels
        self.projections = {
            label: tf.constant(np.asarray(m, dtype=np.float32))
            for label, m in projections.items()
        }
        self.out_maps = {
            label: tf.constant(np.asarray(m, dtype=np.float32))
            for label, m in out_maps.items()
        }
        self.loss_tracker = tf.keras.metrics.Mean(name="loss_estandarizada")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_real_V2")
        self.built = True

    # ------------------------------------------------------------------
    # Cálculo por configuración
    # ------------------------------------------------------------------
    def _predict_cfg(self, x: tf.Tensor, cfg: str, source: str) -> Dict[str, tf.Tensor]:
        """Para una configuración y origen: dict {destino: (n, C_s)}."""
        u = tf.matmul(tf.cast(x, tf.float32), self.projections[cfg])  # (n, C)
        pred = self.core._predict_from(u, source)                     # {d: (n, C)}
        return {d: tf.matmul(pred[d], self.out_maps[cfg]) for d in self.kinds}

    def call(self, inputs, cfg: str | None = None, source: str | None = None, **kwargs):
        if cfg is None or source is None:
            out = {}
            for c in self.configs:
                c_in = inputs[c] if isinstance(inputs, dict) else inputs
                out[c] = {s: self._predict_cfg(c_in[s], c, s) for s in self.kinds}
            return out
        return self._predict_cfg(inputs, cfg, source)

    # ------------------------------------------------------------------
    # Métricas y bucles de entrenamiento/validación
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return UniversalEEGTransformer._zscore_loss(y_true, y_pred)

    def _config_losses(self, sources, targets) -> tf.Tensor:
        """Suma de pérdidas de ruta estandarizadas sobre todas las configs."""
        terms = []
        for c in self.configs:
            for s in self.kinds:
                preds = self._predict_cfg(sources[c][s], c, s)
                for d in self.kinds:
                    terms.append(self._zscore_loss(targets[c][d], preds[d]))
        n_terms = max(1, len(terms))
        return tf.add_n(terms) / n_terms if terms else tf.zeros(())

    def _real_mse(self, sources, targets) -> tf.Tensor:
        total, count = 0.0, 0
        for c in self.configs:
            for s in self.kinds:
                preds = self._predict_cfg(sources[c][s], c, s)
                for d in self.kinds:
                    total += losses.MSE(targets[c][d], preds[d])
                    count += 1
        return total / max(1, count)

    def train_step(self, data):
        sources, targets = data
        with tf.GradientTape() as tape:
            loss = self._config_losses(sources, targets)
            if self.losses:
                loss = loss + tf.add_n(self.losses)  # regularizaciones L2

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        sources, targets = data
        loss = self._config_losses(sources, targets)
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    # ------------------------------------------------------------------
    # Matrices efectivas por configuración (rendimiento físico)
    # ------------------------------------------------------------------
    def transfer_matrices(self, cfg: str) -> Dict[Tuple[str, str], np.ndarray]:
        """Matrices efectivas de la configuración: ``P_s A_{s->d} Q_s`` (C_s x C_s).

        Con vectores fila (observación ``x (1, C_s)``): ``x ↦ x·P_s ↦ (x·P_s)·A ↦
        (x·P_s·A)·Q_s``, luego ``A^{cfg}_{s->d} = P_s·A_{s->d}·Q_s``.
        """
        core_mats = self.core.transfer_matrices()   # (C, C)
        p = self.projections[cfg].numpy().astype(np.float64)    # (C_s, C)
        q = self.out_maps[cfg].numpy().astype(np.float64)       # (C, C_s)
        return {
            (s, d): (p @ core_mats[(s, d)] @ q) for (s, d), m in core_mats.items()
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "configs": self.configs,
            "n_canonical": self.n_canonical,
            "projections": {
                l: self.projections[l].numpy().tolist() for l in self.configs
            },
            "out_maps": {l: self.out_maps[l].numpy().tolist() for l in self.configs},
            "core_cfg": self.core.get_config(),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        core = UniversalEEGTransformer.from_config(config["core_cfg"])
        projections = {
            l: np.asarray(m, dtype=np.float32) for l, m in config["projections"].items()
        }
        out_maps = {
            l: np.asarray(m, dtype=np.float32) for l, m in config["out_maps"].items()
        }
        return cls(core=core, projections=projections, out_maps=out_maps)