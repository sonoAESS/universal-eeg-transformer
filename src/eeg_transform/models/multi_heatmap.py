"""Autoencoder universal multi-configuración con campo de superficie (heatmap).

La variante ``multi_heatmap`` extiende ``multi_montage``: además de predecir
las referencias intra-configuración ``C_s -> C_s``, entrena explícitamente la
lectura de la actividad como **campo de superficie sobre la malla compartida**
del cuero cabelludo (los "heatmaps" de las figuras topomapa). Para cada
configuración ``s`` el campo se obtiene con la matriz fija

.. math::

    \\hat F^{(d)}_s = \\hat x^{(d)}_s\\, S_s^\\top
    \\qquad S_s: (C_s \\to n_{grid})

donde ``S_s`` es la misma interpolación esférica a la malla ``grid_px × grid_px``
que usan ``scalp_grid_matrix``/``scalp_heatmap_fig`` de la fase anterior. El
término de pérdida es el MSE estandarizado (idéntico en espíritu a la pérdida
por electrodo) evaluado sobre el **patrón espacial suave** en la malla, con
peso ``surface_loss_weight`` en la configuración. Al estar ``S_s`` fijo y ser
independiente de la referencia, todos los campos viven en la misma malla y la
lectura es comparable entre configuraciones (19/64/128/256 canales).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from ..config import REFERENCE_KINDS
from .multi_montage import MultiMontageAutoencoder

KINDS = list(REFERENCE_KINDS)


class MultiHeatmapAutoencoder(MultiMontageAutoencoder):
    """Autoencoder multi-configuración con pérdida de campo de superficie.

    Parameters
    ----------
    surfaces:
        ``{label: matriz (n_grid, C_s)}``: interpolación electrodos -> malla
        compartida del cuero cabelludo por configuración.
    surface_loss_weight:
        Peso relativo del término de superficie frente al MSE por electrodo.
        El término está en la misma escala normalizada que el base (0.1 por
        defecto: la superficie ancla el patrón espacial sin dominar).
    """

    def __init__(
        self,
        core,
        projections: Dict[str, np.ndarray],
        out_maps: Dict[str, np.ndarray],
        surfaces: Dict[str, np.ndarray],
        surface_loss_weight: float = 0.1,
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
        self.surface_loss_weight = float(surface_loss_weight)
        self.surface_tracker = tf.keras.metrics.Mean(name="surface_loss")

    # ------------------------------------------------------------------
    # Campo de superficie en la malla compartida
    # ------------------------------------------------------------------
    def _surface_field(self, refs_by_cfg, preds_by_cfg, kind: str) -> tf.Tensor:
        """Campos de superficie predichos vs reales por configuración."""
        preds, trues = [], []
        for c in self.configs:
            p = tf.matmul(preds_by_cfg[c][kind], self.surfaces[c], transpose_b=True)
            t = tf.matmul(refs_by_cfg[c][kind], self.surfaces[c], transpose_b=True)
            preds.append(p)
            trues.append(t)
        return preds, trues

    def _surface_loss(self, sources, targets) -> tf.Tensor:
        """Pérdida estandarizada sobre el campo en la malla (todas las rutas)."""
        terms = []
        for c in self.configs:
            for s in self.kinds:
                preds = self._predict_cfg(sources[c][s], c, s)
                for d in self.kinds:
                    p = tf.matmul(preds[d], self.surfaces[c], transpose_b=True)
                    t = tf.matmul(targets[c][d], self.surfaces[c], transpose_b=True)
                    terms.append(self._zscore_loss(t, p))
        n_terms = max(1, len(terms))
        return tf.add_n(terms) / n_terms if terms else tf.zeros(())

    # ------------------------------------------------------------------
    # Bueno de entrenamiento/validación
    # ------------------------------------------------------------------
    def train_step(self, data):
        sources, targets = data
        with tf.GradientTape() as tape:
            base = self._config_losses(sources, targets)
            surf = self._surface_loss(sources, targets)
            loss = base + self.surface_loss_weight * surf
            if self.losses:
                loss = loss + tf.add_n(self.losses)  # regularizaciones L2

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "surface_loss": self.surface_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        sources, targets = data
        base = self._config_losses(sources, targets)
        surf = self._surface_loss(sources, targets)
        loss = base + self.surface_loss_weight * surf
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        self.surface_tracker.update_state(surf)
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "surface_loss": self.surface_tracker.result(),
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
        )