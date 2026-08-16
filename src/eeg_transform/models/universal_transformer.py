"""Autoencoder lineal All-to-All para la transformación universal de referencias.

La arquitectura codifica cada referencia de origen con un encoder lineal
dedicado hacia un espacio latente central y decodifica simultáneamente en
todas las referencias destino con cabezales lineales:

.. math::

    x_s \\xrightarrow{W^{enc}_s} z \\xrightarrow{W^{dec}_d} \\hat x_d

Toda la red es **lineal** (sin activaciones), de modo que el mapeo efectivo
por ruta es la matriz :math:`A_{s\\to d} = W^{enc}_s\\, W^{dec}_d`. Esto
respeta la estructura algebraica de los campos electrostáticos: la suma y el
re-escalado de señales se preservan de forma exacta.

La pérdida se calcula en escala adimensional (Z-score por lote) para que las
16 rutas tengan el mismo peso en la retropropagación, y también se registran
métricas en unidades reales (voltios).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses

from ..config import REFERENCE_KINDS, ModelConfig
from ..logging_conf import get_logger

log = get_logger(__name__)

INITIALIZER_SEED = 42


class UniversalEEGTransformer(tf.keras.Model):
    """Autoencoder lineal multientrada/multisalida para referencias de EEG."""

    def __init__(self, n_channels: int, model_cfg: ModelConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        model_cfg = model_cfg or ModelConfig()
        self.n_channels = n_channels
        self.model_cfg = model_cfg
        self.kinds = list(REFERENCE_KINDS)
        self.latent_dim = int(model_cfg.latent_dim)

        regularizer = (
            tf.keras.regularizers.L2(model_cfg.kernel_regularizer_l2)
            if model_cfg.kernel_regularizer_l2 > 0
            else None
        )
        # GlorotUniform (inicialización por defecto) ofrece mejores dinámicas
        # que los inicializadores ortogonales para esta factorización lineal.
        kernel_init = tf.keras.initializers.GlorotUniform(seed=INITIALIZER_SEED)

        self.encoders = {
            k: layers.Dense(
                self.latent_dim,
                use_bias=model_cfg.use_bias,
                kernel_regularizer=regularizer,
                kernel_initializer=kernel_init,
                name=f"enc_{k}",
            )
            for k in self.kinds
        }
        self.decoders = {
            k: layers.Dense(
                n_channels,
                use_bias=model_cfg.use_bias,
                kernel_regularizer=regularizer,
                name=f"dec_{k}",
            )
            for k in self.kinds
        }
        self.loss_tracker = tf.keras.metrics.Mean(name="loss_estandarizada")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_real_V2")
        # Las capas se crean en __init__ (no en build), pero el bucle de
        # entrenamiento invoca _predict_from directamente sin pasar por
        # __call__, por lo que `built` no se marcaría nunca; se fija aquí
        # para permitir saving/checkpoints de Keras 3.
        self.built = True

    def ensure_built(self) -> None:
        """Construye explícitamente las capas Dense (kernels/bias).

        Keras construye las capas perezosamente en la primera llamada; si se
        llama a ``load_weights`` antes de eso, el restablecimiento es un
        *no-op* silencioso y el modelo queda con pesos aleatorios. Este
        método garantiza que las Variables existan antes de cargar peso o
        pre-inicializar.
        """
        for enc in self.encoders.values():
            if not enc.built:
                enc.build((None, self.n_channels))
        for dec in self.decoders.values():
            if not dec.built:
                dec.build((None, self.latent_dim))

    # ------------------------------------------------------------------
    # Inicialización con la factorización empírica óptima
    # ------------------------------------------------------------------
    def init_from_data(self, refs: Dict[str, np.ndarray], ridge: float = 1e-4):
        """Inicializa ``W_enc``/``W_dec`` con la factorización lineal óptima.

        Como todas las referencias provienen de la misma señal por operadores
        lineales, existe una factorización *exacta* con espacio latente de
        dimensión ``C``: si ``z = x_unipolar``, entonces ``W_enc^s`` es la
        regresión ridge ``x_s -> z`` y ``W_dec^d`` la regresión ridge
        ``z -> x_d``. La composición ``W_enc^s W_dec^d`` reproduce entonces
        cada ruta ``s->d`` casi a la perfección desde la época 0, y el
        entrenamiento solo la ajusta.

        Parameters
        ----------
        refs:
            Diccionario ``{kind: (n, C)}`` de referencias alineadas (submuestra
            del split de entrenamiento).
        ridge:
            Regularización relativa (fracción del valor propio medio) para
            estabilizar la inversión de ``X^T X`` (los canales están en la
            misma escala física).
        """
        if self.latent_dim != self.n_channels:
            raise ValueError(
                "init_from_data requiere latent_dim == n_channels "
                f"({self.latent_dim} != {self.n_channels})."
            )
        C = self.n_channels
        # Garantiza que las Variables kernel/bias existan (Keras construye las
        # capas perezosamente en la primera llamada).
        self.ensure_built()

        uni = refs["unipolar"].astype(np.float32)
        means = {k: float(v.mean()) for k, v in refs.items()}

        def _ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            xtx = x.T @ x / n
            lmb = float(ridge * np.trace(xtx) / C + 1e-15)
            return np.linalg.solve(xtx + lmb * np.eye(C), x.T @ y / n).astype(np.float32)

        for d in self.kinds:
            w = _ridge(uni - means["unipolar"], refs[d] - means[d])
            self.decoders[d].kernel.assign(w)
            if self.decoders[d].use_bias:
                self.decoders[d].bias.assign(np.zeros(C, np.float32))
        for s in self.kinds:
            w = _ridge(refs[s] - means[s], uni - means["unipolar"])
            self.encoders[s].kernel.assign(w)
            if self.encoders[s].use_bias:
                self.encoders[s].bias.assign(np.zeros(C, np.float32))

    # ------------------------------------------------------------------
    # Cálculo
    # ------------------------------------------------------------------
    def encode(self, inputs, source: str) -> tf.Tensor:
        return self.encoders[source](inputs)

    def decode(self, latent: tf.Tensor, dest: str) -> tf.Tensor:
        return self.decoders[dest](latent)

    def call(self, inputs, source: str | None = None, **kwargs):
        """Predicción universal.

        Parameters
        ----------
        inputs:
            Si ``source`` es nula, dict ``{kind: tensor (n, C)}`` y se
            devuelve dict con las 4 predicciones por cada origen.
            En caso contrario, matriz ``(n, C)`` del origen y se devuelve
            dict ``{dest: predicción}``.
        """
        if source is None:
            if isinstance(inputs, dict):
                out = {}
                for s in self.kinds:
                    out[s] = self._predict_from(inputs[s], s)
                return out
            raise ValueError("inputs debe ser un dict de referencias.")
        return self._predict_from(inputs, source)

    def _predict_from(self, x: tf.Tensor, source: str) -> Dict[str, tf.Tensor]:
        z = self.encode(x, source)
        return {d: self.decode(z, d) for d in self.kinds}

    # ------------------------------------------------------------------
    # Matrices de transferencia efectivas (rendimiento físico)
    # ------------------------------------------------------------------
    def transfer_matrices(self) -> Dict[Tuple[str, str], np.ndarray]:
        """Devuelve ``A_{s->d} = W^enc_s @ W^dec_d`` para cada ruta (C x C)."""
        # garantiza que las capas estén construidas
        z = tf.zeros((1, self.n_channels))
        z = self._predict_from(z, self.kinds[0])
        out: Dict[Tuple[str, str], np.ndarray] = {}
        for s in self.kinds:
            w_e = self.encoders[s].kernel.numpy()       # (C, latent)
            for d in self.kinds:
                w_d = self.decoders[d].kernel.numpy()   # (latent, C)
                out[(s, d)] = w_e @ w_d
        return out

    # ------------------------------------------------------------------
    # Pérdida estandarizada (Z-score por lote)
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """MSE adimensional con la media/desviación del objetivo."""
        mean, var = tf.nn.moments(y_true, axes=[0, 1])
        std = tf.math.sqrt(var) + 1e-8
        t = (y_true - mean) / std
        p = (y_pred - mean) / std
        return tf.reduce_mean(tf.math.square(t - p))

    def _route_losses(self, refs: Dict[str, tf.Tensor]):
        """Calcula las 16 pérdidas de ruta (dict origen -> {dest: loss})."""
        route_losses: Dict[str, Dict[str, tf.Tensor]] = {}
        for s in self.kinds:
            preds = self._predict_from(refs[s], s)
            route_losses[s] = {
                d: self._zscore_loss(refs[d], preds[d]) for d in self.kinds
            }
        return route_losses

    def _real_mse(self, refs: Dict[str, tf.Tensor]):
        """MSE promediado en unidades reales (V^2)."""
        total, count = 0.0, 0
        for s in self.kinds:
            preds = self._predict_from(refs[s], s)
            for d in self.kinds:
                total += losses.MSE(refs[d], preds[d])
                count += 1
        return total / count

    # ------------------------------------------------------------------
    # Bucles de entrenamiento/validación
    # ------------------------------------------------------------------
    def train_step(self, data):
        refs = self._unpack(data)
        with tf.GradientTape() as tape:
            route_losses = self._route_losses(refs)
            total = tf.reduce_sum(
                tf.stack([tf.stack(list(r.values())) for r in route_losses.values()])
            )
            total = total + tf.add_n(self.losses)  # regularizaciones L2

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total / (len(self.kinds) ** 2))
        self.mse_tracker.update_state(self._real_mse(refs))
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        refs = self._unpack(data)
        route_losses = self._route_losses(refs)
        total = tf.reduce_sum(
            tf.stack([tf.stack(list(r.values())) for r in route_losses.values()])
        )
        self.loss_tracker.update_state(total / (len(self.kinds) ** 2))
        self.mse_tracker.update_state(self._real_mse(refs))
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    @staticmethod
    def _unpack(data) -> Dict[str, tf.Tensor]:
        """Normaliza la entrada: dict o tupla ordenada de las 4 referencias."""
        if isinstance(data, dict):
            return {k: data[k] for k in REFERENCE_KINDS}
        if isinstance(data, (tuple, list)):
            seq = data[0]
            if isinstance(seq, dict):
                return {k: seq[k] for k in REFERENCE_KINDS}
            if isinstance(seq, (tuple, list)) and len(seq) >= len(REFERENCE_KINDS):
                return {k: seq[i] for i, k in enumerate(REFERENCE_KINDS)}
            return {k: data[i] for i, k in enumerate(REFERENCE_KINDS)}
        raise TypeError(f"Formato de entrada no soportado: {type(data)}")

    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "n_channels": self.n_channels,
            "latent_dim": self.latent_dim,
            "kinds": self.kinds,
            "model_cfg": dataclasses.asdict(self.model_cfg),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(
            n_channels=config["n_channels"],
            model_cfg=ModelConfig(**config["model_cfg"]),
        )