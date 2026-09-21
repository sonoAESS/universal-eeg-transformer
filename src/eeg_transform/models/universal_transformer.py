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

Variantes de arquitectura (``model.variant``):

* ``free``
    Autoencoder libre: 8 matrices `C x C` aprendidas sin restricciones.
* ``group``
    El decodificador de cada montaje es la **pseudo-inversa** del encoder del
    mismo montaje (``W^{dec}_k = (W^{enc}_k)^+``); no hay variables de
    decodificador. Esto impone de forma *exacta* dos propiedades físicas que
    el encadenado analítico ``T_d pinv(T_s)`` no satisface (el error de
    composición del baseline analítico es del orden de 1):

    - ``A_{s->s} = W_s (W_s)^+`` es un proyector: identidad sobre el subespacio
      observable (señales sin componente constante).
    - Transitividad exacta: ``A_{s->d} A_{d->u} = A_{s->u}`` para todo
      par, de modo que las referencias forman un **grupo** en el subespacio
      observable. Los encoders usan la proyección ``P`` (como en
      ``projected``), lo que garantiza la exactitud incluso con las matrices
      empíricas de inicialización.

* ``projected``
    Cada matriz aprendida se parametriza como ``W = P W_raw`` con ``P =
    I - 11^T/C``; por construcción toda salida anula el modo constante
    instantáneo (todas las rutas son "referencias válidas", no pueden
    introducir un DC espurio).

* ``soft_group``
    Arquitectura libre (8 matrices) con una **penalización suave** de la
    consistencia de composición en la pérdida: cada lote suma
    ``w * <error de composición>`` a la pérdida total. Interpola entre
    ``free`` (sin restricción) y ``group`` (estructura exacta): obliga a
    componer bien ``A_{s->d} A_{d->u} ≈ A_{s->u}`` sin imponer la
    pseudo-inversa rígida que degrada la convergencia de ``group``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer

from ..config import MODEL_VARIANTS, REFERENCE_KINDS, ModelConfig
from ..logging_conf import get_logger

log = get_logger(__name__)

INITIALIZER_SEED = 42

VARIANT_FREE = "free"
VARIANT_GROUP = "group"
VARIANT_PROJECTED = "projected"
VARIANT_SOFT_GROUP = "soft_group"


class _CenteredDense(layers.Layer):
    """Dense lineal cuya salida anula el modo constante instantáneo.

    Modos de centrado sobre el kernel efectivo:

    * ``"col"`` (``W_eff = P @ W``): ``1^T W_eff = 0`` → la salida anula una
      entrada constante sobre canales (propiedad básica de referencia).
    * ``"both"`` (``W_eff = P @ W @ P``): además ``W_eff @ 1 = 0`` → el
      *rowspace* vive en el subespacio observable. Con ``P`` el proyector de
      centrado ``I - 11^T/C``. Requiere ``units == n_channels`` y garantiza
      que una familia ``W_k`` comparta el mismo subespacio observable, lo que
      hace exacta la composición de grupo de la variante ``group``.
    """

    def __init__(
        self,
        units: int,
        n_channels: int,
        use_bias: bool = False,
        kernel_regularizer: Regularizer | None = None,
        kernel_initializer: Initializer | Any = None,
        name: str = "centered_dense",
        center_mode: str = "col",
    ):
        super().__init__(name=name)
        self.units = units
        self.n_channels = n_channels
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.center_mode = center_mode
        self.proj = tf.constant(
            np.eye(n_channels, dtype=np.float32)
            - np.ones((n_channels, n_channels), dtype=np.float32) / n_channels,
            dtype=tf.float32, name="p_centrado",
        )
        if self.center_mode not in ("col", "both"):
            raise ValueError(f"center_mode debe ser 'col' o 'both': {center_mode}")

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.n_channels, self.units),
            initializer=self.kernel_initializer or "glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,), initializer="zeros",
                trainable=True,
            )
        self.built = True

    def _effective(self, w: tf.Tensor) -> tf.Tensor:
        out = tf.matmul(self.proj, w)          # (n_channels, units)
        if self.center_mode == "both":
            out = tf.matmul(out, self.proj)    # (n_channels, n_channels)
        return out

    def call(self, inputs):
        out = tf.matmul(inputs, self._effective(self.kernel))
        if self.use_bias:
            out = out + self.bias
        return out

    def get_config(self):
        from tensorflow.keras.initializers import serialize as s_init
        from tensorflow.keras.regularizers import serialize as s_reg

        cfg = {
            "units": self.units,
            "n_channels": self.n_channels,
            "use_bias": self.use_bias,
            "kernel_regularizer": s_reg(self.kernel_regularizer)
            if self.kernel_regularizer else None,
            "kernel_initializer": s_init(self.kernel_initializer)
            if isinstance(self.kernel_initializer, Initializer) else None,
            "name": self.name,
            "center_mode": self.center_mode,
        }
        return cfg


class UniversalEEGTransformer(tf.keras.Model):
    """Autoencoder lineal multientrada/multisalida para referencias de EEG.

    En modo **montaje** (``projection`` no nula) el modelo unifica grabaciones
    con cualquier configuración de electrodos: cada entrada del montaje fuente
    ``X_src (n, C_s)`` se proyecta al espacio canónico con la matriz fija
    ``P (C_s, C)`` (solución inversa con el lead field o spline/heatmap, ver
    :mod:`mapping`) y el autoencoder lineal aprende el refinamiento y la
    conversión entre las cuatro referencias canónicas ``(n, C)``.
    """

    def __init__(
        self,
        n_channels: int,
        model_cfg: ModelConfig | None = None,
        projection: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        model_cfg = model_cfg or ModelConfig()
        self.n_channels = n_channels
        self.model_cfg = model_cfg
        self.kinds = list(REFERENCE_KINDS)
        self.latent_dim = int(model_cfg.latent_dim)
        self.variant = (
            model_cfg.variant if model_cfg.variant in MODEL_VARIANTS
            else VARIANT_FREE
        )
        # Proyección fija de montaje C_s -> C_canónica (None = modo canónico).
        self.projection = (
            np.asarray(projection, dtype=np.float32) if projection is not None else None
        )
        self.n_input_channels = (
            self.projection.shape[0] if self.projection is not None else n_channels
        )

        regularizer = (
            tf.keras.regularizers.L2(model_cfg.kernel_regularizer_l2)
            if model_cfg.kernel_regularizer_l2 > 0
            else None
        )
        kernel_init = tf.keras.initializers.GlorotUniform(seed=INITIALIZER_SEED)

        def _make_enc(k: str) -> layers.Layer:
            if self.variant == VARIANT_PROJECTED:
                return _CenteredDense(
                    self.latent_dim, n_channels,
                    use_bias=model_cfg.use_bias,
                    kernel_regularizer=regularizer,
                    kernel_initializer=kernel_init, name=f"enc_{k}",
                    center_mode="col",
                )
            if self.variant == VARIANT_GROUP:
                # Centrado doble (observable) para que las pseudo-inversas de
                # montajes distintos compartan subespacio y la composición de
                # grupo sea exacta.
                return _CenteredDense(
                    self.latent_dim, n_channels,
                    use_bias=model_cfg.use_bias,
                    kernel_regularizer=regularizer,
                    kernel_initializer=kernel_init, name=f"enc_{k}",
                    center_mode="both",
                )
            return layers.Dense(
                self.latent_dim, use_bias=model_cfg.use_bias,
                kernel_regularizer=regularizer, kernel_initializer=kernel_init,
                name=f"enc_{k}",
            )

        def _make_dec(k: str) -> layers.Layer | None:
            if self.variant == VARIANT_GROUP:
                # El decodificador es la pseudo-inversa del encoder del mismo
                # montaje; no hay variables propias.
                return None
            if self.variant == VARIANT_PROJECTED:
                return _CenteredDense(
                    n_channels, n_channels,
                    use_bias=model_cfg.use_bias,
                    kernel_regularizer=regularizer,
                    kernel_initializer=kernel_init, name=f"dec_{k}",
                )
            return layers.Dense(
                n_channels, use_bias=model_cfg.use_bias,
                kernel_regularizer=regularizer, kernel_initializer=kernel_init,
                name=f"dec_{k}",
            )

        self.encoders = {k: _make_enc(k) for k in self.kinds}
        self.decoders = {k: _make_dec(k) for k in self.kinds}
        self.loss_tracker = tf.keras.metrics.Mean(name="loss_estandarizada")
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_real_V2")
        # Las capas se crean en __init__ (no en build), pero el bucle de
        # entrenamiento invoca _predict_from directamente sin pasar por
        # __call__, por lo que `built` no se marcaría nunca; se fija aquí
        # para permitir saving/checkpoints de Keras 3.
        self.built = True

    # ------------------------------------------------------------------
    # Construcción / persistencia
    # ------------------------------------------------------------------
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
            if dec is not None and not dec.built:
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
        ``z -> x_d``.

        * En ``free``/``projected`` se asignan encoder y decoder por ruta.
        * En ``group`` solo se asignan los encoders ``W_enc^k = ridge(x_k ->
          z)``; los decodificadores (pseudo-inversas) se derivan en el forward.

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
        self.ensure_built()

        uni = refs["unipolar"].astype(np.float32)
        means = {k: float(v.mean()) for k, v in refs.items()}

        def _ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            xtx = x.T @ x / n
            lmb = float(ridge * np.trace(xtx) / C + 1e-15)
            return np.linalg.solve(xtx + lmb * np.eye(C), x.T @ y / n).astype(np.float32)

        if self.variant != VARIANT_GROUP:
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
        if self.variant == VARIANT_GROUP:
            w = self._effective_kernel_tf(self.encoders[dest])  # (C, C)
            winv = tf.linalg.pinv(w)
            return tf.matmul(latent, winv)
        return self.decoders[dest](latent)

    def _effective_kernel_tf(self, layer: layers.Layer) -> tf.Tensor:
        w = layer.kernel
        if isinstance(layer, _CenteredDense):
            return layer._effective(w)
        return w

    def _transfer_matrices_tf(self) -> Dict[Tuple[str, str], tf.Tensor]:
        """Matrices efectivas ``A_{s->d}`` como tensores (variante ``group`` no
        es diferenciable de forma estable a través de ``pinv``; se excluye)."""
        enc = {k: self._effective_kernel_tf(self.encoders[k]) for k in self.kinds}
        dec = {k: self._effective_kernel_tf(self.decoders[k]) for k in self.kinds}
        return {
            (s, d): tf.matmul(enc[s], dec[d])
            for s in self.kinds for d in self.kinds
        }

    def _soft_group_penalty(self) -> tf.Tensor:
        """Penalización suave de composición ``s->d->u`` en TF.

        Suma las normas relativas ``||P(A_{s->d} A_{d->u} - A_{s->u})P||_F /
        ||P A_{s->u} P||_F`` sobre todos los tripletes. Alcanza 0 si el
        modelo aprende la estructura de grupo de forma exacta.
        """
        C = self.n_channels
        p = tf.constant(
            np.eye(C, dtype=np.float32) - np.ones((C, C), dtype=np.float32) / C,
            dtype=tf.float32,
        )
        mats = self._transfer_matrices_tf()
        p_mats = {key: tf.matmul(tf.matmul(p, m), p) for key, m in mats.items()}

        def _fro(m: tf.Tensor) -> tf.Tensor:
            return tf.sqrt(tf.reduce_sum(tf.square(m)))

        terms = []
        for s in self.kinds:
            for d in self.kinds:
                for u in self.kinds:
                    lhs = tf.matmul(p_mats[(s, d)], p_mats[(d, u)])
                    rhs = p_mats[(s, u)]
                    denom = _fro(rhs) + 1e-8
                    terms.append(_fro(lhs - rhs) / denom)
        return tf.reduce_sum(terms) / (len(self.kinds) ** 3)

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
        if self.projection is not None:
            # Proyección fija del montaje fuente al espacio canónico (C_s -> C).
            x = tf.matmul(tf.cast(x, tf.float32), tf.constant(self.projection))
        z = self.encode(x, source)
        return {d: self.decode(z, d) for d in self.kinds}

    # ------------------------------------------------------------------
    # Matrices de transferencia efectivas (rendimiento físico)
    # ------------------------------------------------------------------
    def _effective_kernel(self, layer: layers.Layer | None) -> np.ndarray:
        if layer is None:
            return None
        w = layer.kernel.numpy().astype(np.float64)
        if isinstance(layer, _CenteredDense):
            p = np.eye(self.n_channels, dtype=np.float64) - np.ones(
                (self.n_channels, self.n_channels), dtype=np.float64) / self.n_channels
            out = p @ w
            if layer.center_mode == "both":
                out = out @ p
            return out
        return w

    def transfer_matrices(self) -> Dict[Tuple[str, str], np.ndarray]:
        """Devuelve ``A_{s->d}`` para cada ruta (C x C), según la variante.

        * ``free``/``projected``/``soft_group``: ``W_enc^s @ W_dec^d``.
        * ``group``: ``W_enc^s @ pinv(W_enc^d)``.
        * **modo montaje**: ``P @ W_enc^s @ W_dec^d`` con ``P`` la proyección
          fija ``C_s -> C``; cada matriz mapea la observación del montaje
          fuente a los canales canónicos ``(C_s, C)``.
        """
        enc = {k: self._effective_kernel(self.encoders[k]) for k in self.kinds}
        dec = {k: self._effective_kernel(self.decoders[k]) for k in self.kinds}
        out: Dict[Tuple[str, str], np.ndarray] = {}
        for s in self.kinds:
            for d in self.kinds:
                if self.variant == VARIANT_GROUP:
                    core = enc[s] @ np.linalg.pinv(enc[d], rcond=1e-8)
                else:
                    core = enc[s] @ dec[d]
                if self.projection is not None:
                    core = np.float64(self.projection) @ core  # (C_s, C)
                out[(s, d)] = core
        return out

    def composition_error(self) -> Dict[Tuple[str, str, str], float]:
        """Error de composición (física de grupo) por triplete ``s->d->u``.

        Mide ``||P(A_{s->d} A_{d->u} - A_{s->u})P||_F / ||P A_{s->u} P||_F``
        con ``P`` el proyector de centrado. Es 0 exacto para la variante
        ``group`` y ``~1`` para el encadenado analítico ``T_d pinv(T_s)``.

        En **modo montaje** las matrices ``A`` mapean ``C_s -> C`` y la
        composición entre rutas no está definida en el mismo espacio; se
        devuelve un dict vacío (las métricas de montaje usan la reconstrucción
        directa).
        """
        if self.projection is not None:
            return {}
        C = self.n_channels
        p = np.eye(C) - np.ones((C, C)) / C
        mats = self.transfer_matrices()
        rows: Dict[Tuple[str, str, str], float] = {}
        for s in self.kinds:
            for d in self.kinds:
                for u in self.kinds:
                    lhs = p @ (mats[(s, d)] @ mats[(d, u)]) @ p
                    rhs = p @ mats[(s, u)] @ p
                    denom = np.linalg.norm(rhs, "fro") + 1e-15
                    rows[(s, d, u)] = float(np.linalg.norm(lhs - rhs, "fro") / denom)
        return rows

    # ------------------------------------------------------------------
    # Pérdida estandarizada (Z-score por lote)
    # ------------------------------------------------------------------
    @staticmethod
    def _zscore_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """MSE adimensional con la media/desviación del objetivo.

        Los canales objetivo SIN varianza en el lote (canales planos marcados
        como malos, p. ej. Fp en eegbci) se excluyen de la pérdida: su std es
        exactamente 0 y normalizar por él amplificaría el error de la
        predicción en órdenes de magnitud absurdos.
        """
        mean, var = tf.nn.moments(y_true, axes=[0, 1])
        std = tf.math.sqrt(var)
        c_dim = tf.shape(y_true)[-1]
        # válido si aporta información (std apreciable frente al máximo del lote)
        valid = std > 1e-3 * tf.reduce_max(std) + 1e-12
        std_safe = tf.where(valid, std, tf.ones_like(std))
        t = (y_true - mean) / std_safe
        p = (y_pred - mean) / std_safe
        sq = tf.math.square(t - p) * tf.cast(valid, y_true.dtype)
        n_valid = tf.cast(tf.reduce_sum(tf.cast(valid, tf.int32)), y_true.dtype)
        n_lead = tf.cast(
            tf.size(y_true) // c_dim, y_true.dtype
        )
        denom = tf.maximum(n_valid * n_lead, tf.constant(1.0, y_true.dtype))
        return tf.reduce_sum(sq) / denom

    def _route_losses(self, sources, targets):
        """Calcula las 16 pérdidas de ruta (dict origen -> {dest: loss}).

        ``sources`` y ``targets`` son dicts por referencia: en modo canónico
        coinciden; en modo montaje ``sources[k]`` es la observación del
        montaje fuente ``(n, C_s)`` y ``targets[d]`` la referencia canónica
        ``(n, C)``.
        """
        route_losses: Dict[str, Dict[str, tf.Tensor]] = {}
        for s in self.kinds:
            preds = self._predict_from(sources[s], s)
            route_losses[s] = {
                d: self._zscore_loss(targets[d], preds[d]) for d in self.kinds
            }
        return route_losses

    def _real_mse(self, sources, targets):
        """MSE promediado en unidades reales (V^2)."""
        total, count = 0.0, 0
        for s in self.kinds:
            preds = self._predict_from(sources[s], s)
            for d in self.kinds:
                total += losses.MSE(targets[d], preds[d])
                count += 1
        return total / count

    def _split(self, data):
        """Separa fuente y objetivo: modo canónico (mismos arrays) o montaje.

        En modo montaje el dataset entrega la tupla ``(sources, targets)`` con
        ``sources[k]`` de ``(n, C_s)`` (observaciones del montaje fuente) y
        ``targets[d]`` de ``(n, C)`` (referencias canónicas).
        """
        if self.projection is None:
            refs = self._unpack(data)
            return refs, refs
        if isinstance(data, (tuple, list)) and len(data) == 2 and isinstance(data[0], dict) and isinstance(data[1], dict):
            return data[0], data[1]
        if isinstance(data, (tuple, list)) and len(data) and isinstance(data[0], (tuple, list)):
            seq = data[0]
            if len(seq) == 2 and isinstance(seq[0], dict) and isinstance(seq[1], dict):
                return seq[0], seq[1]
        raise TypeError(
            "Modo montaje: se esperaba (sources, targets) como dos dicts "
            "de referencias."
        )

    # ------------------------------------------------------------------
    # Bucles de entrenamiento/validación
    # ------------------------------------------------------------------
    def train_step(self, data):
        sources, targets = self._split(data)
        with tf.GradientTape() as tape:
            route_losses = self._route_losses(sources, targets)
            total = tf.reduce_sum(
                tf.stack([tf.stack(list(r.values())) for r in route_losses.values()])
            )
            if self.variant == VARIANT_SOFT_GROUP and self.projection is None:
                total = total + self.model_cfg.comp_penalty_weight * self._soft_group_penalty()
            if self.losses:
                total = total + tf.add_n(self.losses)  # regularizaciones L2

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total / (len(self.kinds) ** 2))
        self.mse_tracker.update_state(self._real_mse(sources, targets))
        return {
            "loss": self.loss_tracker.result(),
            "loss_estandarizada": self.loss_tracker.result(),
            "mse_real_V2": self.mse_tracker.result(),
        }

    def test_step(self, data):
        sources, targets = self._split(data)
        route_losses = self._route_losses(sources, targets)
        total = tf.reduce_sum(
            tf.stack([tf.stack(list(r.values())) for r in route_losses.values()])
        )
        if self.variant == VARIANT_SOFT_GROUP and self.projection is None:
            total = total + self.model_cfg.comp_penalty_weight * self._soft_group_penalty()
        self.loss_tracker.update_state(total / (len(self.kinds) ** 2))
        self.mse_tracker.update_state(self._real_mse(sources, targets))
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
            "projection": (self.projection.tolist()
                           if self.projection is not None else None),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        projection = config.get("projection")
        if projection is not None:
            projection = np.asarray(projection, dtype=np.float32)
        return cls(
            n_channels=config["n_channels"],
            model_cfg=ModelConfig(**config["model_cfg"]),
            projection=projection,
        )