"""Entrenamiento del transformador universal (lotes sobre referencias alineadas)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ..config import EEGTransformConfig, REFERENCE_KINDS
from ..data.dataset import MultiReferenceDataset
from ..logging_conf import get_logger
from ..models.universal_transformer import UniversalEEGTransformer

log = get_logger(__name__)


def is_montage_variant(cfg: EEGTransformConfig) -> bool:
    """True si la variante consume un montaje fuente (``montage_*``)."""
    return cfg.model.variant.startswith("montage_")


def build_montage_inputs(cfg: EEGTransformConfig, ds: MultiReferenceDataset):
    """Insume de montaje (observaciones fuente + proyección) para entrenar."""
    from ..experiments.montage import build_montage_inputs as _build

    return _build(ds, cfg)


def build_tf_dataset(
    ds: MultiReferenceDataset,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    buffer: int = 20_000,
    prefetch: int = 4,
    dtype: str = "float32",
    sources: dict[str, dict[str, np.ndarray]] | None = None,
) -> tf.data.Dataset:
    """Dataset de TF con los 4 montajes alineados como tupla de tensores.

    En modo montaje (``sources`` no nulo) cada elemento es la tupla
    ``(dict_fuente, dict_objetivo)``: ``dict_fuente[k]`` son las
    observaciones ``(n, C_s)`` del montaje fuente y ``dict_objetivo[k]`` las
    referencias canónicas ``(n, C)``.
    """
    idx = ds.split_idx[split]
    if sources is None:
        arrays = tuple(
            ds.refs[k][idx].astype(dtype) for k in REFERENCE_KINDS
        )
        dset = tf.data.Dataset.from_tensor_slices(arrays)
        _n = arrays[0].shape[0]
    else:
        src_arrays = {
            k: sources[split][k].astype(dtype) for k in REFERENCE_KINDS
        }
        tgt_arrays = {
            k: ds.refs[k][idx].astype(dtype) for k in REFERENCE_KINDS
        }
        dset = tf.data.Dataset.from_tensor_slices((src_arrays, tgt_arrays))
        _n = next(iter(src_arrays.values())).shape[0]
    if shuffle:
        dset = dset.shuffle(min(buffer, _n), seed=seed,
                            reshuffle_each_iteration=True)
    dset = dset.repeat()
    return dset.batch(batch_size, drop_remainder=False).prefetch(prefetch)


def build_model(
    cfg: EEGTransformConfig,
    n_channels: int,
    projection: np.ndarray | None = None,
) -> UniversalEEGTransformer:
    """Instancia el modelo según la configuración.

    ``projection`` (matriz ``C_s -> C``) activa el **modo montaje**: las
    entradas son observaciones del montaje fuente proyectadas al espacio
    canónico antes del autoencoder (ver :mod:`models.universal_transformer`).
    """
    model_cfg = cfg.model
    if model_cfg.latent_dim in (0, -1):
        model_cfg.latent_dim = n_channels
    return UniversalEEGTransformer(
        n_channels=n_channels, model_cfg=model_cfg, projection=projection
    )


class _History:
    """Contenedor mínimo con la API de ``keras.callbacks.History``."""

    def __init__(self, history: dict):
        self.history = history


def load_history(history_csv: str | Path) -> _History:
    """Reconstruye un objeto historia desde el CSV de entrenamiento."""
    import pandas as pd

    if not Path(history_csv).exists():
        return _History({})
    df = pd.read_csv(history_csv)
    # Keras escribe métricas como cadenas (p. ej. "'0.123'"); se normalizan
    # a float y se guarda el diccionario {métrica: lista por época}.
    history: dict[str, list[float]] = {}
    for col in df.columns:
        try:
            history[col] = list(pd.to_numeric(df[col], errors="coerce").fillna(0.0))
        except Exception:
            history[col] = list(df[col])
    return _History(history)


def train(
    ds: MultiReferenceDataset,
    cfg: EEGTransformConfig,
    force: bool = False,
) -> tuple[UniversalEEGTransformer, Any]:
    """Entrena el transformador universal sobre el dataset multi-referencia.

    Con una variante ``montage_*`` entrena sobre las observaciones del montaje
    fuente: proyecta cada entrada con la matriz fija ``P`` y aprende a estimar
    las referencias canónicas desde **cualquier** configuración de electrodos.
    """
    tcfg = cfg.training
    run_dir = Path(tcfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    montage = build_montage_inputs(cfg, ds) if is_montage_variant(cfg) else None

    checkpoint = run_dir / "best.weights.h5"
    history_csv = run_dir / "history.csv"
    if checkpoint.exists() and not force:
        log.info("Checkpoint previo detectado (%s) — reutilizando.", checkpoint)
        n = max(1, ds.n_channels)
        model = build_model(cfg, n, projection=montage.projection if montage else None)
        model.ensure_built()
        model.load_weights(str(checkpoint))
        history = load_history(history_csv)
        return model, history
    if history_csv.exists():
        # Un reentrenamiento con --force debe partir de un log limpio (evita
        # concatenar épocas de corridas distintas en un mismo CSV).
        history_csv.unlink()

    tf.random.set_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    n = max(1, ds.n_channels)
    model = build_model(cfg, n, projection=montage.projection if montage else None)

    # Pre-entrenamiento lineal: factorización empírica óptima de las rutas.
    # Sitúa las matrices efectivas cerca de las analíticas desde la época 0,
    # evitando que el gradiente quede atrapado en cuencas degeneradas del
    # autoencoder lineal (las rutas son exactamente lineales por construcción).
    if model.latent_dim == n:
        init_n = 20_000
        init_idx = ds.split_idx["train"][:init_n]
        if montage is None:
            init_refs = {k: ds.refs[k][init_idx] for k in REFERENCE_KINDS}
        else:
            # El ancla del latente es unipolar proyectada al espacio canónico.
            init_refs = {
                k: (montage.src_refs["train"][k][:init_n] @ montage.projection)
                .astype(np.float32)
                for k in REFERENCE_KINDS
            }
        model.init_from_data(init_refs)
        log.info("Inicialización lineal empírica desde %d muestras.", len(init_idx))

    model.compile(
        optimizer=tf.keras.optimizers.get(
            {"class_name": cfg.model.optimizer,
             "config": {"learning_rate": cfg.model.learning_rate}}
        )
    )

    train_ds = build_tf_dataset(
        ds, "train", tcfg.batch_size, shuffle=True, seed=tcfg.seed,
        buffer=tcfg.shuffle_buffer, prefetch=tcfg.prefetch, dtype=cfg.dataset.dtype,
        sources=montage.src_refs if montage else None,
    )
    val_ds = build_tf_dataset(
        ds, "val", tcfg.batch_size, shuffle=False, prefetch=tcfg.prefetch,
        dtype=cfg.dataset.dtype,
        sources=montage.src_refs if montage else None,
    )

    steps_per_epoch = max(1, len(ds.split_idx["train"]) // tcfg.batch_size)
    validation_steps = max(1, len(ds.split_idx["val"]) // tcfg.batch_size)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss_estandarizada",
            save_best_only=tcfg.save_best_only, save_weights_only=True, mode="min",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss_estandarizada", patience=tcfg.early_stop_patience,
            restore_best_weights=True, mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss_estandarizada", factor=tcfg.reduce_lr_factor,
            patience=tcfg.reduce_lr_patience, min_lr=tcfg.min_lr, mode="min",
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"),
                                      append=False),
    ]

    log.info("Entrenando %s épocas (batch=%d)...", tcfg.epochs, tcfg.batch_size)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tcfg.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # model.load_weights(checkpoint)  # restore_best_weights ya lo hace
    model.load_weights(str(checkpoint))
    model.save(run_dir / "model.keras")
    log.info("Modelo guardado en %s", run_dir / "model.keras")
    return model, history