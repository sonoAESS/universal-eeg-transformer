"""Helpers reutilizables para los cuadernos de exploración de variantes.

Centraliza la lógica compartida de todos los notebooks de ``notebooks/`` para
que las celdas queden limpias y mantengan buenas prácticas: misma ruta de
carga de datos, misma evaluación, mismas figuras. Cada notebook solo declara
su ``config_path`` y delega aquí el resto.

Funciones expuestas (todas devuelven objetos ``pandas``/``matplotlib`` listos
para mostrarse inline):

* :func:`load_experiment`   → ``(cfg, ds)`` cargados desde el YAML.
* :func:`train_variant`     → entrena/reutiliza el modelo y devuelve la historia.
* :func:`evaluate`          → tablas de métricas, error Frobenius y composición.
* :func:`plot_training`, :func:`plot_routes`, :func:`plot_traces`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .config import EEGTransformConfig, load_config
from .data.dataset import MultiReferenceDataset, build_dataset
from .evaluation import metrics, plots
from .logging_conf import get_logger
from .training.trainer import build_model, train

log = get_logger(__name__)


def load_experiment(
    config_path: str | Path,
) -> tuple[EEGTransformConfig, MultiReferenceDataset]:
    """Carga la configuración YAML y construye/cachea el dataset."""
    cfg = load_config(config_path)
    ds = build_dataset(cfg)
    log.info("Experimento cargado: %s (variante %s, %d canales)",
             Path(config_path).name, cfg.model.variant, ds.n_channels)
    return cfg, ds


def ensure_model(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
) -> object:
    """Instancia el modelo con los pesos del checkpoint (best.weights.h5).

    Fallbacks en orden: checkpoint guardado → modelo recién entrenado por
    :func:`train_variant`. Nunca devuelve pesos aleatorios salvo que se pida
    ``train`` explícito y no exista checkpoint.
    """
    run_dir = Path(cfg.training.run_dir)
    checkpoint = run_dir / "best.weights.h5"
    model = build_model(cfg, ds.n_channels)
    model.ensure_built()
    if checkpoint.exists():
        model.load_weights(str(checkpoint))
        log.info("Modelo cargado desde %s", checkpoint)
    return model


def train_variant(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
    force: bool = False,
) -> tuple[object, object]:
    """Entrena la variante (o reutiliza el checkpoint si existe).

    Returns
    -------
    ``(model, history)`` con el modelo ya cargado con los mejores pesos;
    ``history`` es un objeto con atributo ``.history`` (dict por métrica).
    """
    model, history = train(ds, cfg, force=force)
    return model, history


def evaluate(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
    model,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Métricas por ruta, error Frobenius y consistencia de composición."""
    uni_idx = ds.ch_names.index(cfg.data.unipolar_ref_ch)
    metrics_df = metrics.evaluate_routes(model, ds, split="test")
    err_matrix = metrics.transfer_error_matrix(model, ds, uni_idx, cfg)
    cons = metrics.composition_error_table(model, ds)
    return metrics_df, err_matrix, cons


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen diagonal/cruzada en unidades legibles (RMSE µV)."""
    sm = metrics.summarize(metrics_df, "test").copy()
    for col in ("mse", "rmse", "mae"):
        sm[col + "_uV"] = sm[col] * 1e6
    return sm[["tipo", "rmse_uV", "mae_uV", "r"]]


def plot_training(history_csv: str | Path):
    """Figura de curvas de aprendizaje."""
    plots.set_plot_backend("inline")
    return plots.learning_curves_fig(history_csv)


def plot_routes(metrics_df: pd.DataFrame, value_col: str = "rmse",
                title: str = "RMSE real por ruta (µV)"):
    """Figura heatmap de una métrica por ruta origen→destino."""
    plots.set_plot_backend("inline")
    df = metrics_df.copy()
    if value_col in ("rmse", "mae", "mse"):
        df[value_col] = df[value_col] * 1e6
    return plots.heatmap_fig(df, value_col, title)


def plot_traces(model, ds, cfg: EEGTransformConfig):
    """Figura de trazas reales vs predichas (canal de evaluación)."""
    plots.set_plot_backend("inline")
    return plots.traces_fig(
        model, ds, split="test",
        channel=cfg.evaluation.plot_channel,
        n_samples=cfg.evaluation.n_plot_samples,
    )


def config_table(cfg: EEGTransformConfig) -> pd.DataFrame:
    """Tabla plana de la configuración (para mostrar en markdown/HTML)."""
    rows = []
    for sec in ("data", "leadfield", "dataset", "model", "training"):
        block = getattr(cfg, sec)
        for f in dataclasses.fields(block):
            rows.append({"sección": sec, "parámetro": f.name,
                         "valor": getattr(block, f.name)})
    return pd.DataFrame(rows)