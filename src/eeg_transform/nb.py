"""Helpers reutilizables para los cuadernos de exploración de variantes.

Centraliza la lógica compartida de todos los notebooks de ``notebooks/`` para
que las celdas queden limpias y mantengan buenas prácticas: misma ruta de
carga de datos, misma evaluación, mismas figuras. Cada notebook solo declara
su ``config_path`` y delega aquí el resto.

Funciones expuestas (todas devuelven objetos ``pandas``/``matplotlib`` listos
para mostrarse inline):

* :func:`load_experiment`   → ``(cfg, ds)`` cargados desde el YAML.
* :func:`train_variant`     → entrena/reutiliza el modelo y devuelve la historia.
* :func:`evaluate_montage` / :func:`evaluate_multiconfig*` → métricas.
* :func:`plot_training`, :func:`plot_routes`, :func:`plot_scalp`,
  :func:`plot_multiconfig*`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .config import EEGTransformConfig, load_config
from .data.dataset import MultiReferenceDataset, build_dataset
from .evaluation import metrics, plots
from .logging_conf import get_logger
from .training.trainer import (
    build_montage_inputs,
    is_montage_variant,
    load_multiconfig_data,
    train,
)

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


def montage_inputs(cfg: EEGTransformConfig, ds: MultiReferenceDataset):
    """Insumo de montaje (observaciones + proyección) para variantes ``montage_*``."""
    if not is_montage_variant(cfg):
        return None
    return build_montage_inputs(cfg, ds)


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


def evaluate_montage(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
    model,
    montage: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluación de montaje: métricas por ruta y resumen modelo vs proyección.

    Returns
    -------
    ``(metrics_df, summary)`` con la línea base analítica (``obs @ P``) en
    columnas ``*_proy`` para cuantificar la ganancia del aprendizaje.
    """
    montage = montage or montage_inputs(cfg, ds)
    if montage is None:
        raise ValueError("evaluate_montage requiere una variante 'montage_*'.")
    metrics_df = metrics.evaluate_montage_routes(model, ds, montage,
                                                 split="test")
    summary = metrics.summarize_montage(metrics_df)
    return metrics_df, summary


def plot_scalp(cfg, ds, model, montage=None):
    """Heatmap de cuero cabelludo (electrodos como manchas) para notebooks."""
    plots.set_plot_backend("inline")
    montage = montage or montage_inputs(cfg, ds)
    if montage is None:
        raise ValueError("plot_scalp requiere una variante 'montage_*'.")
    return plots.scalp_heatmap_fig(
        model, ds, montage, cfg, split="test",
        n_samples=cfg.evaluation.n_plot_samples,
        grid_px=cfg.mapping.grid_px,
    )


# ---------------------------------------------------------------------------
# Múltiples configuraciones de electrodos (variante ``multi_montage``)
# ---------------------------------------------------------------------------

def multiconfig_data(cfg: EEGTransformConfig, ds: MultiReferenceDataset):
    """Datos multi-configuración (observaciones por configuración + refs)."""
    return load_multiconfig_data(cfg, ds)


def evaluate_multiconfig(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
    model,
    data: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por configuración (diagonal/cruzada + línea base analítica).

    Returns
    -------
    ``(metrics_df, summary)`` con la línea base analítica
    (``T_d @ pinv(T_s)`` sobre la ref de ancla) en columnas ``*_ana``.
    """
    data = data or multiconfig_data(cfg, ds)
    metrics_df = metrics.evaluate_multiconfig_routes(model, data, split="test")
    summary = metrics.summarize_multiconfig(metrics_df)
    return metrics_df, summary


def evaluate_multiconfig_windowed(
    cfg: EEGTransformConfig,
    ds: MultiReferenceDataset,
    model,
    data: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas por configuración con la cabeza temporal activa (causal).

    Igual esquema que :func:`evaluate_multiconfig` (diagonal/cruzada + línea
    base analítica), pero evaluando las rutas por **ventanas causales**:
    cada instante se predice desde la ventana de contexto previo y se conserva
    el último paso. Es la medida que captura el beneficio de la recurrencia en
    inferencia; con ``temporal_window = 0`` coincide con la evaluación
    instantánea.

    Returns
    -------
    ``(metrics_df, summary)``.
    """
    data = data or multiconfig_data(cfg, ds)
    metrics_df = metrics.evaluate_multiconfig_windowed(
        model, data, split="test",
        window=cfg.model.temporal_window, stride=1,
    )
    summary = metrics.summarize_multiconfig(metrics_df)
    return metrics_df, summary


def plot_multiconfig_heatmap(metrics_df: pd.DataFrame, title: str):
    """Figura heatmap RMSE (µV, escala log10) por config origen→destino."""
    plots.set_plot_backend("inline")
    fig = plots.multiconfig_heatmap_fig(metrics_df)
    fig.suptitle(title)
    return fig


def plot_multiconfig_bars(metrics_df: pd.DataFrame, title: str):
    """Figura de barras RMSE/ve por configuración (línea base analítica)."""
    plots.set_plot_backend("inline")
    fig = plots.multiconfig_bars_fig(metrics_df)
    fig.suptitle(title)
    return fig


def plot_multiconfig_scalps(cfg: EEGTransformConfig, model, data=None,
                            split: str = "test",
                            n_samples: int | None = None):
    """Mapas de calor del cuero cabelludo por configuración (uno por label).

    Returns
    -------
    ``dict {label: Figure}`` con los topomapas de cada configuración
    no-canónica (Obs → Modelo → Verdad, intra-configuración).
    """
    plots.set_plot_backend("inline")
    if data is None:
        raise ValueError("plot_multiconfig_scalps requiere `data`.")
    n_samples = n_samples or cfg.evaluation.n_plot_samples
    return {
        label: plots.multiconfig_scalp_fig(
            model, data, label, split=split,
            n_samples=n_samples, grid_px=cfg.mapping.grid_px,
        )
        for label in data.order
        if label != "canonical"
    }


def evaluate_multiconfig_surface(
    model, data, split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas del campo de superficie (malla compartida) por configuración.

    Returns
    -------
    ``(metrics_surface_df, summary)`` con RMSE/r/VE del "heatmap" predicho
    (patrón espacial), válido solo para la variante ``multi_heatmap``.
    """
    surface_df = metrics.evaluate_multiconfig_surface_routes(model, data,
                                                             split=split)
    summary = metrics.summarize_multiconfig_surface(surface_df)
    return surface_df, summary


def plot_multiconfig_surface(metrics_surface_df: pd.DataFrame, title: str):
    """Figura de barras del campo de superficie por configuración."""
    plots.set_plot_backend("inline")
    fig = plots.multiconfig_surface_fig(metrics_surface_df)
    fig.suptitle(title)
    return fig


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


def config_table(cfg: EEGTransformConfig) -> pd.DataFrame:
    """Tabla plana de la configuración (para mostrar en markdown/HTML)."""
    rows = []
    for sec in ("data", "leadfield", "dataset", "model", "training"):
        block = getattr(cfg, sec)
        for f in dataclasses.fields(block):
            rows.append({"sección": sec, "parámetro": f.name,
                         "valor": getattr(block, f.name)})
    return pd.DataFrame(rows)