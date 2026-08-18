"""Figuras de evaluación: curvas de aprendizaje, matrices de error y trazas.

Cada figura se construye con ``*_fig`` (devuelve ``matplotlib.figure.Figure``,
reutilizable para mostrarla inline en notebooks) y se guarda con las
funciones ``plot_*`` del CLI a ``runs/<nombre>/figs/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import REFERENCE_KINDS  # noqa: E402
from ..logging_conf import get_logger  # noqa: E402

log = get_logger(__name__)

KINDS = list(REFERENCE_KINDS)
KIND_LABELS = {
    "unipolar": "Unipolar",
    "bipolar": "Bipolar",
    "car": "CAR",
    "rest": "REST",
}

# El CLI usa backend Agg (headless); los notebooks pueden cambiar a inline
# SIN recargar el módulo llamando a `set_plot_backend("inline")`.
_PLOT_BACKEND = None


def set_plot_backend(backend: str) -> None:
    """Cambia el backend de matplotlib en caliente (p. ej. 'inline')."""
    global _PLOT_BACKEND
    if backend != _PLOT_BACKEND:
        matplotlib.use(backend, force=True)
        _PLOT_BACKEND = backend


def _save(fig, run_dir: Path, name: str) -> None:
    out = run_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Figura guardada: %s", out / name)


def learning_curves_fig(history_csv: str | Path) -> plt.Figure:
    """Curvas de entrenamiento (pérdida estandarizada y MSE real)."""
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["loss_estandarizada"], label="train", color="rebeccapurple", lw=2)
    axes[0].plot(df["val_loss_estandarizada"], label="val", ls="--", color="gold", lw=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida estandarizada")
    axes[0].set_title("Pérdida por época")
    axes[0].legend(); axes[0].grid(alpha=0.4)
    if "mse_real_V2" in df:
        axes[1].plot(df["mse_real_V2"], color="teal", lw=2)
        if "val_mse_real_V2" in df:
            axes[1].plot(df["val_mse_real_V2"], ls="--", color="coral", lw=2)
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("MSE real (V²)")
        axes[1].set_title("MSE real por época")
        axes[1].grid(alpha=0.4)
    fig.suptitle("Curvas de entrenamiento")
    fig.tight_layout()
    return fig


def heatmap_fig(metrics_df: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    """Heatmap log10 de una métrica por ruta origen→destino."""
    table = pd.DataFrame(index=KINDS, columns=KINDS, dtype=float)
    for _, row in metrics_df.iterrows():
        table.loc[row["origen"], row["destino"]] = row[value_col]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(table.values.astype(float) + 1e-30), cmap="viridis")
    ax.set_xticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS])
    ax.set_yticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS])
    ax.set_xlabel("Destino"); ax.set_ylabel("Origen")
    ax.set_title(title)
    for i in range(len(KINDS)):
        for j in range(len(KINDS)):
            ax.text(j, i, f"{table.values[i, j]:.1e}",
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10(valor)")
    fig.tight_layout()
    return fig


def traces_fig(model, ds, split: str, channel: str, n_samples: int) -> plt.Figure:
    """Trazas reales vs predichas para las rutas cruzadas más relevantes."""
    ch = ds.ch_names.index(channel)
    idx = ds.split_idx[split][:n_samples]
    refs = {k: ds.refs[k][idx] for k in KINDS}

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(17, 7), sharex=True)
    routes = [("unipolar", "rest"), ("bipolar", "rest"), ("car", "rest"),
              ("rest", "unipolar"), ("bipolar", "car"), ("car", "car")]
    # inputs por lotes con tf
    import tensorflow as tf
    for ax, (s, d) in zip(axes.ravel(), routes):
        x = tf.convert_to_tensor(refs[s], tf.float32)
        pred = model(x, source=s)[d].numpy()
        t = np.arange(len(idx))
        ax.plot(t, refs[d][:, ch], color="black", lw=1.0, alpha=0.85, label="Real")
        ax.plot(t, pred[:, ch], color="dodgerblue", lw=1.2, ls=":", alpha=0.9,
                label="Predicción")
        ax.set_title(f"{KIND_LABELS[s]} → {KIND_LABELS[d]}")
        ax.grid(alpha=0.35)
        lo, hi = np.percentile(refs[d][:, ch], [1, 99]) * 1.3
        if hi - lo < 1e-30:  # canal plano
            lo, hi = -1.0, 1.0
        ax.set_ylim(lo, hi)
    axes[0, 0].legend()
    fig.suptitle(f"Canal {channel}: señales reales y predichas (test)")
    fig.tight_layout()
    return fig


def plot_learning_curves(history_csv: str | Path, run_dir: Path) -> None:
    _save(learning_curves_fig(history_csv), run_dir, "learning.png")


def plot_heatmap(
    metrics_df: pd.DataFrame, value_col: str, title: str,
    run_dir: Path, fname: str,
) -> None:
    _save(heatmap_fig(metrics_df, value_col, title), run_dir, fname)


def plot_traces(model, ds, split: str, channel: str, n_samples: int,
                run_dir: Path) -> None:
    _save(traces_fig(model, ds, split, channel, n_samples), run_dir,
          "traces.png")


# ---------------------------------------------------------------------------
# Montaje: heatmaps de cuero cabelludo (actividad como "manchas")
# ---------------------------------------------------------------------------
def _render_scalp_field(
    values: np.ndarray,
    matrix: np.ndarray,
    valid: np.ndarray,
    grid_px: int,
) -> np.ndarray:
    """Interpola un vector de canales a imagen ``(grid_px, grid_px)`` (mask)."""
    out = (matrix @ values).astype(np.float64)   # (n_grid,)
    out[~valid] = np.nan                          # fuera del cuero cabelludo
    return out.reshape(grid_px, grid_px)


def _electrode_disc(positions: np.ndarray, grid_px: int):
    """Electrodos 3D proyectados a coordenadas de píxel (vista central)."""
    from ..mapping import _project_to_disc

    xy = _project_to_disc(np.asarray(positions, dtype=np.float64))
    px = ((xy[:, 0] + 1.0) / 2.0 * (grid_px - 1))
    py = ((1.0 - (xy[:, 1] + 1.0) / 2.0) * (grid_px - 1))
    return px, py


def scalp_heatmap_fig(
    model,
    ds,
    montage_inputs,
    cfg,
    split: str = "test",
    n_samples: int = 400,
    grid_px: int | None = None,
) -> plt.Figure:
    """Mapas de calor del cuero cabelludo por referencia (montaje → canónico).

    Cada fila es una referencia canónica; cada columna una etapa de la
    unificación de montajes:

    1. **Observación** del montaje fuente (electrodos como puntos, actividad
       interpolada como manchas; los electrodos se muestran superpuestos).
    2. **Proyección analítica** (``obs @ P``, sin aprendizaje).
    3. **Modelo** (autoencoder universal que refina la proyección).
    4. **Verdad canónica** (objetivo).

    El instante mostrado es el de máxima amplitud global de la verdad canónica
    (máxima varianza entre canales) para que las manchas sean visibles.
    """
    import tensorflow as tf

    from ..mapping import scalp_grid_matrix

    grid_px = grid_px or getattr(cfg.mapping, "grid_px", 48)
    idx = ds.split_idx[split][:n_samples]
    M_src, valid_s, _ = scalp_grid_matrix(montage_inputs.src_positions, grid_px)
    M_can, valid_c, _ = scalp_grid_matrix(np.asarray(ds.ch_positions), grid_px)
    px_s, py_s = _electrode_disc(montage_inputs.src_positions, grid_px)
    px_c, py_c = _electrode_disc(np.asarray(ds.ch_positions), grid_px)

    fig, axes = plt.subplots(len(KINDS), 4, figsize=(17, 4.2 * len(KINDS)))
    for row_k, kind in enumerate(KINDS):
        target = ds.refs[kind][idx]
        t = int(np.argmax(target.std(0)))
        obs = montage_inputs.src_refs[split][kind][:n_samples]
        pred = model(tf.convert_to_tensor(obs, tf.float32), source=kind)[kind].numpy()
        base = obs @ montage_inputs.projection

        ims = [
            _render_scalp_field(obs[t], M_src, valid_s, grid_px),
            _render_scalp_field(base[t], M_can, valid_c, grid_px),
            _render_scalp_field(pred[t], M_can, valid_c, grid_px),
            _render_scalp_field(target[t], M_can, valid_c, grid_px),
        ]
        titles = [f"Observación\nmontaje {montage_inputs.montage}",
                  "Proyección analítica", "Modelo", "Verdad canónica"]
        vmax = max(np.nanmax(np.abs(m)) for m in ims) or 1.0
        artist: plt.Figure | None = None
        for col, (im, title) in enumerate(zip(ims, titles)):
            ax = axes[row_k, col]
            im_ = ax.imshow(im, cmap="seismic", vmin=-vmax, vmax=vmax,
                            origin="upper")
            if col == 0:
                artist = im_
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{KIND_LABELS[kind]} — {title}", fontsize=9)
            px, py = (px_s, py_s) if col == 0 else (px_c, py_c)
            ax.scatter(px, py, s=6, color="k", marker="o", zorder=5, linewidths=0)
        fig.colorbar(artist, ax=axes[row_k, :3], shrink=0.75, pad=0.01)

    fig.suptitle(
        "Unificación de montajes: la actividad se estima desde el montaje "
        "fuente como manchas sobre el cuero cabelludo",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_scalp_heatmap(model, ds, montage_inputs, cfg, run_dir: Path,
                       split: str = "test", n_samples: int = 400) -> None:
    """Guarda el heatmap de montaje en ``runs/<dir>/figs/scalp_heatmap.png``."""
    fig = scalp_heatmap_fig(model, ds, montage_inputs, cfg, split=split,
                            n_samples=n_samples)
    fig.savefig(run_dir / "figs" / "scalp_heatmap.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)