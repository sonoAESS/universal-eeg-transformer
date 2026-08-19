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
    has_real = {"mse_real_V2" in df, "val_mse_real_V2" in df}
    ncols = 2 if ("mse_real_V2" in df or "val_mse_real_V2" in df) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.6))
    if ncols == 1:
        axes = [axes]
    axes[0].plot(df["loss_estandarizada"], label="train",
                 color="rebeccapurple", lw=2)
    axes[0].plot(df["val_loss_estandarizada"], label="val", ls="--",
                 color="gold", lw=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida estandarizada")
    axes[0].set_title("Pérdida estandarizada por época")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(alpha=0.4)
    for ax in axes[1:]:
        ax.plot(df["mse_real_V2"], label="train", color="teal", lw=2)
        if "val_mse_real_V2" in df:
            ax.plot(df["val_mse_real_V2"], ls="--", color="coral", lw=2,
                    label="val")
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE real (V²)")
        ax.set_title("MSE real por época")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.4)
    fig.suptitle("Curvas de entrenamiento", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def heatmap_fig(metrics_df: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    """Heatmap log10 de una métrica por ruta origen→destino."""
    table = pd.DataFrame(index=KINDS, columns=KINDS, dtype=float)
    for _, row in metrics_df.iterrows():
        table.loc[row["origen"], row["destino"]] = row[value_col]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vals = np.log10(table.values.astype(float) + 1e-30)
    im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS])
    ax.set_yticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS])
    ax.set_xlabel("Destino"); ax.set_ylabel("Origen")
    ax.set_title(title, pad=10)
    for i in range(len(KINDS)):
        for j in range(len(KINDS)):
            cell = table.values[i, j]
            if not np.isfinite(cell):
                continue
            span = vals.max() - vals.min()
            bright = span > 0 and (vals[i, j] - vals.min()) / span > 0.55
            color = "white" if bright else "black"
            ax.text(j, i, f"{cell:.1e}", ha="center", va="center",
                    color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, label="log10(valor)")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def traces_fig(model, ds, split: str, channel: str, n_samples: int) -> plt.Figure:
    """Trazas reales vs predichas para las rutas cruzadas más relevantes.

    Las amplitudes se muestran en µV (los arrays internos están en voltios) y
    cada submódulo tiene leyenda propia para evitar el solapamiento de trazas
    entre paneles.
    """
    ch = ds.ch_names.index(channel)
    idx = ds.split_idx[split][:n_samples]
    refs = {k: ds.refs[k][idx] for k in KINDS}

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(17, 8), sharex=True)
    routes = [("unipolar", "rest"), ("bipolar", "rest"), ("car", "rest"),
              ("rest", "unipolar"), ("bipolar", "car"), ("car", "car")]
    import tensorflow as tf

    for ax, (s, d) in zip(axes.ravel(), routes):
        x = tf.convert_to_tensor(refs[s], tf.float32)
        pred = model(x, source=s)[d].numpy()
        t = np.arange(len(idx))
        real = refs[d][:, ch] * 1e6
        pred_uV = pred[:, ch] * 1e6
        ax.plot(t, real, color="black", lw=1.1, alpha=0.85, label="Real")
        ax.plot(t, pred_uV, color="dodgerblue", lw=1.3, ls=":", alpha=0.9,
                label="Predicción")
        ax.set_title(f"{KIND_LABELS[s]} → {KIND_LABELS[d]}", fontsize=10)
        ax.grid(alpha=0.35)
        lo, hi = np.percentile(real, [1, 99])
        margin = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - margin, hi + margin)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("t (muestras)")
    for ax in axes[:, 0]:
        ax.set_ylabel("µV")
    fig.suptitle(f"Canal {channel}: señales reales y predichas (test)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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

    fig, axes = plt.subplots(len(KINDS), 4,
                             figsize=(6.2 * 4, 3.6 * len(KINDS)),
                             constrained_layout=True)
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
        titles = [f"Observación · {montage_inputs.montage}",
                  "Proyección analítica", "Modelo", "Verdad canónica"]
        vmax = max(np.nanmax(np.abs(m)) for m in ims) or 1.0
        artist = None
        for col, (im, title) in enumerate(zip(ims, titles)):
            ax = axes[row_k, col]
            im_ = ax.imshow(im, cmap="seismic", vmin=-vmax, vmax=vmax,
                            origin="upper", interpolation="bicubic")
            if col == 0:
                artist = im_
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{KIND_LABELS[kind]} — {title}", fontsize=10, pad=6)
            if col == 0:
                px, py_ = px_s, py_s
            else:
                px, py_ = px_c, py_c
            ax.scatter(px, py_, s=8, color="k", marker="o", zorder=5,
                       linewidths=0.2, edgecolors="w")
        fig.colorbar(artist, ax=list(axes[row_k, :]), shrink=0.85, pad=0.015,
                     label="potencial (V)")
    for col, label in enumerate(["etapa 1: fuente", "etapa 2: P",
                                 "etapa 3: modelo", "etapa 4: objetivo"]):
        axes[-1, col].set_xlabel(label, fontsize=9)

    fig.suptitle(
        "Unificación de montajes: la actividad se estima desde el montaje "
        "fuente como manchas sobre el cuero cabelludo",
        fontsize=11, y=0.995,
    )
    return fig


def plot_scalp_heatmap(model, ds, montage_inputs, cfg, run_dir: Path,
                       split: str = "test", n_samples: int = 400) -> None:
    """Guarda el heatmap de montaje en ``runs/<dir>/figs/scalp_heatmap.png``."""
    fig = scalp_heatmap_fig(model, ds, montage_inputs, cfg, split=split,
                            n_samples=n_samples)
    fig.savefig(run_dir / "figs" / "scalp_heatmap.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-configuración (variante multi_montage)
# ---------------------------------------------------------------------------
def multiconfig_heatmap_fig(metrics_df: pd.DataFrame,
                            value_col: str = "rmse") -> plt.Figure:
    """Facets de heatmap RMSE (µV, log10) por configuración de electrodos."""
    labels = list(metrics_df["config"].unique())
    cols = 2
    rows = int(np.ceil(len(labels) / 2))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.2 * rows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, label in zip(axes, labels):
        sub = metrics_df[metrics_df["config"] == label]
        table = pd.DataFrame(index=KINDS, columns=KINDS, dtype=float)
        for _, row in sub.iterrows():
            table.loc[row["origen"], row["destino"]] = row[value_col] * 1e6
        vals = np.log10(table.values.astype(float) + 1e-30)
        im = ax.imshow(vals, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS],
                      fontsize=7)
        ax.set_yticks(range(len(KINDS)), [KIND_LABELS[k] for k in KINDS],
                      fontsize=7)
        ax.set_xlabel("Destino"); ax.set_ylabel("Origen", fontsize=8)
        ax.set_title(label, fontsize=10)
        for i in range(len(KINDS)):
            for j in range(len(KINDS)):
                val = table.values[i, j]
                if not np.isfinite(val):
                    continue
                color = "white" if vals[i, j] > (vals.min() + vals.max()) / 2 \
                    else "black"
                ax.text(j, i, f"{val:.1f}".replace(".", ","), ha="center",
                        va="center", color=color, fontsize=7)
        fig.colorbar(im, ax=ax, label="log10(RMSE µV)", pad=0.015)
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"RMSE por ruta y configuración ({value_col} → µV)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def multiconfig_bars_fig(metrics_df: pd.DataFrame) -> plt.Figure:
    """Barras agrupadas: RMSE cruzado (µV) modelo vs línea base analítica."""
    summ = []
    for label, g in metrics_df.groupby("config", sort=False):
        off = g[g["origen"] != g["destino"]]
        summ.append({
            "config": label,
            "Modelo": off["rmse"].mean() * 1e6,
            "Análitico (T_d pinv(T_s))": off["rmse_ana"].mean() * 1e6,
            "r_modelo": off["r"].mean(),
            "r_analitico": off["r_ana"].mean(),
        })
    df = pd.DataFrame(summ)
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.bar(x - width / 2, df["Modelo"], width, color="teal",
                label="Modelo")
    b2 = ax.bar(x + width / 2, df["Análitico (T_d pinv(T_s))"], width,
                color="coral", label="Analítico")
    ax.set_xticks(x, df["config"])
    ax.set_ylabel("RMSE cruzado (µV)")
    ax.set_title("Conversión de referencias intra-configuración")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.4)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", xy=(b.get_x() + b.get_width() / 2,
                                                     b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    for i, r in enumerate(df["r_modelo"]):
        ax.text(i - width / 2, df["Modelo"][i] + 1, f"r={r:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def multiconfig_scalp_fig(
    model,
    data,
    cfg_label: str,
    split: str = "test",
    n_samples: int = 400,
    grid_px: int = 48,
) -> plt.Figure:
    """Heatmaps del cuero cabelludo intra-configuración (Obs → Modelo → Verdad).

    Cada fila es una referencia; las columnas muestran la observación del
    montaje, la predicción del modelo y la verdad canónica, todo **en la propia
    configuración** (C_s electrodos). El instante es el de máxima varianza de
    la verdad.
    """
    import tensorflow as tf

    from ..mapping import scalp_grid_matrix

    mc = data.configs[cfg_label]
    M_cfg, valid, _ = scalp_grid_matrix(mc.positions, grid_px)
    px, py = _electrode_disc(mc.positions, grid_px)
    n = min(n_samples, mc.refs[split]["rest"].shape[0])

    fig, axes = plt.subplots(len(KINDS), 3,
                             figsize=(6.2 * 3, 3.6 * len(KINDS)),
                             constrained_layout=True)
    for row_k, kind in enumerate(KINDS):
        target = mc.refs[split][kind][:n]
        t = int(np.argmax(target.std(0)))
        obs = mc.refs[split][kind][:n]
        pred = model(tf.convert_to_tensor(obs, tf.float32), cfg=cfg_label,
                     source=kind)[kind].numpy()
        ims = [
            _render_scalp_field(obs[t], M_cfg, valid, grid_px),
            _render_scalp_field(pred[t], M_cfg, valid, grid_px),
            _render_scalp_field(target[t], M_cfg, valid, grid_px),
        ]
        titles = ["Observación", "Modelo", "Verdad canónica"]
        vmax = max(np.nanmax(np.abs(m)) for m in ims) or 1.0
        artist = None
        for col, (im, title) in enumerate(zip(ims, titles)):
            ax = axes[row_k, col]
            im_ = ax.imshow(im, cmap="seismic", vmin=-vmax, vmax=vmax,
                            origin="upper", interpolation="bicubic")
            if col == 0:
                artist = im_
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{KIND_LABELS[kind]} — {title}", fontsize=10, pad=6)
            ax.scatter(px, py, s=8, color="k", marker="o", zorder=5,
                       linewidths=0.2, edgecolors="w")
        fig.colorbar(artist, ax=list(axes[row_k, :]), shrink=0.85, pad=0.015,
                     label="potencial (V)")
    fig.suptitle(
        f"Conversión de referencias en la configuración '{cfg_label}' "
        f"({mc.n_channels} electrodos)",
        fontsize=11, y=0.995,
    )
    return fig


def plot_multiconfig_heatmap(metrics_df: pd.DataFrame, run_dir: Path) -> None:
    _save(multiconfig_heatmap_fig(metrics_df), run_dir, "multiconfig_rmse.png")


def plot_multiconfig_bars(metrics_df: pd.DataFrame, run_dir: Path) -> None:
    _save(multiconfig_bars_fig(metrics_df), run_dir, "multiconfig_bars.png")


def plot_multiconfig_scalps(model, data, run_dir: Path,
                            split: str = "test") -> None:
    """Guardar un scalp heatmap por configuración no-canónica."""
    for label in data.order:
        if label == "canonical":
            continue
        fig = multiconfig_scalp_fig(model, data, label, split=split)
        fig.savefig(run_dir / "figs" / f"scalp_{label}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)