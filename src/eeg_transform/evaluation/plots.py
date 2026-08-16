"""Figuras de evaluación: curvas de aprendizaje, matrices de error y trazas."""

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


def _save(fig, run_dir: Path, name: str) -> None:
    out = run_dir / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Figura guardada: %s", out / name)


def plot_learning_curves(history_csv: str | Path, run_dir: Path) -> None:
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
    _save(fig, run_dir, "learning.png")


def plot_heatmap(metrics_df: pd.DataFrame, value_col: str, title: str,
                 run_dir: Path, fname: str) -> None:
    table = pd.DataFrame(
        index=KINDS, columns=KINDS, dtype=float
    )
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
    _save(fig, run_dir, fname)


def plot_traces(
    model,
    ds,
    split: str,
    channel: str,
    n_samples: int,
    run_dir: Path,
) -> None:
    """Trazas reales vs predichas para las rutas cruzadas más relevantes."""
    ch = ds.ch_names.index(channel)
    idx = ds.split_idx[split][:n_samples]
    refs = {k: ds.refs[k][idx] for k in KINDS}

    chart = {"unipolar": "car", "bipolar": "rest", "car": "rest"}
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
    _save(fig, run_dir, "traces.png")