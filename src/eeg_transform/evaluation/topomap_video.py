"""Vídeo-topomapa comparativo: entrada observada → estimaciones por método.

Genera una animación (``matplotlib.animation`` → GIF/MP4) donde cada marco
muestra, sobre la configuración de electrodos elegida (canónica 64 ó ``10-20``),
el topomapa de la **referencia observada** y, a su lado, los topomapas de las
referencias estimadas por cada método:

* **Analítico** — el mejor estimador lineal sin aprendizaje: ``T_d·pinv(T_s)``
  (:func:`~eeg_transform.references.inter_reference_matrix`).
* **Modelos entrenados** — ``free``, ``multi_montage``, ``multi_heatmap_v2`` y
  ``universal_refs`` (cabezas temporal ``conv`` y ``gru``) reentrenados sobre
  las 7 referencias del esquema actual.

Cada panel de estimación incluye las métricas de la ruta **en la ventana causal**
que termina en el instante mostrado (RMSE µV, VE, r), calculadas con la misma
convención de :func:`~eeg_transform.evaluation.metrics._route_stats` (subespacio
observable centrado por instante).

Los métodos temporales se evalúan con inferencia causal ``stride=1`` (cada
instante predice desde su propia ventana de contexto) para que las métricas de
ventana sean comparables con las instantáneas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from ..config import EEGTransformConfig, REFERENCE_KINDS
from ..logging_conf import get_logger
from ..mapping import scalp_grid_matrix
from ..references import inter_reference_matrix
from .metrics import _route_stats
from .plots import _electrode_disc, _render_scalp_field

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registro de métodos y constantes
# ---------------------------------------------------------------------------

#: nombre del método → ruta del YAML con el que se (re)entrena el modelo.
#: ``universal_refs_conv/gru`` reutilizan los YAML del barrido comparativo
#: (ya entrenados, esquema de 7 refs, sujetos 1-2) para no volver a entrenar.
VIDEO_MODELS: dict[str, str] = {
    "free": "config/topomap_video_free.yaml",
    "multi_montage": "config/topomap_video_multi_montage.yaml",
    "multi_heatmap_v2": "config/topomap_video_multi_heatmap_v2.yaml",
    "universal_refs_conv": "config/universal_refs_cmp_conv.yaml",
    "universal_refs_gru": "config/universal_refs_cmp_gru.yaml",
}

KIND_LABELS_ES = {
    "unipolar": "Unipolar",
    "linked_mastoids": "Mastoides lig.",
    "linked_ears": "Orejas lig.",
    "bipolar": "Bipolar",
    "car": "CAR",
    "rest": "REST",
    "laplacian": "Laplaciano",
}

#: orden estable de las columnas de referencias estimadas.
TARGET_RFS = list(REFERENCE_KINDS)

#: configuraciones de electrodos soportadas como entrada del vídeo.
INPUT_CONFIGS = ("canonical", "10-20")

#: ``rest_rcond`` por defecto para la línea base analítica del dataset suelto.
_REST_RCOND_DEFAULT = 1e-4


@dataclass
class SegmentData:
    """Ventana temporal contigua extraída de una configuración.

    ``index`` son los índices globales del split; ``obs`` y ``targets`` viven
    en esa rebanada (``L`` muestras). ``targets[d]`` es la verdad canónica de
    la referencia ``d`` en la propia configuración.
    """

    label: str
    positions: np.ndarray
    n_channels: int
    index: np.ndarray
    obs: np.ndarray                       # (L, C_s) fuente observada
    targets: dict[str, np.ndarray]        # {d: (L, C_s)}
    unipolar_ref_index: int
    leadfield: np.ndarray
    rest_rcond: float | None


@dataclass
class MethodResult:
    """Predicciones y métricas de un método sobre un segmento.

    ``preds[d]`` está alineado a **muestras** del segmento ``(L, C_s)``; los
    instantes previos a ``window-1`` (sin contexto causal) quedan en ``NaN``
    para los métodos temporales. ``metrics`` es un ``DataFrame`` con una fila
    por (marco, instante, ruta) y las métricas de la ventana causal
    ``[t-window+1, t]``.
    """

    name: str
    frames: np.ndarray                    # índices locales de marco (L_f,)
    preds: dict[str, np.ndarray]          # {d: (L, C_s)} alineado a muestras
    metrics: pd.DataFrame                 # marco, t, instante_s, ruta, rmse_uV...


# ---------------------------------------------------------------------------
# Carga / entrenamiento de los métodos
# ---------------------------------------------------------------------------

def _load_checkpoint(model, path: Path) -> None:
    """Carga pesos; si el checksum de variables falla (drift entre códigos,
    p. ej. ``_v2_params`` de ``MultiHeatmapTemporal``), reintenta con
    ``skip_mismatch``. Las variables omitidas solo participan en pérdidas de
    entrenamiento, no en la predicción del vídeo."""
    try:
        model.load_weights(str(path))
    except ValueError as e:
        log.warning("Carga estricta fallida (%s). Reintentando con "
                    "skip_mismatch=True.", e)
        model.load_weights(str(path), skip_mismatch=True)


def load_config_path(name: str) -> EEGTransformConfig:
    """Carga el YAML de entrenamiento del método (``VIDEO_MODELS``)."""
    from ..config import load_config

    return load_config(Path(VIDEO_MODELS[name]))


def load_methods(names: list[str], force: bool = False) -> dict[str, tuple]:
    """Carga (o entrena) cada modelo pedido.

    Returns
    -------
    ``{nombre: (cfg, ds, data, model)}`` con ``model`` cargado con los mejores
    pesos; ``data`` es el conjunto multi-configuración o ``None`` para la
    variante ``free`` (que consume el dataset directo).
    """
    from ..data.dataset import build_dataset
    from ..training.trainer import (
        build_model,
        is_multiconfig_variant,
        load_multiconfig_data,
        train,
    )

    out: dict[str, tuple] = {}
    for name in names:
        if name not in VIDEO_MODELS:
            raise ValueError(f"Método desconocido '{name}' (válidos: "
                             f"{sorted(VIDEO_MODELS)}).")
        cfg = load_config_path(name)
        ds = build_dataset(cfg)
        checkpoint = Path(cfg.training.run_dir) / "best.weights.h5"
        if is_multiconfig_variant(cfg):
            data = load_multiconfig_data(cfg, ds)
            if not force and checkpoint.exists():
                from ..training.trainer import build_multiconfig_model

                model = build_multiconfig_model(cfg, data)
                model.core.ensure_built()
                _load_checkpoint(model, checkpoint)
            else:
                model, _ = train(ds, cfg, force=True)
            out[name] = (cfg, ds, data, model)
        else:
            if not force and checkpoint.exists():
                n = max(1, ds.n_channels)
                model = build_model(cfg, n)
                model.ensure_built()
                _load_checkpoint(model, checkpoint)
            else:
                model, _ = train(ds, cfg, force=True)
            out[name] = (cfg, ds, None, model)
        log.info("Método %s listo.", name)
    return out


# ---------------------------------------------------------------------------
# Segmento y predicciones
# ---------------------------------------------------------------------------

def select_segment(
    cfg_label: str,
    split: str,
    source: str,
    data,
    ds,
    start: int,
    duration_s: float,
    sfreq: float = 160.0,
) -> SegmentData:
    """Extrae un segmento contiguo de ``duration_s`` segundos del split.

    ``data`` es el conjunto multi-configuración (usa su configuración
    ``cfg_label``) o ``None`` (variante ``free``, que lee del ``ds``). Los
    índices se recortan al rango de muestras disponible del split.
    """
    if data is not None:
        mc = data.configs[cfg_label]
        n = mc.refs[split][source].shape[0]
        positions = np.asarray(mc.positions, dtype=np.float64)
        n_channels = mc.n_channels
        unipolar_ref_index = mc.unipolar_ref_index
        leadfield = mc.leadfield.astype(np.float64)
        rest_rcond = mc.rest_rcond
    else:
        n = len(ds.split_idx[split])
        positions = np.asarray(ds.ch_positions, dtype=np.float64)
        n_channels = ds.n_channels
        unipolar_ref_index = ds.ch_names.index(
            ds.meta.get("unipolar_ref_ch", "Cz"))
        leadfield = ds.leadfield.matrix.astype(np.float64)
        rest_rcond = _REST_RCOND_DEFAULT

    length = max(1, int(round(duration_s * sfreq)))
    start = max(0, min(int(start), max(0, n - length)))
    seg_idx = np.arange(start, start + min(length, n - start))

    if data is not None:
        mc = data.configs[cfg_label]
        obs = mc.refs[split][source][seg_idx].astype(np.float64)
        targets = {
            d: mc.refs[split][d][seg_idx].astype(np.float64)
            for d in TARGET_RFS
        }
        index = seg_idx
    else:
        idx = ds.split_idx[split][seg_idx]
        obs = ds.refs[source][idx].astype(np.float64)
        targets = {d: ds.refs[d][idx].astype(np.float64) for d in TARGET_RFS}
        index = idx

    return SegmentData(
        label=cfg_label, positions=positions, n_channels=n_channels,
        index=index, obs=obs, targets=targets,
        unipolar_ref_index=unipolar_ref_index, leadfield=leadfield,
        rest_rcond=rest_rcond,
    )


def _analytical_routes(seg: SegmentData, source: str) -> dict[str, np.ndarray]:
    """``obs @ T_d·pinv(T_s)`` por destino — estimación lineal sin aprendizaje."""
    return {
        d: seg.obs @ inter_reference_matrix(
            source, d, seg.n_channels,
            unipolar_ref_index=seg.unipolar_ref_index,
            lead_field=seg.leadfield, rest_rcond=seg.rest_rcond,
            positions=seg.positions,
        )
        for d in TARGET_RFS
    }


def _predict_instant(model, seg: SegmentData, cfg_label: str,
                     source: str) -> dict[str, np.ndarray]:
    """Predicción instantánea (tensor 2-D) de todos los destinos."""
    x = tf.convert_to_tensor(seg.obs.astype(np.float32), dtype=tf.float32)
    if cfg_label and getattr(model, "configs", None):
        preds = model(x, cfg=cfg_label, source=source)
    else:
        preds = model(x, source=source)
    return {d: np.asarray(preds[d]).astype(np.float64) for d in TARGET_RFS}


def _predict_causal(model, seg: SegmentData, cfg_label: str, source: str,
                    window: int
                    ) -> dict[str, np.ndarray]:
    """Predicción causal ``stride=1`` alineada a **muestras** del segmento.

    Devuelve un dict de arrays ``(L, C_s)``; los instantes sin contexto
    completo (``< window-1``) quedan como ``NaN``. La salida del instante ``t``
    es el último paso de la ventana que termina en ``t``.
    """
    wins = np.lib.stride_tricks.sliding_window_view(seg.obs, window, axis=0)
    x = tf.convert_to_tensor(
        wins.transpose(0, 2, 1).astype(np.float32), dtype=tf.float32)
    preds_all = model(x, cfg=cfg_label, source=source)
    L = len(seg.obs)
    out = {}
    for d in TARGET_RFS:
        seq = np.full((L, seg.n_channels), np.nan, dtype=np.float64)
        seq[window - 1:] = np.asarray(preds_all[d])[:, -1, :]
        out[d] = seq
    return out


def predict_method(
    name: str,
    method_kind: str,
    model,
    cfg_label: str,
    source: str,
    seg: SegmentData,
    window: int,
    sfreq: float = 160.0,
) -> MethodResult:
    """Predice con un método y calcula las métricas de ventana causal.

    ``method_kind`` en ``{"analytical", "instant", "temporal"}``. Todas las
    predicciones quedan alineadas a **muestras** del segmento. Los marcos de la
    animación arrancan en ``2·(window-1)`` para que cada ventana causal tenga
    exactamente ``window`` muestras con predicción (los temporales precisan
    contexto previo). Las métricas de cada ruta usan :func:`_route_stats`.
    """
    if method_kind == "analytical":
        preds = _analytical_routes(seg, source)
    elif method_kind == "temporal":
        preds = _predict_causal(model, seg, cfg_label, source, window)
    else:
        preds = _predict_instant(model, seg, cfg_label, source)

    frames = np.arange(2 * max(0, window - 1), len(seg.obs))

    rows = []
    for f, k in enumerate(frames):
        k = int(k)
        win = np.arange(k - window + 1, k + 1)
        for d in TARGET_RFS:
            st = _route_stats(seg.targets[d][win], preds[d][win])
            rows.append({
                "marco": int(f), "t": k, "instante_s": k / sfreq, "ruta": d,
                "rmse_uV": st["rmse"] * 1e6, "r": st["r"], "ve": st["ve"],
            })
    metrics = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["marco", "t", "instante_s", "ruta", "rmse_uV", "r", "ve"])
    return MethodResult(name=name, frames=frames, preds=preds, metrics=metrics)


# ---------------------------------------------------------------------------
# Figura / animación
# ---------------------------------------------------------------------------

def _settick(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def topomap_video_fig(
    seg: SegmentData,
    results: list[MethodResult],
    grid_px: int = 64,
    vmax_frac: float = 0.995,
    sfreq: float = 160.0,
    window: int = 64,
):
    """Construye la figura y el callback de actualización de la animación.

    Filas = métodos; columnas = [entrada observada] + referencias estimadas.
    Devuelve ``(fig, update)`` donde ``update(f)`` refresca los topomapas y las
    métricas del marco ``f``. Cada columna (referencia) normaliza con su propio
    percentil ``vmax_frac``, estable entre marcos, para que se vea la actividad
    de cada referencia sin que las de mayor amplitud laven a las demás.
    """

    n_rows = len(results)
    n_cols = 1 + len(TARGET_RFS)
    fig_width = n_cols * 2.4 + 0.9
    fig_height = n_rows * 2.3 + 0.6

    if n_rows == 1:
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height),
                                 constrained_layout=True)
        axes = axes[None, :]
    else:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(fig_width, fig_height),
                                 constrained_layout=True)

    M, valid, _ = scalp_grid_matrix(seg.positions, grid_px)
    px, py = _electrode_disc(seg.positions, grid_px)

    # Escala de color POR columna (referencia): una única escala para las 7
    # referencias lava las de menor amplitud (unipolar/bipolar/CAR frente a
    # REST/Laplaciano), y los NaN del calentamiento temporal invalidaban el
    # percentil global. Cada columna usa su p(vmax_frac) estable en el tiempo,
    # con un suelo de 10 µV para las columnas planas.
    def _column_vmax(values: list[np.ndarray]) -> float:
        if not values:
            return 1e-5
        stack = np.concatenate([np.abs(v.reshape(-1)) for v in values])
        vm = float(np.nanpercentile(stack, 100.0 * vmax_frac))
        if not np.isfinite(vm) or vm <= 0:
            vm = 1e-5
        return vm

    entry_vmax = _column_vmax([seg.obs])
    kind_vmax = {kind: _column_vmax([r.preds[kind] for r in results])
                 for kind in TARGET_RFS}
    column_vmax = [entry_vmax] + [kind_vmax[k] for k in TARGET_RFS]

    ims: list[list] = []
    texts: list[list] = []
    for row, res in enumerate(results):
        row_ims = []
        row_texts = []
        ax0 = axes[row, 0]
        _settick(ax0)
        ax0.set_ylabel(res.name, fontsize=8, rotation=90, labelpad=2)
        for col in range(n_cols):
            ax = axes[row, col]
            im = ax.imshow(np.full((grid_px, grid_px), np.nan), cmap="RdBu_r",
                           vmin=-column_vmax[col], vmax=column_vmax[col],
                           origin="upper", interpolation="bicubic")
            _settick(ax)
            ax.scatter(px, py, s=5, color="k", marker="o", zorder=5,
                       linewidths=0.15, edgecolors="w")
            row_ims.append(im)
            caption = ax.text(0.5, -0.015, "", transform=ax.transAxes,
                              fontsize=6.5, ha="center", va="top")
            row_texts.append(caption)
            if col == 0:
                ax.set_title(f"entrada\n{seg.label} · {seg.obs.shape[1]} elec.",
                             fontsize=8, pad=5)
        ims.append(row_ims)
        texts.append(row_texts)

    # cabecera de columnas (referencias) con nombres en español y la escala de
    # cada columna para que la normalización por referencia sea explícita.
    axes[0, 0].set_title(f"entrada · {seg.obs.shape[1]} elec.\n"
                         f"±{entry_vmax * 1e6:.0f} µV", fontsize=7, pad=10)
    for col, kind in enumerate(TARGET_RFS, start=1):
        axes[0, col].set_title(f"{KIND_LABELS_ES.get(kind, kind)}\n"
                               f"±{kind_vmax[kind] * 1e6:.0f} µV",
                               fontsize=7, pad=10)

    ax_prog = fig.add_axes([0.58, 0.005, 0.3, 0.012])
    _settick(ax_prog)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.set_title("progreso", fontsize=6, pad=1)
    bar, = ax_prog.plot([0, 1], [0, 1], color="crimson", lw=3,
                        solid_capstyle="butt")

    title = fig.text(
        0.5, 0.99,
        f"Conversión de referencias · configuración '{seg.label}' "
        f"({seg.n_channels} elec.) · observada: {seg.obs.shape[0]} muestras",
        ha="center", fontsize=10)

    n_frames = len(results[0].frames)
    frame_range = results[0].frames

    def update(f: int):
        k = int(frame_range[f])
        artists = []
        for row, res in enumerate(results):
            # panel de entrada: observado en el instante mostrado
            ims[row][0].set_data(_render_scalp_field(
                seg.obs[k:k + 1][0], M, valid, grid_px))
            texts[row][0].set_text(f"t={k} ({k / sfreq:.2f} s)")
            artists += [ims[row][0], texts[row][0]]
            for col, kind in enumerate(TARGET_RFS, start=1):
                # marco f ↔ instante k por construcción (frames alineados).
                sub = res.metrics[(res.metrics["marco"] == f)
                                  & (res.metrics["ruta"] == kind)]
                if len(sub):
                    m = sub.iloc[0]
                    cap = f"{m['rmse_uV']:.2f} µV · r={m['r']:.2f} " \
                          f"· VE={m['ve']:.2f}"
                else:
                    cap = ""
                field = _render_scalp_field(res.preds[kind][k], M, valid,
                                            grid_px)
                ims[row][col].set_data(field)
                texts[row][col].set_text(cap)
                artists += [ims[row][col], texts[row][col]]
        t_frac = k / max(1, frame_range[-1])
        bar.set_data([0, t_frac], [0, 1])
        artists.append(bar)
        title.set_text(
            f"Conversión de referencias · configuración '{seg.label}' "
            f"({seg.n_channels} elec.) · t={k} ({k / sfreq:.2f} s) · "
            f"ventana causal {window} muestras")
        artists.append(title)
        return tuple(artists)

    fig._topomap_n_frames = n_frames
    return fig, update


def save_topomap_video(fig, update, n_frames: int, out: str | Path,
                       fps: int = 15, format: str = "gif") -> Path:
    """Serializa la animación a GIF (Pillow) o MP4 (ffmpeg)."""
    from matplotlib.animation import FuncAnimation

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    if format == "mp4":
        out_path = out.with_suffix(".mp4")
        writer = "ffmpeg"
    else:
        out_path = out.with_suffix(".gif")
        writer = "pillow"
    anim.save(str(out_path), writer=writer, fps=fps, dpi=110)
    plt.close(fig)
    log.info("Vídeo guardado: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def run_topomap_video(
    models: list[str],
    cfg_label: str,
    source: str,
    duration_s: float,
    window: int,
    start: int = 0,
    sfreq: float = 160.0,
    grid_px: int = 64,
    fps: int = 15,
    formats: tuple[str, ...] = ("gif",),
    out_dir: str | Path = "runs/topomap_video/figs",
    force_train: bool = False,
    frame: int | None = None,
) -> dict:
    """Genera el vídeo-topomapa comparativo completo.

    Entrena (o reutiliza) los modelos pedidos, selecciona el segmento de
    ``duration_s`` segundos del split ``test`` y serializa la animación. Si se
    da ``frame``, en su lugar guarda una imagen estática PNG del instante
    ``frame`` (índice de muestra, acotado al rango válido). Devuelve
    ``{"figuras": {"gif": Path, ...}, "metrics": Path}``.
    """
    from .plots import set_plot_backend

    set_plot_backend("agg")

    if cfg_label not in INPUT_CONFIGS:
        raise ValueError(f"Entrada no soportada: {cfg_label!r} "
                         f"(válidas: {INPUT_CONFIGS}).")
    if not models:
        raise ValueError("Debe pedirse al menos un modelo "
                         "(la línea base analítica siempre se incluye).")

    # soporte de entrada limitado por arquitectura: free y universal_refs solo
    # se entrenaron en la configuración canónica.
    only_canonical = {"free", "universal_refs_conv", "universal_refs_gru"}
    for name in models:
        if name not in VIDEO_MODELS:
            raise ValueError(f"Método desconocido '{name}' (válidos: "
                             f"{sorted(VIDEO_MODELS)}).")
        if name in only_canonical and cfg_label != "canonical":
            raise ValueError(f"'{name}' solo soporta la entrada 'canonical' "
                             f"(no '{cfg_label}').")

    loaded = load_methods(models, force=force_train)
    first = loaded[models[0]]

    for name in models:
        _cfg, _ds, data, _model = loaded[name]
        if data is not None and cfg_label not in data.order:
            raise ValueError(
                f"{name!r} no soporta la entrada '{cfg_label}' "
                f"(configs: {data.order}).")

    seg = select_segment(
        cfg_label=cfg_label, split="test", source=source,
        data=first[2], ds=first[1],
        start=start, duration_s=duration_s, sfreq=sfreq,
    )
    if len(seg.obs) <= 2 * max(0, window - 1):
        raise ValueError("duration_s demasiado corto para la ventana causal "
                         f"({len(seg.obs)} ≤ {2 * max(0, window - 1)} "
                         "muestras).")

    kind_for = {
        "free": "instant", "multi_montage": "instant",
        "multi_heatmap_v2": "instant",
        "universal_refs_conv": "temporal", "universal_refs_gru": "temporal",
    }

    results: list[MethodResult] = [predict_method(
        "Analítico", "analytical", None, cfg_label, source, seg, window,
        sfreq=sfreq)]
    for name in models:
        _cfg, _ds, _data, model = loaded[name]
        results.append(predict_method(
            name, kind_for[name], model, cfg_label, source, seg, window,
            sfreq=sfreq))

    n_frames = len(results[0].frames)
    out_dir = Path(out_dir)
    fname = f"topomap_video_{cfg_label}_{source}"
    fig, update = topomap_video_fig(seg, results, grid_px=grid_px,
                                sfreq=sfreq, window=window)

    saved: dict[str, Path] = {}
    if frame is not None:
        k0 = int(results[0].frames[0])
        k = max(k0, min(int(frame), int(results[0].frames[-1])))
        update(int(k - k0))
        png_path = out_dir / f"{fname}_t{k}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        log.info("Imagen estática: %s (instante %d)", png_path, k)
        saved["png"] = png_path
    else:
        for fmt in formats:
            saved[fmt] = save_topomap_video(fig, update, n_frames,
                                            out_dir / fname, fps=fps,
                                            format=fmt)

    metrics_all = pd.concat(
        [r.metrics.assign(method=r.name) for r in results], ignore_index=True)
    metrics_path = out_dir / f"{fname}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_all.to_csv(metrics_path, index=False)
    log.info("Métricas por ventana: %s", metrics_path)
    return {"figuras": saved, "metrics": metrics_path}