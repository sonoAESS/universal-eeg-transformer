"""Tests del vídeo-topomapa comparativo (``evaluation/topomap_video.py``).

Cubren la alineación de la ventana causal, las predicciones por método, el
cálculo de métricas por ventana y la construcción de la figura de animación
con datos sintéticos pequeños (sin descargas ni entrenamiento).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from eeg_transform.evaluation import topomap_video as tv
from eeg_transform.config import REFERENCE_KINDS
from eeg_transform.references import inter_reference_matrix


def _sphere_positions(n: int, radius: float = 0.088) -> np.ndarray:
    th = np.linspace(0.4, np.pi - 0.4, n)
    ph = np.linspace(0.0, 2 * np.pi, n)
    return np.stack(
        [radius * np.sin(th) * np.cos(ph),
         radius * np.sin(th) * np.sin(ph),
         radius * np.cos(th)], axis=1
    )


def _segment(L=200, n_ch=19, seed=0, rest_rcond=1e-4) -> tv.SegmentData:
    """Segmento con ``targets`` **consistentes** con la línea base analítica
    (``targets[d] = obs @ T_d``) usando las matrices reales del paquete."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(L, n_ch))
    pos = _sphere_positions(n_ch)
    lf = rng.normal(size=(n_ch, n_ch))
    targets = {
        d: obs @ inter_reference_matrix(
            "car", d, n_ch, unipolar_ref_index=0, lead_field=lf,
            rest_rcond=rest_rcond, positions=pos)
        for d in tv.TARGET_RFS
    }
    return tv.SegmentData(
        label="10-20", positions=pos, n_channels=n_ch,
        index=np.arange(L), obs=obs, targets=targets,
        unipolar_ref_index=0, leadfield=lf, rest_rcond=rest_rcond,
    )


class _FakeModel:
    """Sustituto de modelo multi-configuración: identidad en 2-D (instantáneo)
    y promedio de ventana en 3-D (temporal), emitiendo la misma salida para
    todos los destinos (comprobación de alineación)."""

    configs = ["10-20"]

    def __call__(self, x, cfg=None, source=None):
        if tf.rank(x) == 2:
            m = x
        else:
            m = tf.reduce_mean(x, axis=1, keepdims=True)
        return {d: m for d in REFERENCE_KINDS}


def test_predict_method_analytical_alignment_and_metrics():
    seg = _segment(L=120, n_ch=10, seed=0)
    w = 8
    res = tv.predict_method("ana", "analytical", None, "10-20", "car", seg, w)
    # ventanas causales exactas (y no cuelgan del futuro) → arrancan en 2·(w-1).
    assert res.frames[0] == 2 * (w - 1)
    assert res.frames[-1] == len(seg.obs) - 1
    assert res.preds["car"].shape == (len(seg.obs), 10)
    assert res.metrics["marco"].nunique() == len(res.frames)
    assert res.metrics["ruta"].nunique() == len(tv.TARGET_RFS)
    assert np.isfinite(res.metrics["rmse_uV"]).all()
    # el marco 0 apunta al instante k = 2·(w-1) del segmento.
    assert (res.metrics[res.metrics["marco"] == 0]["t"] == res.frames[0]).all()
    # instantes estrictamente crecientes.
    t = res.metrics.drop_duplicates("marco")["t"].to_numpy()
    assert (np.diff(t) > 0).all()


def test_analytical_route_nearly_exact_on_consistent_segment():
    # targets construidos con las mismas matrices → la estimación analítica
    # CAR→CAR debe dejar un RMSE de ventana despreciable.
    seg = _segment(L=90, n_ch=8, seed=3)
    res = tv.predict_method("ana", "analytical", None, "10-20", "car", seg, 6)
    raw = seg.targets["car"][6:].std() * 1e6
    rmse = res.metrics[res.metrics["ruta"] == "car"]["rmse_uV"]
    assert (rmse < 1e-4 * raw).all()


def test_predict_temporal_nan_warmup_and_same_frames():
    seg = _segment(L=80, n_ch=8, seed=1)
    w = 6
    res = tv.predict_method("temp", "temporal", _FakeModel(), "10-20", "car",
                            seg, w)
    # sin contexto causal previo → NaN en [0, w-2]; el resto finito.
    assert np.isnan(res.preds["car"][: w - 1]).all()
    assert np.isfinite(res.preds["car"][w - 1:]).all()
    # la alineación muestral de los marcos coincide con la instantánea.
    ref = tv.predict_method("ins", "instant", _FakeModel(), "10-20", "car",
                            seg, w)
    assert np.array_equal(res.frames, ref.frames)


def test_topomap_video_fig_builds_and_updates():
    seg = _segment(L=60, n_ch=8, seed=2)
    w = 5
    res_a = tv.predict_method("Analítico", "analytical", None, "10-20", "car",
                              seg, w)
    res_t = tv.predict_method("temp", "temporal", _FakeModel(), "10-20", "car",
                              seg, w)
    fig, update = tv.topomap_video_fig(seg, [res_a, res_t], grid_px=8,
                                       window=w, sfreq=128.0)
    artists = update(0)
    # una imagen + un texto por panel (2 métodos × 8 columnas) + barra + título.
    assert len(artists) == 2 * 8 * 2 + 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_select_segment_free_path():
    from types import SimpleNamespace

    n_c = 40
    refs = {k: np.random.default_rng(7).normal(size=(n_c, 6))
            for k in tv.TARGET_RFS}
    ds = SimpleNamespace(
        n_channels=6,
        ch_names=["Fz", "Cz", "Pz", "O1", "O2", "T7"],
        ch_positions=_sphere_positions(6),
        meta={"unipolar_ref_ch": "Cz"},
        refs=refs,
        split_idx={"test": np.arange(n_c)},
        leadfield=SimpleNamespace(matrix=np.eye(6)),
    )
    seg = tv.select_segment("canonical", "test", "unipolar", None, ds,
                            start=1000, duration_s=3.0, sfreq=10.0)
    # start desborda y se recorta; quedan 30 muestras, referenciadas a Cz.
    assert seg.obs.shape == (30, 6)
    assert seg.label == "canonical"
    assert seg.unipolar_ref_index == 1


def test_select_segment_multiconfig_path():
    rng = np.random.default_rng(8)

    class _MCConfig:
        positions = _sphere_positions(19)
        n_channels = 19
        unipolar_ref_index = 4
        leadfield = np.eye(19)
        rest_rcond = 1e-4
        refs = {"test": {k: rng.normal(size=(100, 19))
                         for k in tv.TARGET_RFS}}

    class _Data:
        configs = {"10-20": _MCConfig()}
        order = ["10-20"]

    seg = tv.select_segment("10-20", "test", "rest", _Data(), None,
                            start=0, duration_s=1.0, sfreq=10.0)
    assert seg.obs.shape == (10, 19)
    assert seg.rest_rcond == 1e-4


def test_run_topomap_video_validates_before_loading():
    # validaciones de entrada/soporte que no requieren datos ni modelos.
    with pytest.raises(ValueError):
        tv.run_topomap_video(models=[], cfg_label="canonical",
                             source="unipolar", duration_s=3.0, window=8)
    with pytest.raises(ValueError):
        tv.run_topomap_video(models=["free"], cfg_label="10-20",
                             source="unipolar", duration_s=3.0, window=8)
    with pytest.raises(ValueError):
        tv.run_topomap_video(models=["free"], cfg_label="otra",
                             source="unipolar", duration_s=3.0, window=8)