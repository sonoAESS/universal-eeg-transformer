"""Tests de la dimensión temporal (universal_refs).

Cubre:

* ``build_multiconfig_windowed``: formas, contigüidad temporal y balanceo.
* ``TemporalResidualHead``: arranque idéntico al modelo instantáneo
  (capa final a cero) y actualización durante el entrenamiento.
* ``MultiHeatmapTemporal``: dispatch por variante, penalización de modos
  espaciales (l<=1) y round-trip de configuración.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from eeg_transform.config import EEGTransformConfig  # noqa: E402
from eeg_transform.experiments.multi import build_multiconfig  # noqa: E402
from eeg_transform.models.multi_heatmap import (  # noqa: E402
    MultiHeatmapAutoencoder,
    MultiHeatmapTemporal,
    mode_projector_matrix,
)
from eeg_transform.training.trainer import (  # noqa: E402
    build_multiconfig_model,
    build_multiconfig_windowed,
    count_windows,
)


def _sphere_positions(n, radius=0.088, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.35, np.pi - 0.35, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack(
        [radius * np.sin(th) * np.cos(ph),
         radius * np.sin(th) * np.sin(ph),
         radius * np.cos(th)], axis=1)


KINDS = ("unipolar", "linked_mastoids", "linked_ears", "bipolar",
         "car", "rest", "laplacian")


class _LeadField:
    def __init__(self, matrix):
        self.matrix = matrix


from dataclasses import dataclass  # noqa: E402


@dataclass
class _FakeDataset:
    n_channels: int
    ch_names: list
    ch_positions: np.ndarray
    leadfield: _LeadField
    refs: dict
    split_idx: dict
    meta: dict

    @classmethod
    def make(cls, n_samples=120, seed=0):
        names = [f"c{i}" for i in range(42)]
        names[20] = "Cz"
        rng = np.random.default_rng(seed)
        pos = _sphere_positions(42, seed=seed)
        lf = rng.normal(size=(42, 24)).astype(np.float32)
        # señal suave temporalmente (AR) para que las ventanas tengan estructura
        base = np.zeros((n_samples, 42), np.float32)
        z = rng.normal(size=(n_samples, 4)).astype(np.float32)
        for t in range(1, n_samples):
            base[t] = 0.95 * base[t - 1] + z[t] @ rng.normal(size=(4, 42)) * 0.1
        refs = {k: base.copy() for k in KINDS}
        idx = np.arange(n_samples)
        return cls(n_channels=42, ch_names=names, ch_positions=pos,
                   leadfield=_LeadField(lf), refs=refs,
                   split_idx={"train": idx[:80], "val": idx[80:100],
                              "test": idx[100:]},
                   meta={"unipolar_ref_ch": "Cz"})


def _multi_cfg(tmp_path, window=0, budget=60, stride=None, cell="conv"):
    cfg = EEGTransformConfig.from_dict({
        "dataset": {"cache_dir": str(tmp_path)},
        "leadfield": {"src_grid_mm": 30.0},
        "model": {"variant": "universal_refs", "latent_dim": 32,
                  "temporal_window": window,
                  "temporal_stride": stride if stride else (max(1, window // 2) if window else 1),
                  "temporal_channels": 16,
                  "temporal_layers": 1,
                  "temporal_kernel": 5,
                  "temporal_cell": cell,
                  "mode_penalty_weight": 0.1 if window else 0.0},
        "mapping": {
            "method": "spline",
            "configs": ["canonical"],
            "multi_max_samples_per_split": budget,
        },
    })
    return cfg


def test_count_windows():
    assert count_windows(100, 32, 1) == 69
    assert count_windows(32, 32, 1) == 1
    assert count_windows(10, 32, 1) == 0
    assert count_windows(101, 10, 10) == 10


def test_windowed_dataset_shapes_and_contiguity(tmp_path):
    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=8, stride=4)
    data = build_multiconfig(ds, cfg, force=True)

    dset = build_multiconfig_windowed(data, "train", batch_size=6,
                                      window=8, stride=4, shuffle=False)
    (x, y), = dset.take(1)
    arr = x["canonical"]["unipolar"].numpy()
    assert arr.shape == (6, 8, 42)

    # contigüidad contra el array presupuestado del config (lo ventaneado)
    canon_train = data["canonical"].refs["train"]
    assert np.allclose(arr[0], canon_train["unipolar"][:8], atol=1e-4)
    # ventana siguiente con stride 4: filas 4..12
    second = x["canonical"]["car"].numpy()[1]
    assert np.allclose(second, canon_train["car"][4:12], atol=1e-4)


@pytest.mark.parametrize("cell", ["conv", "gru", "lstm", "rnn"])
def test_temporal_head_zero_init_matches_instantaneous(tmp_path, cell):
    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=8, cell=cell)
    data = build_multiconfig(ds, cfg, force=True)
    model = build_multiconfig_model(cfg, data)

    assert isinstance(model, MultiHeatmapTemporal)
    core_built = model.core
    core_built.ensure_built()
    model(tf.zeros((2, 8, 19), tf.float32), cfg="10-20", source="rest") \
        if "10-20" in model.configs else None

    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 8, 42)).astype(np.float32)
    out_dyn = dict(model._predict_cfg(x, "canonical", "unipolar"))
    out_lin = MultiHeatmapAutoencoder._predict_cfg(model, x, "canonical",
                                                   "unipolar")
    for k in KINDS:
        assert np.allclose(out_dyn[k].numpy(), out_lin[k].numpy(), atol=1e-6), (
            f"con cabeza a cero la ruta {k} debe coincidir con el modelo "
            f"instantáneo (cell={cell})"
        )


@pytest.mark.parametrize("cell", ["conv", "gru", "lstm", "rnn"])
def test_residual_updates_after_train_step(tmp_path, cell):
    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=8, cell=cell)
    data = build_multiconfig(ds, cfg, force=True)
    model = build_multiconfig_model(cfg, data)
    model.core.ensure_built()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2))

    dset = build_multiconfig_windowed(data, "train", batch_size=8,
                                      window=8, stride=1)
    (x, y), = dset.take(1)

    # primer forward para materializar las variables de la cabeza (Keras 3
    # construye los pesos en el primer call, no en build explícito)
    model(x)

    head_vars_before = [v.numpy().copy()
                        for v in model.head.trainable_variables]
    assert head_vars_before, "la cabeza no materializó variables"

    out = model.train_step((x, y))
    assert "mode_loss" in out and "residual_norm" in out

    changed = any(
        not np.allclose(b, a.numpy())
        for b, a in zip(head_vars_before, model.head.trainable_variables)
    )
    assert changed, f"la cabeza temporal (cell={cell}) no se actualizó"


def test_mode_projector_annihilates_low_orders():
    pos = _sphere_positions(24, seed=5)
    p = mode_projector_matrix(pos)
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    modes = np.hstack([np.ones((24, 1)), unit]).T   # (4 modos, 24 ch)
    # cada modo proyectado por filas debe anularse
    assert np.abs(modes @ p).max() < 1e-6   # tolerancia float32
    # proyector idempotente y simétrico
    assert np.allclose(p @ p, p, atol=1e-6)
    assert np.allclose(p, p.T, atol=1e-6)


@pytest.mark.parametrize("cell", ["conv", "gru", "lstm", "rnn"])
def test_universal_refs_roundtrip(tmp_path, cell):
    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=8, cell=cell)
    data = build_multiconfig(ds, cfg, force=True)
    model = build_multiconfig_model(cfg, data)
    model.core.ensure_built()
    model.compile(optimizer="adam")

    restored = MultiHeatmapTemporal.from_config(model.get_config())
    assert restored.temporal_window == model.temporal_window
    assert restored.temporal_cell == model.temporal_cell
    assert restored.head.cell == model.head.cell
    assert set(restored.mode_projectors) == set(model.mode_projectors)


def test_spectral_band_table_selectivity():
    """La tabla espectral localiza el error en la banda correcta."""
    from eeg_transform.evaluation.metrics import spectral_band_table

    sfreq = 160.0
    t = np.arange(2000) / sfreq
    # true: alpha pura; pred: alpha con interferencia gamma grande y delta chica
    true = (np.sin(2 * np.pi * 10 * t)[:, None]
            * np.ones((1, 4))).astype(np.float64)
    pred = true + 0.3 * np.sin(2 * np.pi * 38 * t)[:, None] \
        + 0.05 * np.sin(2 * np.pi * 2 * t)[:, None]

    tabla = spectral_band_table(true, pred, sfreq=sfreq).set_index("banda")
    # el RMSE en gamma debe superar claramente al de bandas sin error
    assert tabla.loc["gamma", "rmse"] > 4 * tabla.loc["delta", "rmse"]
    assert tabla.loc["gamma", "rmse"] > 4 * tabla.loc["beta", "rmse"]
    # energía concentrada en alpha
    assert tabla.loc["alpha", "energia_frac"] > 0.8
    # predicción perfecta => ve ~1 y rmse ~0 en todas las bandas
    perfecta = spectral_band_table(true, true, sfreq=sfreq)
    assert perfecta["rmse"].max() < 1e-12


@pytest.mark.parametrize("cell", ["conv", "gru"])
def test_windowed_eval_shapes_and_alignment(tmp_path, cell):
    """El evaluador ventaneado alinea las predicciones y respeta la causalidad."""
    from eeg_transform.evaluation.metrics import (
        evaluate_multiconfig_windowed,
        predict_multiconfig_windowed,
    )

    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=8, stride=1, cell=cell)
    data = build_multiconfig(ds, cfg, force=True)
    model = build_multiconfig_model(cfg, data)
    model.core.ensure_built()

    test_n = ds.split_idx["test"].shape[0]
    window = 8
    expected = test_n - window + 1
    label = "canonical"

    preds = predict_multiconfig_windowed(model, data, split="test",
                                         window=window, stride=1)
    for s in KINDS:
        for d in KINDS:
            arr = preds[label][s][d]
            assert arr.shape[0] == expected, (cell, s, d, arr.shape)

    # cabeza a cero => la predicción ventaneada (residuo nulo) coincide con la
    # predicción instantánea del modelo sobre la muestra final de cada ventana
    target_idx = ds.split_idx["test"][window - 1:]
    ref_s = ds.refs["unipolar"][target_idx].astype(np.float32)
    inst = model(tf.convert_to_tensor(ref_s), cfg=label, source="unipolar")
    win = preds[label]["unipolar"]["unipolar"]
    assert np.allclose(win, inst["unipolar"].numpy(), atol=1e-6)

    # el DataFrame de métricas ventaneadas conserva el esquema por ruta
    rules = evaluate_multiconfig_windowed(model, data, split="test",
                                          window=window, stride=1)
    assert len(rules) == len(data.order) * len(KINDS) ** 2
    assert sorted(rules.columns) == sorted(
        ["config", "origen", "destino", "mse", "rmse", "mae", "r", "ve",
         "rmse_ana", "r_ana", "ve_ana"])


def test_windowed_eval_falls_back_when_window_0(tmp_path):
    """Con ventana 0 el evaluador ventaneado delega en el instantáneo."""
    from eeg_transform.evaluation import metrics
    from eeg_transform.evaluation.metrics import evaluate_multiconfig_windowed

    ds = _FakeDataset.make()
    cfg = _multi_cfg(tmp_path, window=0)
    data = build_multiconfig(ds, cfg, force=True)
    model = build_multiconfig_model(cfg, data)
    model.core.ensure_built()

    rules = evaluate_multiconfig_windowed(model, data, split="test", window=0)
    base = metrics.evaluate_multiconfig_routes(model, data, split="test")
    pd.testing.assert_frame_equal(rules, base)
